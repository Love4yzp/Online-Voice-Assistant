#!/usr/bin/env python3
"""Engine abstraction tests — no network, no audio device, no wake model.

Run with pytest or directly:

    python3 tests/test_engines.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:      # allow running without `pip install -e .`
    sys.path.insert(0, str(SRC))

os.environ.setdefault("OVA_HOME", str(ROOT))

from ova.engines import EngineError, build_engine          # noqa: E402
from ova.engines import pipeline as pipe                   # noqa: E402
from ova.engines.base import ENGINE_ALIASES                # noqa: E402
from ova.llm import (  # noqa: E402
    EN_SYSTEM_PROMPT, SYSTEM_PROMPT, CloudError, system_prompt,
)


@contextlib.contextmanager
def patch(obj, **attrs):
    """Temporarily replace module attributes (keeps the real ones intact)."""
    saved = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


def _fake_wav_bytes() -> bytes:
    """A tiny valid 16 kHz stereo WAV, standing in for qwen3-tts output."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    return buf.getvalue()


# --- factory -----------------------------------------------------------------

def test_default_engine_is_pipeline():
    engine = build_engine({}, asr=object())
    assert engine.name == "pipeline"
    assert engine.needs_transcript is True


def test_engine_aliases_are_accepted():
    assert ENGINE_ALIASES["e2e"] == "e2e"
    assert ENGINE_ALIASES["glm-4-voice"] == "e2e"
    assert ENGINE_ALIASES["端到端"] == "e2e"
    assert ENGINE_ALIASES["pipeline"] == "pipeline"
    assert ENGINE_ALIASES["半在线"] == "pipeline"


def test_unknown_engine_raises_with_hint():
    try:
        build_engine({"engine": "gpt4o-realtime"})
    except EngineError as exc:
        assert "pipeline" in str(exc) and "e2e" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected EngineError for an unknown engine")


def test_case_and_whitespace_tolerated():
    engine = build_engine({"engine": "  Pipeline  "}, asr=object())
    assert engine.name == "pipeline"


# --- pipeline helpers --------------------------------------------------------

def test_clip_passes_short_text_through():
    assert pipe.clip("你好呀") == "你好呀"


def test_clip_cuts_at_sentence_boundary():
    text = "第一句话在这里。" + "第二句话很长很长很长" * 12
    out = pipe.clip(text, maxlen=40)
    assert len(out) <= 41
    assert out.endswith("。")


def test_clip_english_does_not_append_chinese_punctuation():
    text = "A reply without any sentence boundary that is deliberately long " * 4
    assert pipe.clip(text, maxlen=40).endswith(".")


def test_weather_tool_follows_dialogue_language_without_network():
    from ova import tools
    from urllib.parse import parse_qs, urlsplit

    seen = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
        def read(self):
            return b"Hangzhou: Clear, +23C, humidity 70%, wind 5km/h"

    def urlopen(url, timeout=12.0):
        seen.append(parse_qs(urlsplit(url).query))
        return Response()

    with patch(tools.urllib.request, urlopen=urlopen):
        assert "humidity" in tools.query_weather("Hangzhou", lang="en")
        assert "杭州" in tools.query_weather("Hangzhou")
    assert seen[0]["lang"] == ["en"]
    assert seen[1]["lang"] == ["zh"]


def test_ask_with_weather_plain_reply():
    with patch(pipe, chat_once=lambda messages, tools=None, timeout=30.0: {
            "content": "杭州今天小雨。"}):
        assert pipe.ask_with_weather("杭州天气") == "杭州今天小雨。"


def test_ask_with_weather_runs_tool_round():
    calls = []

    def fake_chat(messages, tools=None, timeout=30.0):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{
                "id": "call_1",
                "function": {"name": "query_weather",
                             "arguments": '{"city_slug": "Hangzhou"}'}}]}
        return {"content": "杭州 23 度，小雨。"}

    with patch(pipe, chat_once=fake_chat,
               query_weather=lambda slug, lang=None: f"{slug}: 小雨 23C"):
        reply = pipe.ask_with_weather("杭州天气怎么样")
    assert reply == "杭州 23 度，小雨。"
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "tool"
    assert "小雨" in calls[1][-1]["content"]


def test_ask_with_weather_empty_reply_raises():
    with patch(pipe, chat_once=lambda messages, tools=None, timeout=30.0: {"content": "  "}):
        try:
            pipe.ask_with_weather("随便说点什么")
        except CloudError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected CloudError on empty reply")


# --- reply language ----------------------------------------------------------

def test_system_prompt_follows_the_reply_language():
    """中文口径原样保留；英文用独立英文人设（旋钮长按切到 English）。"""
    assert system_prompt(None) == SYSTEM_PROMPT
    assert system_prompt("zh") == SYSTEM_PROMPT
    assert system_prompt(" 中文 ") == SYSTEM_PROMPT
    assert system_prompt("en") == EN_SYSTEM_PROMPT
    for alias in ("en", "ENGLISH", "英文"):                # 大小写/别名
        prompt = system_prompt(alias)
        assert "Always answer in English" in prompt
        assert "用中文" not in prompt                       # 不再和中文指令并存
        assert "query_weather" in prompt                    # 天气工具要求仍在
    assert "query_weather" in system_prompt(None)           # 中文分支不受影响


def test_ask_with_weather_uses_the_reply_language():
    calls = []

    def fake_chat(messages, tools=None, timeout=30.0):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "", "tool_calls": [{
                "id": "call_1",
                "function": {"name": "query_weather",
                             "arguments": '{"city_slug": "Hangzhou"}'}}]}
        return {"content": "Hangzhou, 23C, light rain."}

    weather_langs = []
    with patch(pipe, chat_once=fake_chat,
               query_weather=lambda slug, lang=None: weather_langs.append(lang) or f"{slug}: light rain 23C"):
        reply = pipe.ask_with_weather("杭州天气怎么样", "en")
    assert weather_langs == ["en"]
    assert reply == "Hangzhou, 23C, light rain."
    assert calls[0][0]["content"] == EN_SYSTEM_PROMPT   # 英文人设（含天气要求）
    assert calls[1][0] == calls[0][0]        # 工具轮复用同一份 system message


def test_english_reply_is_retried_when_it_comes_back_in_chinese():
    """EN 模式下模型仍回中文时，自动再要一次英文（只重试一次）。"""
    calls = []

    def fake_chat(messages, tools=None, timeout=30.0):
        calls.append(messages)
        if len(calls) == 1:
            return {"content": "智慧空间是融合物联网的智能环境。"}
        return {"content": "The Smart Space is an intelligent environment."}

    with patch(pipe, chat_once=fake_chat):
        reply = pipe.ask_with_weather("介绍一下智慧空间", "en")
    assert reply == "The Smart Space is an intelligent environment."
    assert len(calls) == 2
    assert calls[1][-1]["content"] == pipe.LANG_RETRY_MESSAGE
    assert calls[1][-2] == {"role": "assistant",
                            "content": "智慧空间是融合物联网的智能环境。"}


def test_language_guard_leaves_zh_and_mostly_english_alone():
    """中文模式不重问；英文回答里夹一个中文词也不算跑偏。"""
    calls = []

    def zh_chat(messages, tools=None, timeout=30.0):
        calls.append(messages)
        return {"content": "你好呀，有什么我可以帮你的吗？"}

    with patch(pipe, chat_once=zh_chat):
        assert pipe.ask_with_weather("你好", "zh").startswith("你好呀")
    assert len(calls) == 1                  # zh 模式不回炉

    calls.clear()

    def en_chat(messages, tools=None, timeout=30.0):
        calls.append(messages)
        return {"content": "The Smart Space (智慧空间) uses IoT and AI."}

    with patch(pipe, chat_once=en_chat):
        reply = pipe.ask_with_weather("介绍一下智慧空间", "en")
    assert reply == "The Smart Space (智慧空间) uses IoT and AI."
    assert len(calls) == 1                  # 中文占比低，不触发重试


# --- pipeline respond --------------------------------------------------------

def test_respond_returns_playable_reply(tmp_path):
    engine = build_engine({"engine": "pipeline"}, asr=object())
    with patch(pipe,
               ask_with_weather=lambda text, lang=None: "欢迎来到智慧零售区。",
               synthesize=lambda text, timeout=60.0: _fake_wav_bytes()):
        reply = engine.respond(np.zeros(16000, dtype=np.int16), "介绍智慧零售")

    assert reply.text == "欢迎来到智慧零售区。"
    assert reply.transcript == "介绍智慧零售"
    assert reply.temporary is True          # playback loop deletes it afterwards
    assert reply.timeout_s == 90.0
    assert reply.meta["engine"] == "pipeline"
    assert reply.meta["provider"] == "qwen"
    assert reply.audio_path.is_file()
    with wave.open(str(reply.audio_path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 2
    reply.audio_path.unlink()


def test_respond_maps_chat_failure_to_engine_error():
    engine = build_engine({"engine": "pipeline"}, asr=object())

    def boom(text, lang=None):
        raise CloudError("HTTP 429: rate limited")

    with patch(pipe, ask_with_weather=boom):
        try:
            engine.respond(np.zeros(1600, dtype=np.int16), "你好")
        except EngineError as exc:
            assert "chat failed" in str(exc) and "429" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected EngineError")


def test_respond_maps_tts_failure_to_engine_error():
    engine = build_engine({"engine": "pipeline"}, asr=object())

    def boom(text, timeout=60.0):
        raise CloudError("tts audio download failed")

    with patch(pipe,
               ask_with_weather=lambda text, lang=None: "好的。",
               synthesize=boom):
        try:
            engine.respond(np.zeros(1600, dtype=np.int16), "你好")
        except EngineError as exc:
            assert "tts failed" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected EngineError")


def test_respond_hands_the_dialogue_language_to_the_chat_call():
    """编排层放进 cfg 的 reply_lang 会一路传到千问的 system message。"""
    engine = build_engine({"engine": "pipeline"}, asr=object())
    seen: list[list[dict]] = []

    def fake_chat(messages, tools=None, timeout=30.0):
        seen.append(messages)
        return {"content": "Sure."}

    with patch(pipe, chat_once=fake_chat,
               synthesize=lambda text, timeout=60.0: _fake_wav_bytes()):
        english = engine.respond(np.zeros(1600, dtype=np.int16), "hello",
                                 {"reply_lang": "en"})
        chinese = engine.respond(np.zeros(1600, dtype=np.int16), "你好",
                                 {"reply_lang": "zh"})

    assert english.text == "Sure." and chinese.text == "Sure."
    assert seen[0][0]["content"] == EN_SYSTEM_PROMPT   # 英文人设
    assert seen[1][0]["content"] == SYSTEM_PROMPT      # 中文仍是原口径
    for reply in (english, chinese):        # 同毫秒的两个回复可能同名 tmp 文件
        reply.audio_path.unlink(missing_ok=True)


# --- direct runner (no pytest required) -------------------------------------

def _main() -> int:
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        kwargs = {}
        if "tmp_path" in test.__code__.co_varnames[: test.__code__.co_argcount]:
            import tempfile
            ctx = tempfile.TemporaryDirectory()
            kwargs["tmp_path"] = Path(ctx.name)
        try:
            test(**kwargs)
        except Exception:  # noqa: BLE001 - report and keep going
            failed += 1
            print(f"FAIL: {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS: {test.__name__}")
    print(f"\n{'FAILED' if failed else 'OK'}: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
