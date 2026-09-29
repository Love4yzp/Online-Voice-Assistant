#!/usr/bin/env python3
"""Semi-online engine: local ASR text -> Qwen chat (tools) -> Qwen TTS.

This is the behaviour the project has always had, moved behind the Engine
interface unchanged: the same log lines, the same console events ("llm" and
"tts" cards) and the same /tmp reply file naming.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from ova.config import svc_event
from ova.engines.base import EngineError, Reply
from ova.lang import normalise as normalise_lang
from ova.llm import CloudError, chat_once, system_prompt
from ova.tools import WEATHER_TOOL, load_tool_calls, query_weather
from ova.tts import synthesize

LOG = logging.getLogger("dialogue.pipeline")


def clip(text: str, maxlen: int = 130) -> str:
    """Keep replies short for spoken delivery: cut at a sentence boundary."""
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen]
    for sep in ("。", "！", "？"):
        idx = cut.rfind(sep)
        if idx > maxlen // 2:
            cut = cut[:idx + 1]
            break
    else:
        cut = cut + "." if cut.isascii() else cut + "。"
    LOG.info("REPLY_CLIPPED len=%d -> %d", len(text), len(cut))
    return cut


CJK_RE = re.compile(r"[\u3400-\u9fff]")
LANG_RETRY_MESSAGE = "Answer again in English only, with no Chinese characters."
# 英文模式下小模型偶尔仍用中文作答（2026-09-29：中文提问 + 中文产品名时复现），
# 回答里中文占比超过这个比例就再要一次英文；只重试一次，失败就用重试结果。
LANG_RETRY_CJK_RATIO = 0.2


def ensure_reply_language(reply: str, lang: str | None,
                          messages: list[dict]) -> str:
    """Re-ask once when an English turn still comes back in Chinese.

    ``messages`` 是这一轮已经用过的对话（含工具结果），重试只追加一轮
    assistant/user 提醒，不让模型重新跑工具。
    """
    if not reply or normalise_lang(lang) != "en":
        return reply
    ratio = len(CJK_RE.findall(reply)) / len(reply)
    if ratio < LANG_RETRY_CJK_RATIO:    # 偶尔夹一个中文词不算跑偏
        return reply
    LOG.warning("LANG_RETRY reply not in English (cjk=%.2f), asking once more",
                ratio)
    try:
        message = chat_once(messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": LANG_RETRY_MESSAGE},
        ])
    except CloudError as exc:
        LOG.warning("LANG_RETRY failed: %s", exc)
        return reply
    return (message.get("content") or "").strip() or reply


def ask_with_weather(text: str, lang: str | None = None) -> str:
    """Qwen with a weather tool: answer plain, or fetch live weather and
    compose a short spoken summary from the fetched facts.

    ``lang`` is the dialogue language of this turn (``reply_lang``); ``en``
    asks for an English answer (regenerated once if the model ignores it).
    The same system message is reused for the tool round, so the weather
    summary keeps the requested language too.
    """
    messages = [
        {"role": "system", "content": system_prompt(lang)},
        {"role": "user", "content": text},
    ]
    first = chat_once(messages, tools=[WEATHER_TOOL])
    calls = load_tool_calls(first)
    if not calls:
        reply = (first.get("content") or "").strip()
        if not reply:
            raise CloudError("empty assistant reply")
        return ensure_reply_language(reply, lang, messages)
    # Tool round: run every requested tool, then ask Qwen to compose.
    messages.append(first)
    for name, args, call_id in calls:
        if name == "query_weather":
            try:
                result = query_weather(args.get("city_slug", "Hangzhou"), lang=lang)
            except Exception as exc:  # noqa: BLE001 - report failure to the model
                result = (f"Weather lookup failed: {exc}" if normalise_lang(lang) == "en"
                          else f"天气查询失败: {exc}")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
    second = chat_once(messages)
    reply = (second.get("content") or "").strip()
    if not reply:
        raise CloudError("empty assistant reply after tool call")
    LOG.info("QWEN_TOOL_ROUND city=%s", ", ".join(
        str(a.get("city_slug")) for _, a, _ in calls))
    return ensure_reply_language(reply, lang, messages)


class PipelineEngine:
    """Local ASR + Qwen chat + Qwen TTS (needs the ASR transcript)."""

    name = "pipeline"
    needs_transcript = True

    def __init__(self, asr=None, cfg: dict | None = None):
        self.asr = asr
        self.cfg = cfg or {}

    def respond(self, samples, text: str, cfg: dict | None = None) -> Reply:
        cfg = cfg or self.cfg
        t0 = time.monotonic()
        # 对话语言由编排层透传（cfg["reply_lang"]，展厅旋钮长按切换）
        reply_lang = cfg.get("reply_lang")
        try:
            reply_text = clip(ask_with_weather(text, reply_lang))
        except CloudError as exc:
            raise EngineError(f"chat failed: {exc}") from exc
        LOG.info("QWEN_REPLY lang=%s text=%s", reply_lang or "-", reply_text[:80])
        svc_event("llm", f"千问回答: {reply_text[:80]}", "ok", text=reply_text[:120])

        try:
            wav = synthesize(reply_text)
        except CloudError as exc:
            raise EngineError(f"tts failed: {exc}") from exc

        tmp = Path(f"/tmp/hjw_reply_{os.getpid()}_{int(time.time() * 1000)}.wav")
        tmp.write_bytes(wav)
        LOG.info("REPLY_START")
        svc_event("tts", "播放回答…", "info")
        return Reply(
            audio_path=tmp,
            text=reply_text,
            transcript=text,
            timeout_s=90.0,
            temporary=True,
            meta={
                "engine": self.name,
                "provider": "qwen",
                "elapsed_s": round(time.monotonic() - t0, 2),
            },
        )
