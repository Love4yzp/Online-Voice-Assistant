#!/usr/bin/env python3
"""Orchestration tests: routing + engine hand-off without hardware or network.

`ova.dialogue` imports `ova.wake`, which imports the openWakeWord runtime at
module level. The runtime is not needed for these tests (nothing under test
scores audio), so it is stubbed before the import — the same trick used when
running the pipeline on a laptop.

Run with pytest or directly:

    python3 tests/test_dialogue_routing.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("OVA_HOME", str(ROOT))

# --- stub openwakeword (imported by ova.wake, unused by these tests) ---------
if "openwakeword" not in sys.modules:
    _oww = types.ModuleType("openwakeword")
    _oww_model = types.ModuleType("openwakeword.model")

    class _Model:  # noqa: D401 - minimal stand-in
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            return {}

        def reset(self):
            pass

    _oww_model.Model = _Model
    _oww.model = _oww_model
    sys.modules["openwakeword"] = _oww
    sys.modules["openwakeword.model"] = _oww_model

from ova.dialogue import (                                   # noqa: E402
    BargeCommand,
    _is_continue_command,
    _is_stop_command,
    _playback_from_text,
    play_asset,
    run_dialogue_round,
)
from ova.engines.base import EngineError, Reply               # noqa: E402

ASSETS = ROOT / "assets"


class FakeBackend:
    output_device = "fake_sink"

    def __init__(self):
        self.played: list[Path] = []

    def play_file(self, path, timeout=None):
        self.played.append(Path(path))

    def played_names(self) -> list[str]:
        return [p.name for p in self.played]


class FakeEngine:
    """Records what it was asked; returns a Reply pointing at a real WAV."""

    def __init__(self, needs_transcript=False, reply: Reply | None = None,
                 error: Exception | None = None):
        self.name = "fake"
        self.needs_transcript = needs_transcript
        self.calls: list[tuple] = []
        self._reply = reply
        self._error = error

    def respond(self, samples, text, cfg):
        self.calls.append((samples, text))
        if self._error is not None:
            raise self._error
        if self._reply is not None:
            return self._reply
        wav = ROOT / "tests" / "asr_zh_smart_retail.wav"
        return Reply(audio_path=wav, text="好的。", transcript=text,
                     timeout_s=42.0, temporary=False,
                     meta={"engine": self.name})


# --- local command routing (shared by both engines) --------------------------

def test_stop_and_continue_word_lists():
    assert _is_stop_command("停止吧") and _is_stop_command("Stop")
    assert _is_continue_command("继续") and _is_continue_command("go on")
    assert not _is_stop_command("智慧零售")
    assert not _is_continue_command("介绍应急救灾")


def test_pipeline_stop_command_goes_idle_without_engine_call():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="停止",
                                samples=np.zeros(1600, dtype=np.int16), engine=engine)
    assert state is None
    assert engine.calls == []


def test_pipeline_routes_showroom_intro_to_local_wav():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="介绍智慧零售",
                                samples=np.zeros(1600, dtype=np.int16), engine=engine)
    assert state is not None
    assert state.name == "智慧零售讲解"
    assert state.path.is_file()
    assert state.path.name == "smart_retail_zh.wav"
    assert engine.calls == []          # intros never reach the engine (no cloud cost)


def test_pipeline_english_intro_routes_to_english_wav():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={},
                                text="introduce Smart Retail",
                                samples=np.zeros(1600, dtype=np.int16), engine=engine)
    assert state is not None
    assert state.path.name == "smart_retail_en.wav"


def test_continue_command_resumes_previous_playback():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    previous = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="介绍智慧零售",
                                   samples=np.zeros(1600, dtype=np.int16), engine=engine)
    resumed = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="继续",
                                  samples=np.zeros(1600, dtype=np.int16),
                                  previous=previous, engine=engine)
    assert resumed is previous
    assert engine.calls == []


# --- engine hand-off ---------------------------------------------------------

def test_pipeline_engine_receives_transcript_and_ack_plays_first():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    cfg = {"ack_before_reply": True}
    samples = np.zeros(1600, dtype=np.int16)
    state = _playback_from_text(backend, ASSETS, asr=None, cfg=cfg, text="今天天气怎么样",
                                samples=samples, engine=engine)
    assert state is not None and state.text == "好的。"
    assert state.timeout_s == 42.0 and state.cleanup is False
    assert len(engine.calls) == 1 and engine.calls[0][1] == "今天天气怎么样"
    assert backend.played_names() == ["ack_think.wav"]
    assert state.name == "回答"


def test_ack_can_be_disabled():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=True)
    _playback_from_text(backend, ASSETS, asr=None, cfg={"ack_before_reply": False},
                        text="你好", samples=np.zeros(1600, dtype=np.int16), engine=engine)
    assert backend.played == []


def test_e2e_engine_skips_routing_and_gets_audio_only():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=False)
    samples = np.arange(1600, dtype=np.int16)
    # text="" because no local ASR ran; the stop/intro branches must be skipped
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="",
                                samples=samples, engine=engine)
    assert state is not None
    assert len(engine.calls) == 1
    got_samples, got_text = engine.calls[0]
    assert got_text == ""
    assert np.array_equal(got_samples, samples)


def test_e2e_engine_without_audio_plays_fallback():
    backend, engine = FakeBackend(), FakeEngine(needs_transcript=False)
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={}, text="",
                                samples=None, engine=engine)
    assert state is None
    assert engine.calls == []
    assert backend.played_names() == ["fallback_question.wav"]


def test_engine_error_plays_network_fallback():
    backend = FakeBackend()
    engine = FakeEngine(needs_transcript=True,
                        error=EngineError("HTTP 429: rate limited"))
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={"ack_before_reply": False},
                                text="你好", samples=np.zeros(1600, dtype=np.int16),
                                engine=engine)
    assert state is None
    assert backend.played_names() == ["fallback_net.wav"]


def test_unexpected_engine_bug_is_contained():
    backend = FakeBackend()
    engine = FakeEngine(needs_transcript=False, error=ValueError("boom"))
    state = _playback_from_text(backend, ASSETS, asr=None, cfg={"ack_before_reply": False},
                                text="", samples=np.zeros(1600, dtype=np.int16),
                                engine=engine)
    assert state is None
    assert backend.played_names() == ["fallback_net.wav"]


def test_fallback_follows_current_language_and_explicit_override(tmp_path):
    root = tmp_path / "assets"
    folder = root / "fallback"
    folder.mkdir(parents=True)
    (folder / "fallback_net.wav").write_bytes(b"zh")
    (folder / "fallback_net_en.wav").write_bytes(b"en")
    backend = FakeBackend()
    cfg = {"dialogue_lang_file": str(tmp_path / "lang.state")}
    lang_state = tmp_path / "lang.state"
    # Test with the real language state reader, including the default.
    lang_state.write_text("en\n")
    play_asset(backend, root, "fallback_net.wav", cfg)
    play_asset(backend, root, "fallback_net.wav", cfg, lang="zh")
    lang_state.write_text("zh\n")
    play_asset(backend, root, "fallback_net.wav", cfg)
    (folder / "fallback_net_en.wav").unlink()
    lang_state.write_text("en\n")
    play_asset(backend, root, "fallback_net.wav", cfg)
    assert backend.played_names() == [
        "fallback_net_en.wav", "fallback_net.wav", "fallback_net.wav", "fallback_net.wav"]


# --- round-level wiring ------------------------------------------------------

def test_run_dialogue_round_plays_fallback_when_nothing_recorded(monkeypatch=None):
    """No speech twice in a row -> two prompts, no engine call, no exception."""
    import ova.dialogue as dlg

    backend, engine = FakeBackend(), FakeEngine(needs_transcript=False)
    original = dlg.listen_question
    dlg.listen_question = lambda *a, **k: None
    try:
        run_dialogue_round(backend, ASSETS, asr=None, cfg={}, engine=engine)
    finally:
        dlg.listen_question = original
    assert backend.played_names() == ["fallback_listen.wav", "fallback_question.wav"]
    assert engine.calls == []


def test_run_dialogue_round_transcribes_only_when_needed():
    import ova.dialogue as dlg

    class FakeAsr:
        def __init__(self):
            self.calls = 0

        def transcribe(self, samples):
            self.calls += 1
            return "介绍智慧零售"

    backend = FakeBackend()
    original_listen, original_loop = dlg.listen_question, dlg._run_playback_loop
    dlg.listen_question = lambda *a, **k: np.zeros(1600, dtype=np.int16)
    dlg._run_playback_loop = lambda *a, **k: None
    try:
        # pipeline: uses the local ASR transcript
        asr = FakeAsr()
        engine = FakeEngine(needs_transcript=True)
        run_dialogue_round(backend, ASSETS, asr=asr, cfg={}, engine=engine)
        assert asr.calls == 1

        # e2e: must NOT pay the local ASR cost
        asr2 = FakeAsr()
        engine2 = FakeEngine(needs_transcript=False)
        run_dialogue_round(backend, ASSETS, asr=asr2, cfg={}, engine=engine2)
        assert asr2.calls == 0
    finally:
        dlg.listen_question, dlg._run_playback_loop = original_listen, original_loop


def test_barge_command_dataclass_defaults():
    command = BargeCommand()
    assert command.samples is None and command.text == ""


# --- direct runner (no pytest required) -------------------------------------

def _main() -> int:
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
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
