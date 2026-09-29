"""Qwen chat client (OpenAI-compatible) with optional function calling."""

import os

from ova.api import CloudError, http_json  # noqa: F401 (re-export)
from ova.lang import normalise as normalise_lang

CHAT_URL = os.getenv("CHAT_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-flash")

SYSTEM_PROMPT = os.getenv(
    "QWEN_SYSTEM_PROMPT",
    "你是语音助手 Jarvis。用户询问天气、气温、下雨等问题时，必须调用 "
    "query_weather 工具查实时数据后再回答；其他问题（攻略、常识、闲聊等）"
    "直接正常回答，不要说自己是受限的或只能查天气。"
    "用中文口语化回答，尽量60字以内，直接给答案，不要markdown或列表。",
)

# English dialogue language (POST /lang, 展厅旋钮长按). ``system_prompt("en")``
# returns a **full English persona** instead of appending one English sentence to
# the Chinese one: the Chinese ``用中文口语化回答`` contradicts it, and qwen-flash
# then follows the question's language — English only stuck for English
# questions (2026-09-29 实测 2/2 中文提问得到中文回答)，切英文形同虚设。
# 去掉中文指令后同样的问题 2/2 得到英文回答。
EN_INSTRUCTION = ("Always answer in English, even when the question is in "
                  "Chinese. Keep it conversational and under 60 words; give "
                  "the answer directly, no markdown or lists.")

EN_SYSTEM_PROMPT = os.getenv(
    "QWEN_SYSTEM_PROMPT_EN",
    "You are Jarvis, the voice assistant of this robot. When the user asks "
    "about weather, temperature or rain you must call the query_weather tool "
    "first and answer from the live data; for anything else (guides, general "
    "knowledge, small talk) answer normally — never claim you are limited to "
    "weather. " + EN_INSTRUCTION,
)


def system_prompt(lang: str | None = None) -> str:
    """The system prompt for one reply language.

    ``zh``/None keep ``SYSTEM_PROMPT`` (Chinese persona, overridable with
    ``QWEN_SYSTEM_PROMPT``) byte for byte; ``lang="en"`` (or
    ``english``/``eng``/``英文``) returns ``EN_SYSTEM_PROMPT`` (overridable
    with ``QWEN_SYSTEM_PROMPT_EN``). See the note on ``EN_INSTRUCTION`` for
    why English does not reuse the Chinese persona.
    """
    if normalise_lang(lang) == "en":
        return EN_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def chat_once(messages: list[dict], tools=None, timeout: float = 30.0) -> dict:
    """One chat completion; returns the raw assistant message dict."""
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "enable_thinking": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    data = http_json(CHAT_URL, payload, timeout)
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CloudError(f"unexpected chat response: {str(data)[:200]}") from exc


def chat(question: str, timeout: float = 30.0) -> str:
    """Plain chat (no tools); returns the assistant text reply."""
    message = chat_once(
        [{"role": "system", "content": system_prompt()},
         {"role": "user", "content": question}],
        timeout=timeout,
    )
    text = (message.get("content") or "").strip()
    if not text:
        raise CloudError("empty assistant reply")
    return text
