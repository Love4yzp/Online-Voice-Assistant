#!/usr/bin/env python3
"""Live weather tool for the Jarvis dialogue (no API key required).

Uses https://wttr.in compact format and returns a short line in the dialogue
language, e.g. "杭州: 小雨, +23°C, 湿度86%, 风16km/h".
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from ova.lang import normalise

LOG = logging.getLogger("dialogue.weather")

# OpenAI-style function definition advertised to Qwen.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "query_weather",
        "description": "查询指定城市当前的实时天气。城市用英文名，"
                       "如 Hangzhou / Shanghai / Beijing；若用户没说明城市，默认 Hangzhou",
        "parameters": {
            "type": "object",
            "properties": {
                "city_slug": {
                    "type": "string",
                    "description": "城市英文名，如 Hangzhou",
                }
            },
            "required": ["city_slug"],
        },
    },
}

CITY_ZH = {
    "Hangzhou": "杭州", "Shanghai": "上海", "Beijing": "北京",
    "Shenzhen": "深圳", "Guangzhou": "广州", "Nanjing": "南京",
    "Suzhou": "苏州", "Chengdu": "成都", "Wuhan": "武汉",
    "Xi'an": "西安", "Chongqing": "重庆", "Tianjin": "天津",
}


def query_weather(city_slug: str, timeout: float = 12.0,
                  lang: str | None = None) -> str:
    """Fetch current weather; return a short line in the chosen language.

    Tries twice (transient network hiccups are common from the robot).
    """
    slug = (city_slug or "Hangzhou").strip()
    english = normalise(lang) == "en"
    query = urllib.parse.urlencode({
        "format": "%l: %C, %t, humidity %h, wind %w" if english else
                  "%l: %C, %t, 湿度%h, 风%w",
        "lang": "en" if english else "zh",
    })
    url = "https://wttr.in/" + urllib.parse.quote(slug) + "?" + query
    last_exc = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace").strip()
            if not text:
                raise RuntimeError("empty weather reply")
            city = slug if english else CITY_ZH.get(slug, slug)
            line = text.replace(slug, city, 1) if text.startswith(slug) else f"{city}: {text}"
            LOG.info("WEATHER city=%s lang=%s -> %s", slug, "en" if english else "zh", line)
            return line
        except Exception as exc:  # noqa: BLE001 - retry on any fetch failure
            last_exc = exc
            LOG.warning("WEATHER attempt %d failed for %s: %s", attempt, slug, exc)
            time.sleep(0.5)
    raise RuntimeError(f"weather fetch failed: {last_exc}")


def load_tool_calls(message: dict):
    """Extract (name, args_dict) list from an assistant tool_calls message."""
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        out.append((name, args, call.get("id", "")))
    return out
