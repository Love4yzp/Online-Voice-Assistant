# 对话语言切换（2026-09-15）

## 目标

给语音对话加一个**持久化的对话语言**：展厅键盘（另一个项目 button-bridge）的旋钮**长按**，
就把这台机器人的对话语言在中 ↔ 英之间切换，**对整个对话生效**——不是"这一次请求用英文"，
而是此后每一次讲解与回答都按这个语言走。

- 复用现有入口：语言切换仍由 `ova-wake` 进程内的 stdlib HTTP 入口提供，只加端点，不加依赖。
- 复用现有讲解与引擎：切换结果接到**讲解选版**（`config/solutions.json` 的 `zh/en` 两套 WAV）
  与 **LLM 回答语言**（千问 system prompt）上，不新增第二条链路。
- 状态落一个状态文件，`ova-wake` 重启后仍在（进程重启不丢；整机重启看文件是否在 `/tmp`）。
- 动机：现场访客有中英两种，讲解员不该为了换语言去改配置或重启服务。

链路（新增部分用 `→` 标出）：

```text
POST /lang {} / {"toggle":true} / {"lang":"en"}  → lang.set_lang() → /tmp/ova_lang.state
GET  /lang                                       → lang.current() → {"ok":true,"lang":"zh"}
                                                        │
页面上注入文本 / 语音识别文本 ──────────────────────────┤ 每次 _playback_from_text()
                                                        ▼
                        effective = 显式 lang（/inject 带的）or 当前对话语言
                             ├─ 讲解选版   → smart_retail_zh.wav / smart_retail_en.wav
                             └─ 普通问答   → engine.respond(..., {**cfg, "reply_lang": effective})
                                              └─ pipeline: 千问 system prompt 跟随语言
```

## 接口

两个端点都由 `ova-wake` 进程提供，默认只监听 `127.0.0.1:8090`（与 `/inject`、`/wake` 同一个监听）。

### `POST /lang`

```json
请求: {}                       // 旋钮长按：切换（zh ↔ en）
请求: {"toggle": true}         // 同上，显式写法
请求: {"lang": "en"}           // 直接设置；接受 en/eng/english/英文(→en) 与 zh/cn/chinese/中文(→zh)

响应: {"ok": true, "lang": "en", "previous": "zh"}
```

- 大小写与空白都忽略（`" ENGLISH "`、`" 中文 "` 都合法）。
- 非法值（`jp`、`""`、`null`…）返回 **400**：`{"ok": false, "error": "unsupported lang 'jp'; expected one of zh, en"}`，
  并且**不会**写脏状态文件。
- 成功即写状态文件、写一条日志与调试台事件（见下），随后所有轮次都按新语言走。

### `GET /lang`

```json
响应: {"ok": true, "lang": "zh"}      // 当前生效的语言
```

其它路径仍是 404 JSON（`POST`/`GET` 都是），未知路径与坏 body 的行为没有变化。

```bash
curl -s -X POST http://127.0.0.1:8090/lang                    # 切换
curl -s -X POST http://127.0.0.1:8090/lang -d '{"lang":"en"}' # 切到英文
curl -s -X POST http://127.0.0.1:8090/lang -d '{"lang":"zh"}' # 切回中文
curl -s http://127.0.0.1:8090/lang                            # 读当前语言
```

## 语言怎么影响一轮对话

| 环节 | 行为 |
|---|---|
| 讲解选版 | 显式 `lang`（`POST /inject {...,"lang":"en"}`）优先；否则用**现场切换过的语言**（状态文件，或特意配成非默认的 `dialogue_lang`）；都没指定时**仍按关键词语言**（`介绍一下智慧零售`→中文、`introduce smart retail`→英文），与语音链路的历史行为一致 |
| LLM 回答（`engine=pipeline`） | `reply_lang` 透传给引擎；`en` 时 system prompt 换成**独立英文人设** `EN_SYSTEM_PROMPT`（不再在中文人设后追加英文句，见下文“为什么换人设”），`zh`/未设定保持原中文口径 byte for byte |
| 英文兜底 | EN 模式下模型仍回中文（中文占比 ≥ `LANG_RETRY_CJK_RATIO=0.2`）时，`ensure_reply_language()` 追加一轮 assistant/user 提醒重问一次；只重试一次，失败就用重试结果，日志 `LANG_RETRY` |
| 天气工具轮 | 复用同一份 system message，所以工具查完后的总结也是同一语言 |
| 语音链路 | 与注入文本共用 `_playback_from_text()`，所以"切到英文后说中文关键词"同样会播英文讲解 |
| 固定兜底音 | `dialogue.play_asset()` 按当前或注入请求显式语言选择 `fallback/*_en.wav`；找不到英文文件会记录警告并播放中文原件。没听清、换个问题、网络异常和可选“好的”缓冲音都有英文录音 |
| `engine=e2e`（GLM-4-Voice） | **不跟随**：音频进音频出，语言由模型自己决定（见已知限制） |
| TTS | 音色不参与（仍是单一音色 `TTS_VOICE`，默认 Cherry），只切文字语言 |

### 为什么换人设（2026-09-29）

原实现 `en = SYSTEM_PROMPT + " Answer in English…"`：中文人设里那句
“**用中文口语化回答**”和追加的英文指令互相矛盾，qwen-flash 会跟着**提问语言**走，
结果英文只在“用户说英文”时生效——展厅里用中文问句就会得到中文回答，旋钮切英文形同虚设。
真机实测（同一个 `ask_with_weather`）：

| 提示词 | “介绍一下智慧空间” | “你好，你是谁？” |
| --- | --- | --- |
| 旧（中文人设 + 追加英文句） | 中文 | 中文 |
| 新（独立英文人设） | 英文 | 英文 |

对应改动：`ova/llm.py` 新增 `EN_SYSTEM_PROMPT`（可用 `QWEN_SYSTEM_PROMPT_EN` 覆盖，
保留 `query_weather` 强制调用要求），`system_prompt("en")` 返回它；
`ova/engines/pipeline.py` 增加 `ensure_reply_language()` 兜底。
实测提示词加强后仍非 100%（中文产品名 + 中文问句偶发中文回答），所以兜底这层不能省。

## 持久化位置与"重启是否保留"

- 状态文件默认：`dialogue_lang_file=/tmp/ova_lang.state`，内容一行 `zh` 或 `en`。
- 展区 Reachy CM4 的 `/etc/ova.env` 已设置 `WAKE_DIALOGUE_LANG_FILE=/var/lib/ova/lang.state`，保留此前的 `en` 状态；OPA 与 KeyMesh 都通过 `/lang` 读取，语音服务或整机重启后仍在。
- 其他设备未设置覆盖项时，`ova-wake` **进程重启**保留 `/tmp` 文件，**整机重启**回到默认 `zh`。
- 读不到文件、文件是目录、内容是乱码（例如被别的程序覆盖）：一律回退到 `dialogue_lang`，
  **不抛异常**，只记一条 debug 日志；写不进去（只读磁盘、父目录不存在等）只记 `LANG_STATE_ERROR` warning，
  HTTP 仍按切换结果回答。

## 参数

| 配置项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `dialogue_lang` | `WAKE_DIALOGUE_LANG` | `zh` | 默认对话语言（`zh`/`en`）；也是状态文件读不到时的回退值 |
| `dialogue_lang_file` | `WAKE_DIALOGUE_LANG_FILE` | `/tmp/ova_lang.state` | 语言状态文件路径（`POST /lang` 写、`GET /lang` 与每轮对话读） |

入口与语言切换互不影响：`inject_port=0` 关掉的是整个 HTTP 入口（`/lang` 一起关），
但语音链路仍会读状态文件，所以手写状态文件同样能切语言。

## 实现位置

- `src/ova/lang.py`（新增，纯 stdlib）
  - `normalise()`：别名/大小写/空白 → `zh`/`en`，非法返回 `None`。
  - `state_path()` / `configured()` / `current()` / `chosen()`：`current()` 是"现在什么语言"
    （状态文件 → 配置 → `zh`）；`chosen()` 是"是否有人特意选过语言"（状态文件，或非默认的配置值），
    讲解选版用它来保留"没切换时按关键词"的老行为。
  - `set_lang()` / `toggle()`：写状态文件（必要时建父目录）+ `LANG_SET` 日志 + `svc_event("system", ...)`。
- `src/ova/inject.py`
  - `InjectHandler.cfg`：由 `make_server(host, port, cfg)` 把解析后的配置绑进 handler 类（不是模块全局），
    `start_inject_server(cfg)` 传入；`/inject`、`/wake` 行为不变。
  - `POST /lang`（`_lang()`）与 `GET /lang`（`do_GET`）；非法语言沿用既有 `ValueError → 400` 路径。
- `src/ova/dialogue.py`
  - `_playback_from_text()`：开头解析 `effective = lang or ova.lang.current(cfg)` 并打 `DIALOGUE_LANG`；
    讲解分支用 `lang or ova.lang.chosen(cfg)` 选变体，聊天分支传 `{**cfg, "reply_lang": effective}`。
  - `play_asset()`：所有固定兜底音都按本轮语言选择 `*_en.wav`；注入请求的显式语言优先，缺失英文文件回退中文。
- `src/ova/llm.py`：`system_prompt(lang=None)` 在英文时返回独立 `EN_SYSTEM_PROMPT`（可用 `QWEN_SYSTEM_PROMPT_EN` 覆盖），中文继续用 `SYSTEM_PROMPT`（`QWEN_SYSTEM_PROMPT` 可覆盖）；`chat()` 行为不变。
- `src/ova/engines/pipeline.py`：`ask_with_weather(text, lang=None)` 用 `system_prompt(lang)`；
  `PipelineEngine.respond()` 从 `cfg["reply_lang"]` 取语言（`QWEN_REPLY` 日志带 `lang=`）。
- `src/ova/wake.py`：`DEFAULTS` / `ENV_MAP` 加 `dialogue_lang`、`dialogue_lang_file`。
- `config/hardware/reachy-mini.json`、`config/example.json`：记录两个新参数。
- `tests/test_lang.py`、`tests/test_inject.py`、`tests/test_engines.py`：见下。

## 日志验收

```text
LANG_SET previous=zh lang=en                      # POST /lang 切换成功
对话语言已切换: zh → en                             # 调试台 system 卡片事件
DIALOGUE_LANG lang=en source=current              # 每轮用的语言（source=explicit/current）
SOLUTION_INTRO_LANG id=smart_retail zh->en        # 讲解选了另一语言版本
SOLUTION_INTRO id=smart_retail lang=en file=.../smart_retail_en.wav
QWEN_REPLY lang=en text=Hangzhou is ...           # pipeline 回答语言
INJECT_READY host=127.0.0.1 port=8090 lang=zh     # 启动时的当前语言
LANG_STATE_ERROR path=/tmp/ova_lang.state ...     # 写不进去（只是 warning）
```

## 现场验收（部署后）

1. `curl -s -X POST :8090/lang` → `{"ok":true,"lang":"en","previous":"zh"}`，日志 `LANG_SET`。
2. `curl -s :8090/lang` → `{"ok":true,"lang":"en"}`。
3. 切到英文后注入中文关键词：`curl -s -X POST :8090/inject -d '{"text":"介绍一下智慧零售"}'`
   → 播 `smart_retail_en.wav`（日志 `SOLUTION_INTRO_LANG id=smart_retail zh->en`）。
4. 切到英文后问天气：`curl -s -X POST :8090/inject -d '{"text":"今天天气怎么样"}'`
   → 回答是英文（日志 `QWEN_REPLY lang=en`）。
5. 打断/停止不受影响：`{"text":"停止"}`、`{"text":"stop"}` 仍走原路由。
6. 重启服务（`systemctl restart ova-wake`）后 `GET /lang` 仍是 `en`；CM4 使用 `/var/lib/ova/lang.state`，整机重启也保留；默认配置 `/tmp` 在整机重启后回到 `zh`。
7. `curl -s -X POST :8090/lang -d '{"lang":"jp"}'` → 400，语言不变。

## 已知限制

- **`engine=e2e` 不跟随语言**：GLM-4-Voice 是音频进音频出，回答语言由模型自己决定，
  `reply_lang` 对它无效（它也不读这个键）。要端到端模式也听话，需要改人设/加语言字段，本次没做。
- **TTS 仍是单一音色**：只切文字语言（中文人设 + 英文指令），英文发音的自然度取决于
  `TTS_VOICE`（默认 Cherry），本次不动音色、不做英文专用音色。
- **讲解选版只认显式切换**：没切换过（状态文件不存在且 `dialogue_lang` 是默认 `zh`）时，
  讲解语言仍由关键词决定——这是为了不改变既有中英文触发词行为（`tests/test_dialogue_routing.py`
  的 `test_pipeline_english_intro_routes_to_english_wav` 覆盖它）。要"没切换也全英文"，
  把 `WAKE_DIALOGUE_LANG=en` 配上即可（此时视为显式选择）。
- **e2e 下的本地 ASR 关键词**：e2e 引擎不做关键词路由，所以英文/中文关键词讲解在 e2e 下本来就不生效。
- **状态文件不是锁**：`POST /lang` 与对话轮次之间没有加锁，极端并发下可能有一轮读到旧值（下一次即可纠正）。
- **无鉴权**：与 `/inject`、`/wake` 一样只绑 `127.0.0.1`；跨机调用需自行加网络层保护。