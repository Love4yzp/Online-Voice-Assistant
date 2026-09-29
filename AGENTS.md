# Online Voice Assistant

## 项目简介

给展厅 Reachy Mini 提供一套语音助手：本地 Hey Jarvis 唤醒，加**两条可切换的对话链路**——
半在线（本地 ASR → 千问 LLM → 千问 TTS）与端到端（录音直发 GLM-4-Voice），以及网页调试台。
当前重点是让机器人能在展厅按固定文案介绍方案。

## 技术栈

- Python 3.10+
- ALSA `arecord` / `aplay`
- openWakeWord ONNX
- sherpa-onnx Silero VAD：默认在 Reachy Mini 上替代旧能量阈值 VAD，可回退 `WAKE_VAD_BACKEND=energy`
- sherpa-onnx 本地 ASR：默认 SenseVoice 中英双语（中/英/粤/日/韩），可回退中文 Paraformer
- DashScope 千问 `qwen-flash` 与 `qwen3-tts-flash`（pipeline 引擎）
- 智谱 GLM-4-Voice（e2e 引擎，音频进/音频出）
- systemd 部署到 Reachy Mini `/home/pollen/ova`

## 目录结构

- `src/ova/`：唤醒、VAD、ASR、LLM、TTS、对话编排、调试台
- `config/`：示例配置、硬件档案、展厅方案讲解配置
- `assets/`：唤醒反馈音、兜底音频、预生成方案讲解音频
- `deploy/`：systemd 服务模板
- `scripts/`：模型下载与服务安装脚本
- `docs/`：架构、项目历史、会话日志

## 当前状态

### 活跃分支

- 无。2026-09-11 已将 `dev` 合并到 `main`；线上当前稳定版本为
  `pipeline + Silero VAD + SenseVoice`，用户现场确认效果良好。

### 已完成功能

- 本地唤醒后播放 Home Assistant 的 `wake_word_triggered` 提示音，不再播放“在呢”。
- 三主题展厅讲解：识别到“智慧零售/智慧空间/应急救灾”或对应英文关键词后，直接播放本地预生成中英文讲解音频。
- 展厅唤醒误触发调优：线上先从 `threshold=0.20 / hits=3` 调到 `0.30 / 4`；2026-09-11 因唤醒变钝，回调为 `0.28 / 4`。
- 正式回答前的缓冲应答默认关闭：保留唤醒成功音 `assets/response.wav`，但 `ack_before_reply=false`，避免用户说完后插入“好的/嘟嘟”再播放模型回答。
- 播放中打断：长讲解或 TTS 播放期间继续监听 `Hey Jarvis`，命中后停止当前播放，支持“停止/继续/切换介绍智慧空间”等后续指令。
- 中英双语 ASR：ASR 从中文 Paraformer 换成 SenseVoice int8（中/英/中英混说、带标点，`ASR_RESULT` 附带 `lang=zh|en`），中文 Paraformer 保留为回退模型。
- 模型 VAD：Reachy Mini 默认用 sherpa-onnx 的 Silero VAD（`models/silero_vad.onnx`），替代旧自适应能量阈值；旧 VAD 可用 `WAKE_VAD_BACKEND=energy` 回退。
- 2026-09-11 线上收口：Reachy Mini 已部署 `pipeline + Silero VAD + SenseVoice`，用户现场确认效果良好；当前没有正式回答前“好的/嘟嘟”缓冲音。
- **双引擎架构**：对话大脑抽成 `src/ova/engines/`（`Engine`/`Reply`/`build_engine`），
  `engine=pipeline`（半在线）与 `engine=e2e`（GLM-4-Voice 音频直进直出）共用唤醒/VAD/播放/打断/事件/调试台；
  切换只改一个配置项。设计说明见 `docs/dual-engine-architecture.md`。

## 重要约定

- Reachy Mini 线上路径是 `/home/pollen/ova`，服务名是 `ova-wake` 与 `ova-console`。
- VAD 由 `vad_backend` 控制：`silero` 使用 `models/silero_vad.onnx`、`vad_threshold=0.50`、`vad_min_speech_s=0.25`；
  `energy` 是旧自适应能量阈值回退。Silero VAD 模型小于 1MB，不需要 PyTorch，走当前 `sherpa_onnx=1.13.7`。
- `models/` 目录会同时放 wake、VAD、ASR 文件；非 wake ONNX 必须加入 `FEATURE_MODELS` 排除列表。
  2026-09-11 曾因 `silero_vad.onnx` 被 openWakeWord 误扫入唤醒模型导致 `Required inputs (['h', 'c'])...`，
  修复后远端自检 `wake_models ['hey_jarvis_v0.1']`。
- ASR 模型目录由 `asr_model_dir` 决定，`src/ova/asr.py` 按目录内容自动判别
  `sense_voice|paraformer|transducer`；默认 `models/asr_sense_voice_zh_en_int8`（中英双语），
  回退中文模型只需改这一项。下载用 `scripts/download_models.sh [sensevoice|small|full]`。
  选型与实测见 `docs/asr-bilingual-models-2026-09-10.md`。
- 方案讲解应优先播放本地预生成 WAV，避免展厅现场依赖实时 TTS 或让大模型改写固定文案。
- **引擎开关**：`engine=pipeline|e2e`（`WAKE_ENGINE`）。两条链路共用唤醒/VAD/播放/打断/事件/调试台；
  `e2e` 需要 `ZHIPUAI_API_KEY`，返回的是 **24kHz** 单声道**裸 PCM**（代码补 WAV 头、做音频规整后重采样到 16k 立体声），
  **没有 Function Calling**（天气工具只在 pipeline），且**非流式**（无"边说边播"，打断只能停播放）。
  固定讲解的关键词路由只对 pipeline 生效；本地 ASR 在 e2e 下仍用于打断后的"停止/继续"指令。
  切换/回退只改配置，不改代码；设计与实测见 `docs/dual-engine-architecture.md`。
  **e2e 音频链路的既定默认值（都是实测定下来的，别随手改）**：`glm_voice_pcm_rate=24000`
  （官方示例写的 44100 是错的，会导致播放加速 1.84 倍、又快又尖听不清）、`glm_voice_target_rms=0.09`
  （对齐千问 TTS 的 0.086；顺序是去直流+淡入淡出 → 压缩峰值 → 响度对齐）、`glm_voice_timeout_s=10`
  （实测 p50 1.4s，超时播兜底音）、人设"不超过10个字"（端到端模型没有硬性长度控制）。
- **两条链路不要并行起两个服务**：ALSA 是共享入口（dsnoop/dmix），同时运行会抢麦克风、抢 CPU、互相打断。
- `e2e` 模式不应在唤醒后再初始化本地 ASR：端到端引擎启动时预热，唤醒音结束后直接进入 VAD；本地 ASR 只在播放中打断后的“停止/继续”短指令里懒加载。
- `assets/*.wav` 会被唤醒应答随机池扫描；兜底音和备份音频应放入子目录。
- 方案讲解音频路径由 `config/solutions.json` 管理，当前为 `smart_retail`、`smart_space`、`emergency_response` 三个主题，各有 `zh/en` 两套 WAV。
- 当前正式中文触发词使用“智慧空间”，不把“智慧家居”作为别名触发。
- 唤醒调优记录见 `docs/wake-tuning-2026-09-10.md`；再次调优时先复核历史 `WAKE_DETECTED` 分数、背景样本、唤醒样本，再决定是否改 `threshold`/`hits`。
- 播放中打断记录见 `docs/barge-in-playback-2026-09-10.md`；打断检测使用独立参数，当前建议 `barge_in_threshold=0.30 / barge_in_hits=4`，并用 `BARGE_LISTENING` 日志观察播放期间峰值。
- 唤醒灵敏度当前试运行 `threshold=0.28 / hits=4`；若仍需叫多次，下一档试 `0.25 / 4`；若误触发回升，退回 `0.30 / 4`。
- 头部待机动作当前恢复为官方 recorded move 库，并新增独立看门狗；记录见 `docs/reachy-demo-official-watchdog-2026-09-10.md`。前一版轻量小幅 `/api/move/goto` 记录见 `docs/reachy-demo-lite-motion-2026-09-10.md`，可作为回退方案。
- **外部文本入口**：`ova-wake` 内置 stdlib HTTP 入口 `POST /inject`（文本 = 一句识别结果，走现有路由：
  停止 / 继续 / 三主题预制讲解 / 普通问答）与 `POST /wake`（= 一次唤醒命中，进入一轮"听访客说话 → 回答"）。
  监听地址/端口由 `WAKE_INJECT_HOST`（默认 `127.0.0.1`）/ `WAKE_INJECT_PORT`（默认 `8090`，**`0` = 关闭**）控制；
  只在 `ova-wake` 进程内起 daemon 线程，不新增第三方依赖。播放中注入先打断当前播放
  （复用 `play_interruptible()` 已有的终止路径，不另杀 `aplay`），注入触发的播放同样写 `speaking_state_file`。
  设计与已知限制见 `docs/external-input-inject-2026-09-15.md`。
- **对话语言（持久切换）**：`POST /lang`（body `{}`/`{"toggle":true}` = 中英互切，`{"lang":"en"}` = 直设，
  非法值 400）与 `GET /lang` 由同一个 `ova-wake` HTTP 入口提供；当前语言存在
  `dialogue_lang_file`（默认 `/tmp/ova_lang.state`，`WAKE_DIALOGUE_LANG_FILE`），读不到/内容非法回退 `dialogue_lang`
  （`WAKE_DIALOGUE_LANG`，默认 `zh`），任何异常都不抛。它接到两处：讲解选版（显式 `lang` 优先，
  其次**现场切换过的**语言，都没指定时仍按关键词语言，保留中英双语触发行为）与 LLM 回答语言
  （`cfg["reply_lang"]` → `pipeline` 的 `llm.system_prompt(lang)`，`en` 追加英文指令）。
  TTS 音色不变，`e2e` 引擎不跟随语言。见 `docs/dialogue-language-switch-2026-09-15.md`。

## 变更日志

| 日期 | 分支 | 说明 |
|---|---|---|
| 2026-09-09 | `dev` | 替换唤醒反馈音，并新增方案一固定文案讲解试点。 |
| 2026-09-10 | `dev` | 基于现场日志与校准录音，将展厅唤醒参数调整为 `threshold=0.30 / hits=4`，降低误触发。 |
| 2026-09-10 | `dev` | 将展厅讲解从“方案一/二/三”改为“智慧零售/智慧空间/应急救灾”三主题，并生成中英文预制音频。 |
| 2026-09-10 | `dev` | 新增播放中 `Hey Jarvis` 打断：可停止、继续或切换到新的展厅讲解。 |
| 2026-09-10 | `dev` | 将播放中打断阈值从 `0.45/4` 调低到 `0.30/4`，补充 `BARGE_LISTENING` 峰值日志用于现场调优。 |
| 2026-09-10 | `dev` | 新增轻量展厅待机动作：待机 20 秒一次，说话时 10 秒一次，替代高频官方 recorded moves。 |
| 2026-09-10 | `feat/asr-sensevoice-bilingual` | ASR 换为 SenseVoice 中英双语（模型类型自动识别，Paraformer 可回退），英文触发词因此真正可用。 |
| 2026-09-10 | `feat/asr-sensevoice-bilingual` | 待机动作恢复官方 recorded move 库，并新增 `reachy-demo-watchdog` 自动处理卡住动作和服务恢复。 |
| 2026-09-10 | `dev` | 分支合并回 `dev`（ASR 双语 + 待机动作 watchdog 两个提交）。 |
| 2026-09-10 | `dev` | 抽出可插拔对话引擎（`src/ova/engines/`）：现有 ASR→千问→TTS 路径包成 `pipeline` 引擎，行为不变。 |
| 2026-09-10 | `dev` | 新增 `e2e` 引擎（GLM-4-Voice 录音直进/音频直出）+ 调试台卡片⑥ + 离线 PoC 脚本；两条链路用一个配置项切换。 |
| 2026-09-10 | `dev` | e2e 引擎上线实测与修复：采样率纠正为 24kHz（官方示例的 44100 会让播放加速 1.84 倍）、去开头爆音与直流、压缩峰值并对齐响度、人设收紧到 10 字、请求超时降到 10 秒。 |
| 2026-09-11 | `dev` | 试运行唤醒 `threshold=0.28 / hits=4`，默认关闭正式回答前的 `ack_think.wav` 缓冲音，并让 e2e 首轮不再在唤醒后阻塞加载 ASR。 |
| 2026-09-11 | `dev` | Reachy Mini 主链路切为 Silero VAD + SenseVoice 中英 ASR；保留 energy VAD 与 Paraformer 中文 ASR 回退。 |
| 2026-09-11 | `dev` | 线上部署并验证 `pipeline + Silero VAD + SenseVoice`：服务 active，KWS 只加载 `hey_jarvis_v0.1`，用户现场确认效果良好。 |
| 2026-09-11 | `main` | 合并 `dev` 到 `main`，主分支收口本轮唤醒、VAD、ASR、双引擎、展厅讲解与待机动作改动。 |
| 2026-09-15 | `main` | 新增外部文本输入入口：`ova-wake` 内置 `POST /inject`（文本当识别结果，复用停止/继续/三主题讲解/普通问答路由，播放中注入先打断当前播放）与 `POST /wake`（当唤醒命中起一轮对话）；`WAKE_INJECT_PORT` 默认 8090、`0`=关闭，仅监听本机，无新依赖。 |
| 2026-09-15 | `main` | 新增持久对话语言切换：展厅旋钮长按 → `POST /lang`（中↔英，`GET /lang` 读当前值），状态存 `dialogue_lang_file`（默认 `/tmp/ova_lang.state`），回退 `dialogue_lang`（默认 `zh`）；切换后讲解选版与 `pipeline` 的 LLM 回答语言都跟随（`reply_lang` → `llm.system_prompt()`），TTS 音色与 `e2e` 引擎不变。新增 `src/ova/lang.py` 与 `tests/test_lang.py`。 |
| 2026-09-29 | `feat/motion-v2` | 修英文切换不生效：`en` 改用独立英文人设 `EN_SYSTEM_PROMPT`（旧的“中文人设 + 追加英文句”自相矛盾，模型跟着提问语言走，中文问句一律中文答）；`pipeline.ensure_reply_language()` 增加中文回答自动重问一次（`LANG_RETRY`，中文占比阈值 0.2）。真机已验证中文提问 3/3 英文回答，zh 口径 byte 不变。见 `docs/dialogue-language-switch-2026-09-15.md`。 |
