# 双语字幕工坊 (BilingualSubStudio) V1.5

面向中文短剧 / 二创短视频出海（YouTube 等平台）的一站式桌面双语字幕本地化工具。  
集成 **离线 PP-OCRv5 高精度双轨提取**、**外置本机 Ollama / 云端 API 一条龙自动翻译（并保留手动 Agent 提示词）** 与 **主字幕 + 单条花字自由 `(X, Y)` 拖拽排版 + 多重防查重批量烧录**。

---

## 核心特性（V1.5）

1. **一键全自动一条龙（提取 → AI 翻译 → 烧录出片）**
   - 选好原视频文件夹后点击顶部 **「一键全自动一条龙」**，自动串行完成：未定制视频红框探测 + PP-OCRv5 提取 → 内置 AI 翻校 + 自动卸载显存 → 全自动防重叠与去重烧录成片。
   - 支持勾选 **「翻译后暂停供抽查」**；同时也 **100% 保留** 第 1/2 页中的 **「手动复制 Agent 提示词」** 按钮，满足外部 Agent 手动精细化翻译工作流。

2. **外置本机 Ollama & 云端 API 双引擎 AI 翻译（8G 显存友好）**
   - 绿色包保持轻巧（约 400 MB），通过本地 `http://127.0.0.1:11434` 调用系统**外置 Ollama**（默认推荐 B 站开源视频字幕模型 `hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M`，亦自动识别腾讯 `Hy-MT` 专用协议），同时兼容任意 OpenAI 兼容云端 API（如 DeepSeek 等，API Key 仅保存在本地 `config/translator.json`）。
   - 采用 **阶段 A 全局方言年代定调 + 阶段 B 12 句滑动窗口上下文翻译**，严格 JSON 校验不碰原时间轴、自动修复 OCR 形近错字，缺失单句自动兜底补译，任务完成后自动发送 `keep_alive=0` 卸载模型释放显存，左侧对照表支持 **双击任意句秒改**。

3. **主字幕与单条花字自由 `(X, Y)` 滑块 + 左图直接拖拽**
   - **主英文字幕**：支持水平 `X (10%~90%)` 与垂直 `Y (8%~98.5%)` 滑块调节，偏离居中时自动触发对称宽度防截断适配；开启「左图拖拽调整 `(X, Y)`」后可直接用鼠标拖动主字幕与当前帧花字，松手自动按集持久化到 `_video_burn_styles.json`。
   - **自动防重叠模式**：自动读取或实时探测原中文字幕真实高度，底部留白充足放正下方，太低贴底自动放正上方，随时可一键恢复自动排版或居中。

4. **离线 PP-OCRv5 双模型 + DirectML 通用显卡加速**
   - 内置 `ch_PP-OCRv5_rec_server.onnx` 高精度模型（抗综艺花字/发光字体）与 `ch_PP-OCRv5_rec_mobile.onnx` 极速模型可选，通过 `onnxruntime-directml` 支持 AMD / NVIDIA / Intel 显卡加速与限流 CPU 自动兜底。

5. **主台词 + 花字双轨独立选区、水印过滤与原子防损坏输出**
   - 支持任意数量批量视频逐集框选专属红/蓝框并自动记忆（`_video_rois.json`）；烧录采用 `.building.mp4` 临时写入 + 原子改名、全路径 `setsar=1` 输出标准 `1:1` 比例，并在对应目录持久化保存 `_extract_log.txt`、`_translate_log.txt` 与 `_burn_log.txt`。

---

## 快速开始

### 方式一：直接下载绿色免安装版（推荐 Windows 用户）
1. 前往本仓库 [Releases](../../releases) 页面下载最新 **`BilingualSubStudio-v1.5.0-Portable-Win64.zip`**。
2. 解压后直接双击运行 **`双语字幕工坊.exe`**（已内置 FFmpeg/FFprobe、DirectML 与 PP-OCRv5 离线模型，无需配置 Python 或 CUDA）。
3. 如需使用**本地离线 AI 自动翻译**：请先在电脑安装 [Ollama](https://ollama.com/) 并拉取 B 站字幕模型（或切换为第 2 页云端 OpenAI 兼容 API / 手动复制 Agent 提示词）：
   ```bash
   ollama pull hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M
   ```

### 方式二：从源码运行与自打包
> 环境要求：Windows 10/11 + Python 3.10+ + 系统中可访问 `ffmpeg` / `ffprobe`（推荐使用 `winget install Gyan.FFmpeg`）。

```bash
# 1. 安装依赖（DirectML 显卡加速通用版）
pip install -r requirements.txt

# 2. 下载并校验离线 PP-OCRv5 模型（约 101 MB）
python download_models.py

# 3. 启动图形界面
python main.py
```

如需自行构建桌面绿色免安装版：
```bash
pip install pyinstaller>=6.10.0
python build_exe.py
```

---

## 仓库目录结构

```text
bilingual-sub-studio/
├── main.py                 # 主入口（GUI 启动、--self-test 端到端与 --test-translate 自测）
├── build_exe.py            # PyInstaller 一键构建 V1.5 绿色免安装版脚本
├── download_models.py      # 离线 PP-OCRv5 模型下载与 SHA-256 校验脚本
├── requirements.txt        # Python 运行依赖清单
├── config/
│   ├── glossary.txt        # 默认多地方言与剧情专有词典
│   └── presets.json        # 横屏 / 竖屏 / 偏高字幕预设选区模板
├── core/
│   ├── runtime_paths.py    # 绿色包 / 开发态路径、翻译器配置与内置 FFmpeg 探测
│   ├── extractor.py        # PP-OCRv5 DirectML 双轨提取、自动红框定位与提示词生成
│   ├── translator.py       # 外置 Ollama (Index-Translate-9B / Hy-MT) 与 API 自动翻校引擎
│   └── burner.py           # 自动防重叠、主字幕与花字自由 X/Y 排版、去重滤镜与原子烧录
└── gui/
    ├── preview_canvas.py   # 交互式红/蓝选区框与烧录字幕/花字直接拖拽画布
    └── app.py              # 一键一条龙与三步骤工作台（提取 / AI翻译校对 / 自由调位烧录）
```
