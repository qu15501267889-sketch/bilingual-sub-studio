# 双语字幕工坊 (BilingualSubStudio) V1.5.1

面向中文短剧 / 二创短视频出海（YouTube 等平台）的一站式桌面双语字幕本地化工具。  
集成 **离线 PP-OCRv5 高精度双轨提取**、**内置/外置 Ollama 与云端 API 一键一条龙自动翻译（完整保留手动 Agent 提示词）**、**`models/llm/` 本地模型一键导入**，以及 **主字幕 + 单条花字自由 `(X, Y)` 拖拽排版与多重防查重批量烧录**。

---

## 核心特性（V1.5.1）

1. **提供双版本发行包（开箱任选，均不含大模型权重以控制包体积）**
   - **内置 Ollama 全能版（`BilingualSubStudio-v1.5.1-Portable-WithOllama-Win64.zip`）**：已内置精简后的跨硬件 Ollama 运行时（支持 **NVIDIA CUDA 12**、**AMD/Intel Vulkan** 与 CPU 回退）。启动软件或运行翻译时若未检测到系统服务会自动静默托管拉起，退出时自动关闭，目标电脑**无需安装 Python、FFmpeg 或系统 Ollama**。
   - **标准轻量版（`BilingualSubStudio-v1.5.1-Portable-Lite-Win64.zip`）**：体积轻巧（约 406 MB），适合已安装过 Ollama 的本机、仅使用云端 API 翻译，或仅做提取/烧录配外部 Agent 翻译的用户。

2. **放入 `.gguf` 模型一键导入（零命令行）**
   - 把推荐翻译模型下载为 `.gguf` 后直接放入程序目录下的 **`models/llm/`** 文件夹，在软件第 2 页点击 **「📥 扫描导入 models/llm 本地模型」**（或直接点击顶部「🚀 一键全自动一条龙」），程序会自动生成最佳模板并注册进 Ollama。

3. **一键全自动一条龙（提取 → AI 翻译 → 烧录出片）**
   - 选好原视频文件夹后点顶部 **「🚀 一键全自动一条龙」**，自动串行完成：未定制视频红框探测 + PP-OCRv5 提取 → 内置 AI 翻校 + 自动卸载显存 → 全自动防重叠与去重烧录成片。
   - 仍 **100% 保留** 第 1/2 页的 **「📋 手动复制 Agent 提示词」** 与第 2 页中英对照双击秒改表。

4. **主字幕与单条花字自由 `(X, Y)` 滑块 + 左图直接拖拽**
   - 第 3 页支持水平 `X (10%~90%)` 与垂直 `Y (8%~98.5%)` 滑块，开启「左图拖拽调整 `(X, Y)`」后可直接按住主字幕或当前帧花字拖到任意位置并按集持久化记忆；自动防重叠模式依然可用，随时一键「居中」或恢复「自动」。

5. **离线 PP-OCRv5 + DirectML 显卡加速 + 原子防损坏输出**
   - 内置 `PP-OCRv5 Server / Mobile`，支持 AMD/NVIDIA DirectML 显卡加速；烧录采用 `.building.mp4` 原子写入与全路径 `setsar=1` 标准比例，日志自动写入 `_extract_log.txt`、`_translate_log.txt` 与 `_burn_log.txt`。

---

## 推荐本地翻译模型与官方下载直链

> 本仓库及 Release 压缩包**均不直接上传大体积 `.gguf` 权重**，请按你的电脑显卡配置从以下经过实测核验的官方 Hugging Face 直链任选下载（只需下载主体 `.gguf`，**不要仅下载** `*mmproj*.gguf` 附件）：

| 推荐定位 | 模型名称 | 体积 | Hugging Face 直接下载链接（点击或用浏览器/curl下载） | 适用与特点 |
| :--- | :--- | :--- | :--- | :--- |
| **🥇 首推（8G 显存甜点）** | **B 站 `Index-Translate-9B (Q4_K_M)`** | `5.78 GB` | https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF/resolve/main/Index-Translate-9B.Q4_K_M.gguf | B 站开源视频字幕/弹幕热梗专用模型；放入 `models/llm/` 导入后**自动同名匹配程序默认配置**，对农村方言和 OCR 错字修复极佳 |
| **🥈 8G 显存偏紧可选** | **B 站 `Index-Translate-9B (IQ4_XS)`** | `5.36 GB` | https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF/resolve/main/Index-Translate-9B.IQ4_XS.gguf | 比 `Q4_K_M` 省约 0.4 GB 显存，方言翻校能力同样强 |
| **⚡ 专业高速专用翻译** | **腾讯 `Hy-MT2-7B (Q4_K_M)`** | ~4.6 GB | https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/main/Hy-MT2-7B-Q4_K_M.gguf | 腾讯混元专用机器翻译，速度快、8G 显存宽裕，程序自动适配 Hy-MT 术语与语境协议 |
| **🪶 无独显/核显极速备选** | **腾讯 `Hy-MT2-1.8B (Q4_K_M)`** | ~1.2 GB | https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M.gguf | 体积仅 1.2 GB，笔记本核显或纯 CPU 也可流畅跑通自动翻译 |
| **🥉 通用中文推理兜底** | **阿里 `Qwen3-8B (Q4_K_M)`** | ~5.0 GB | https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf | 通用知识面广，依靠上下文结合汉字笔画推测 OCR 错字很稳 |

### 📥 模型放到哪里？怎么用？（3 步搞定）
1. **下载后放置路径**：把下载好的 `.gguf` 文件直接拷贝放进绿色包目录下的 **`models\llm\`** 文件夹（旁边会有提示文本 `把下载好的GGUF模型放这里.txt`）。
   - 如果从源码运行，则放进工程根目录下的 `models\llm\`。
2. **软件内一键导入**：启动 **`双语字幕工坊.exe`**，切换到 **「第 2 步：内置 AI 自动翻译 & 校对」**，点击 **`📥 扫描导入 models/llm 本地模型`**，软件会自动识别模型类型、写入无思考快速模板并导入（或直接运行顶部「🚀 一键全自动一条龙」也会自动帮你检测导入）。
3. **已有系统 Ollama 的电脑**：如果你此前已经用 `ollama pull` 下载过模型，软件会直接共用现有模型，无需重复放入 `models\llm\`。

---

## 快速开始

### 方式一：下载绿色免安装版（推荐 Windows 10/11 用户）
1. 前往本仓库 [Releases](../../releases) 页面按需选择下载 **V1.5.1**：
   - 未装过 Ollama 想要完整一条龙：下载 **`BilingualSubStudio-v1.5.1-Portable-WithOllama-Win64.zip`**
   - 仅提取/烧录或已有外部 Ollama / 云端 API：下载 **`BilingualSubStudio-v1.5.1-Portable-Lite-Win64.zip`**
2. 解压后把你选好的 `.gguf` 翻译模型放入 `models\llm\`（如果仅用云端 API 或外部 Agent 可跳过此步），双击 **`双语字幕工坊.exe`** 即可使用。

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

如需自行构建绿色免安装版（同时生成标准轻量版与内置 Ollama 版）：
```bash
pip install pyinstaller>=6.10.0
python build_exe.py --edition both
```

---

## 仓库目录结构

```text
bilingual-sub-studio/
├── main.py                              # 主入口（GUI 启动、--self-test / --test-embedded-ollama / --test-translate）
├── build_exe.py                         # PyInstaller 双版本构建脚本（支持 --edition lite|with-ollama|both）
├── download_models.py                   # 离线 PP-OCRv5 模型下载与 SHA-256 校验脚本
├── requirements.txt                     # Python 运行依赖清单
├── config/
│   ├── glossary.txt                     # 默认多地方言与剧情专有词典
│   └── presets.json                     # 横屏 / 竖屏 / 偏高字幕预设选区模板
├── models/
│   └── llm/
│       └── 把下载好的GGUF模型放这里.txt   # 本地 GGUF 模型存放位置与推荐直链说明
├── core/
│   ├── runtime_paths.py                 # 绿色包 / 开发态路径、翻译器配置与内置组件探测
│   ├── embedded_ollama.py               # 内置 Ollama 引擎自动启停托管与 models/llm GGUF 一键扫描导入
│   ├── extractor.py                     # PP-OCRv5 DirectML 双轨提取、自动红框定位与提示词生成
│   ├── translator.py                    # Ollama (Index-Translate-9B / Hy-MT) 与 API 自动翻校引擎
│   └── burner.py                        # 自动防重叠、主字幕与花字自由 X/Y 排版、去重滤镜与原子烧录
└── gui/
    ├── preview_canvas.py                # 交互式红/蓝选区框与烧录字幕/花字直接拖拽画布
    └── app.py                           # 一键一条龙与三步骤工作台（提取 / AI翻译校对 / 自由调位烧录）
```
