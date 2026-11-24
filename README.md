# 双语字幕工坊 (BilingualSubStudio) V1.3

面向中文短剧 / 二创短视频出海（YouTube 等平台）的一站式桌面双语字幕本地化工具。  
集成 **离线 PP-OCRv5 高精度双轨提取**、**多地方言自适应 Agent 翻译工作流** 与 **全自动防重叠排版 + 多重防查重批量烧录**。

---

## 核心特性（V1.3）

1. **离线 PP-OCRv5 双模型 + DirectML 通用显卡加速**
   - 内置 `ch_PP-OCRv5_rec_server.onnx` 高精度模型与 `ch_PP-OCRv5_rec_mobile.onnx` 极速模型可选，显著降低综艺花字、发光/手写等花哨字体的漏字与错别字率。
   - 基于 `onnxruntime-directml` 实现 **AMD / NVIDIA / Intel 显卡通用硬件加速**，并内置运行时异常与限流 CPU 自动兜底机制，批量跑片不卡电脑。

2. **主台词 + 画面花字/背景大字 双轨提取与水印过滤**
   - **轨道 1（主台词红框）**：粗扫 + 二分精定位边界，自动记录原中文字幕在画面中的真实纵向占比（`*_main.layout.json`）。
   - **轨道 2（花字/背景蓝框）**：自动识别画面注释、招牌与转场提示，过滤固定角落声明水印（如 `AI视频`、`仅供娱乐`），并输出精确空间坐标账本（`*_notes.pos.json`）。

3. **全视频逐集独立选区 & 自动红框定位**
   - 批量文件夹中有多少个视频就在左侧列表完整展示多少条，支持点选任意视频单独框选红/蓝框，自动持久化记忆到 `_video_rois.json`。
   - 支持提取前一键自动探测未定制视频的字幕区域，免除逐集手工调整的繁琐。

4. **多地方言自适应识别 + 整句通顺校对 Agent 提示词**
   - 自动生成 `AGENT_PROMPT.txt`，内置 90 年代北方农村、东北、川渝、粤语、云南、湖南湖北等高频方言表达词典（`config/glossary.txt` 可视化编辑），引导翻译 Agent 按剧情判别年代地域并先通读整句纠错再地道意译。

5. **全自动防重叠上下排版 + YouTube 多重防查重批量烧录**
   - 第 2 步烧录时自动读取或实时探测原中文字幕位置，结合画幅微放大 / 微旋转 / 模糊背景框等缩放形变，智能决策将英文字幕放到 **原字幕正下方**（底部留白充足）或 **原字幕正上方**（原字幕贴底过低），支持单行自适应字号拟合与逐集样式覆盖。
   - 支持硬件编码（AMD AMF / NVIDIA NVENC）与 CPU 软编，内置微变速、微放大、微旋转、电影调色、胶片颗粒、暗角、高斯模糊背景边框、音频指纹微扰、元信息剥离与首尾安全裁剪等防查重选项。

6. **成品原子防损坏 & 完整磁盘日志**
   - 烧录先写入隐藏临时文件 `.视频名.building.mp4`，完整封装后再 `os.replace` 原子改名，杜绝播放器提前扫入半成品导致提示“无效”。
   - 所有视频滤镜链统一 `setsar=1` 输出标准 `1:1` 像素比例，并在中文/成品目录自动保存 `_extract_log.txt` 与 `_burn_log.txt`。

---

## 快速开始

### 方式一：直接下载绿色免安装版（推荐 Windows 用户）
1. 前往本仓库 [Releases](../../releases) 页面下载最新 **`BilingualSubStudio-v1.3.0-Portable-Win64.zip`**。
2. 解压后直接双击运行 **`双语字幕工坊.exe`**（已内置 FFmpeg/FFprobe、DirectML 与全套 PP-OCRv5 离线模型，无需安装 Python 或 CUDA）。

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
├── main.py                 # 主入口（GUI 启动与 --self-test 端到端自测模式）
├── build_exe.py            # PyInstaller 一键构建绿色免安装版脚本
├── download_models.py      # 离线 PP-OCRv5 模型下载与 SHA-256 校验脚本
├── requirements.txt        # Python 运行依赖清单
├── config/
│   ├── glossary.txt        # 默认多地方言与剧情专有词典
│   └── presets.json        # 横屏 / 竖屏 / 偏高字幕预设选区模板
├── core/
│   ├── runtime_paths.py    # 绿色包 / 开发态路径与内置 FFmpeg 自动探测
│   ├── extractor.py        # PP-OCRv5 DirectML 双轨提取、自动红框定位与提示词生成
│   └── burner.py           # 自动防重叠排版、实时预览渲染、去重滤镜与原子烧录
└── gui/
    ├── preview_canvas.py   # 交互式红/蓝选区框与实时字幕预览画布
    └── app.py              # 主双标签页桌面界面（第1步提取 / 第2步烧录）
```
