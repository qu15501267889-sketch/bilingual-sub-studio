import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
APP_NAME = "双语字幕工坊"
DESKTOP_RELEASE_DIR = Path.home() / "Desktop" / "双语字幕工坊_绿色免安装版"


def _resolve_ffmpeg_binary(name: str) -> Path:
    """按环境变量、系统 PATH、WinGet 包目录顺序自动探测 ffmpeg/ffprobe 可执行文件。"""
    env_val = os.environ.get(f"{name.upper()}_EXE", "").strip()
    if env_val:
        env_path = Path(env_val)
        if env_path.is_file():
            return env_path

    which_path = shutil.which(name)
    if which_path:
        return Path(which_path)

    winget_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_dir.is_dir():
        matches = sorted(winget_dir.glob(f"**/{name}.exe"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        f"找不到打包所需的 {name}.exe！请通过环境变量 {name.upper()}_EXE 指定路径，"
        f"将其加入系统 PATH，或使用 winget 安装 Gyan.FFmpeg。"
    )

README_TEXT = """==================== 双语字幕工坊 V1.3（绿色免安装版）====================

【核心升级与特点】
1. 整套文件夹直接拷贝到任何 Windows 10/11 电脑即可使用（无需安装 Python、无需配置 FFmpeg 环境变量、无需 N 卡 CUDA）。
2. 全新升级离线 PP-OCRv5 高精度模型（V1.3 重磅）：内置 ch_PP-OCRv5_rec_server.onnx 高精度核心识别模型，专门大幅降低综艺/手写/发光等花哨字体的错别字率，同时提供极速版可选，全程离线支持 AMD/NVIDIA DirectML 显卡加速。
3. 全自动防重叠排版：第 2 步烧录时自动检测原中文字幕真实高度并结合去重缩放变化——原字幕底部留白充足就放正下方，原字幕太低贴底就自动放正上方！
4. 全视频逐集独立选区：批量文件夹有多少个视频就列多少条！支持「提取前自动识别未定制视频红框」，也可点击清单任意视频拖拽专属红/蓝框（自动记忆在 _video_rois.json）。
5. 成品原子防损坏与完整日志：烧录过程写入 .building.mp4 临时文件完成后再原子改名（杜绝未完成残片导致播放器提示无效），统一输出标准 SAR 1:1，并自动保存 _extract_log.txt 与 _burn_log.txt。

【使用步骤】
1. 双击运行本目录下的「双语字幕工坊.exe」。
2. 第 1 步（提取）：选择原视频文件夹（默认使用 PP-OCRv5 高精度模型 + DirectML 显卡加速），点击「一键批量提取中文 SRT」。提取完成后自动弹出《给 Agent 提示词》，一键复制发给 WorkBuddy 批量翻译。
3. 第 2 步（烧录）：把翻译好的同名英文 SRT 放入英文文件夹，在软件第 2 页点「扫描匹配状态」，默认开启「自动防重叠（优先下方，太低放上方）」，直接点「一键批量烧录全部」即可出片。

【注意】
请勿移动或删除同目录下的 _internal、bin、models 和 config 文件夹，否则程序将无法正常运行。
"""


def main() -> None:
    pyinstaller_exe = Path(sys.executable).resolve().parent / "pyinstaller.exe"
    if not pyinstaller_exe.exists():
        cmd_base = [sys.executable, "-m", "PyInstaller"]
    else:
        cmd_base = [str(pyinstaller_exe)]

    cmd = cmd_base + [
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--name", APP_NAME,
        "--collect-all", "rapidocr",
        "--collect-all", "onnxruntime",
        "--collect-all", "omegaconf",
        "--collect-all", "pyclipper",
        "--collect-all", "shapely",
        "--collect-all", "yaml",
        "--hidden-import", "PIL._tkinter_finder",
        str(PROJECT_DIR / "main.py"),
    ]

    print("=== 开始 PyInstaller V1.3 绿色打包 ===")
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)

    dist_dir = PROJECT_DIR / "dist" / APP_NAME
    bin_dir = dist_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_src = _resolve_ffmpeg_binary("ffmpeg")
    ffprobe_src = _resolve_ffmpeg_binary("ffprobe")
    for src in (ffmpeg_src, ffprobe_src):
        shutil.copy2(src, bin_dir / src.name)
        print(f"已内置: bin/{src.name}")

    # 内置离线 PP-OCRv5 模型目录
    src_models = PROJECT_DIR / "models" / "ppocr_v5"
    dst_models = dist_dir / "models" / "ppocr_v5"
    if dst_models.exists():
        shutil.rmtree(dst_models)
    shutil.copytree(src_models, dst_models)
    for m_name in ("ch_PP-OCRv5_det_mobile.onnx", "ch_PP-OCRv5_rec_server.onnx", "ch_PP-OCRv5_rec_mobile.onnx"):
        mp = dst_models / m_name
        if not mp.exists() or mp.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"缺少或损坏的离线 PP-OCRv5 模型: {mp}")
        print(f"已内置模型: models/ppocr_v5/{m_name} ({mp.stat().st_size / 1024 / 1024:.2f} MB)")

    (dist_dir / "使用说明.txt").write_text(README_TEXT, encoding="utf-8")

    # 预置多地方言词典与选区预设模板到绿色包 config 目录
    from core.runtime_paths import DEFAULT_GLOSSARY, DEFAULT_PRESETS
    cfg_dir = dist_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "presets.json").write_text(json.dumps(DEFAULT_PRESETS, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg_dir / "glossary.txt").write_text(DEFAULT_GLOSSARY.strip() + "\n", encoding="utf-8")

    # 检查是否已带上 DirectML.dll（若缺失则从 site-packages 补拷贝）
    import onnxruntime
    ort_capi = Path(onnxruntime.__file__).resolve().parent / "capi"
    dst_capi = dist_dir / "_internal" / "onnxruntime" / "capi"
    dst_capi.mkdir(parents=True, exist_ok=True)
    for dll_name in ("DirectML.dll", "onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        src_dll = ort_capi / dll_name
        dst_dll = dst_capi / dll_name
        if src_dll.exists() and not dst_dll.exists():
            shutil.copy2(src_dll, dst_dll)
            print(f"补齐 DLL: {dll_name}")
    print("DirectML.dll 就绪:", (dst_capi / "DirectML.dll").exists())

    # 同步覆盖唯一桌面交付目录（先强制关闭旧运行进程释放 DLL/EXE 文件锁）
    subprocess.run(["taskkill", "/F", "/IM", f"{APP_NAME}.exe"], capture_output=True)
    time.sleep(0.8)
    if DESKTOP_RELEASE_DIR.exists():
        shutil.rmtree(DESKTOP_RELEASE_DIR)
    shutil.copytree(dist_dir, DESKTOP_RELEASE_DIR)
    print(f"=== 方案 A 绿色包已交付至桌面: {DESKTOP_RELEASE_DIR} ===")


if __name__ == "__main__":
    main()
