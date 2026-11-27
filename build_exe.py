import argparse
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
LITE_DIST_NAME = "双语字幕工坊_V1.5.2_标准轻量版"
WITH_OLLAMA_DIST_NAME = "双语字幕工坊_V1.5.2_内置Ollama全能版"


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


def _resolve_ollama_install_dir() -> Path:
    """按环境变量、系统 PATH、默认用户安装目录顺序自动定位本机 Ollama 目录。"""
    candidates: list[Path] = []
    env_dir = os.environ.get("OLLAMA_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    which_ol = shutil.which("ollama")
    if which_ol:
        candidates.append(Path(which_ol).resolve().parent)
    candidates.append(Path.home() / "AppData" / "Local" / "Programs" / "Ollama")

    for d in candidates:
        if (d / "ollama.exe").is_file() and (d / "lib" / "ollama").is_dir():
            return d

    raise FileNotFoundError(
        "找不到本机 Ollama 安装目录（需包含 ollama.exe 与 lib/ollama）！"
        "请通过环境变量 OLLAMA_DIR 指定目录，或先安装 Windows 版 Ollama。"
    )


def _bundle_ollama_runtime(dst_dir: Path) -> None:
    """将精简后的跨硬件 Ollama 运行时（CPU + NVIDIA CUDA12 + AMD/Intel Vulkan，不含模型权重）内置到绿色包。"""
    src_root = _resolve_ollama_install_dir()
    dst_ollama = dst_dir / "runtime" / "ollama"
    dst_lib = dst_ollama / "lib" / "ollama"
    dst_models = dst_dir / "runtime" / "ollama_models"

    if dst_ollama.exists():
        shutil.rmtree(dst_ollama)
    dst_lib.mkdir(parents=True, exist_ok=True)
    dst_models.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_root / "ollama.exe", dst_ollama / "ollama.exe")
    src_lib = src_root / "lib" / "ollama"

    # 拷贝根目录通用 DLL / CPU runner
    for item in src_lib.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_lib / item.name)

    # 拷贝跨硬件加速后端（CUDA12 兼顾 NVIDIA；Vulkan 兼顾 AMD/Intel 显卡），控制总包体远低于 2 GB
    for backend_dir in ("cuda_v12", "vulkan"):
        sub_src = src_lib / backend_dir
        if sub_src.is_dir():
            shutil.copytree(sub_src, dst_lib / backend_dir)
        else:
            print(f"[警告] 未发现可选后端子目录，跳过: {sub_src}")

    total_bytes = sum(p.stat().st_size for p in dst_ollama.rglob("*") if p.is_file())
    print(f"已内置 Ollama 运行时 (CUDA12 + Vulkan + CPU): {total_bytes / 1024 / 1024:.1f} MB")


MODEL_LINKS_GUIDE = """
【推荐本地翻译模型与下载直链（任选其一下载，均不含在软件压缩包内）】
请把下载好的 `.gguf` 文件放进本包 `models\\llm\\` 文件夹，然后在软件第 2 页点「扫描导入 models/llm 本地模型」即可：
1. 🥇 首推（短视频/方言/热梗最强，8G 显存甜点）：B 站 Index-Translate-9B
   - Q4_K_M (5.78 GB，导入即默认选中):
     https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF/resolve/main/Index-Translate-9B.Q4_K_M.gguf
   - IQ4_XS (5.36 GB，8G 显存偏紧可选):
     https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF/resolve/main/Index-Translate-9B.IQ4_XS.gguf
2. 🥈 专业高速备选：腾讯 Hy-MT2-7B (Q4_K_M):
   https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/main/Hy-MT2-7B-Q4_K_M.gguf
3. ⚡ 轻量低显存/无独显救星：腾讯 Hy-MT2-1.8B (Q4_K_M):
   https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M.gguf
4. 🥉 通用兜底推理强推：阿里 Qwen3-8B (Q4_K_M):
   https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
"""

README_LITE_TEXT = (
    "==================== 双语字幕工坊 V1.5.2（标准轻量版 · 无内置 Ollama）====================\n\n"
    "【版本说明】\n"
    "1. 本版本体积轻巧（约 406 MB），已完整内置 FFmpeg/FFprobe、PP-OCRv5 离线模型（支持 AMD/NVIDIA DirectML 显卡加速）、自由(X,Y)拖拽排版引擎与 V1.5.2 无损多维防查重引擎。\n"
    "2. 本包【不含】Ollama 运行时本体：若需使用 AI 一键自动翻译，请在本机另行安装 Ollama（或使用第 2 页云端 OpenAI 兼容 API），或者直接使用第 1/2 页保留的「手动复制 Agent 提示词」。\n"
    "3. V1.5.2 防查重支持 4 套档位预设、3 档微扰强度（默认：中，12 项黄金无损组合）及「一键随机组合」，兼顾高过审率与原始画质清晰度。\n"
    "4. 如果电脑尚未安装 Ollama 且不想折腾配置，请到发布页下载【内置 Ollama 全能版】。\n"
    + MODEL_LINKS_GUIDE
    + "\n【注意】\n请勿移动或删除同目录下的 _internal、bin、models 和 config 文件夹。\n"
)

README_WITH_OLLAMA_TEXT = (
    "==================== 双语字幕工坊 V1.5.2（内置 Ollama 全能版 · 开箱即用）====================\n\n"
    "【版本说明】\n"
    "1. 本版本已在 runtime\\ollama\\ 中完整内置跨硬件 Ollama 引擎（支持 NVIDIA CUDA12、AMD/Intel Vulkan 与 CPU 自动回退），无需在目标电脑提前安装 Ollama！\n"
    "2. 启动软件后会自动检测：若系统本身未开外部 Ollama，软件将在后台静默托管启动内置 Ollama（退出软件时自动安全关闭）。\n"
    "3. 一条龙用法：\n"
    "   - 只需从下方直链下载任意一个 `.gguf` 翻译模型放到本目录下的 `models\\llm\\` 文件夹；\n"
    "   - 打开 `双语字幕工坊.exe`，点第 2 页「扫描导入 models/llm 本地模型」（或直接选中原视频文件夹点顶部「🚀 一键全自动一条龙」自动扫描导入并出片）。\n"
    "4. V1.5.2 升级 17 项无损多维防查重引擎：支持 4 套档位预设、3 档微扰强度（默认：中）与「🎲 一键随机组合」（色度亚像素微移、动态 GOP、极微变调改声纹、无感粉红底噪等）。\n"
    "5. 依旧 100% 保留：第 1/2 页「手动复制 Agent 提示词」、第 2 页中英对照双击秒改，以及第 3 页主字幕 + 单条花字左图直接拖拽 (X, Y) 自由定位。\n"
    + MODEL_LINKS_GUIDE
    + "\n【注意】\n请勿移动或删除同目录下的 _internal、bin、models、runtime 和 config 文件夹。\n"
)


def _sync_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建双语字幕工坊 V1.5.2 绿色免安装版")
    parser.add_argument(
        "--edition",
        choices=("lite", "with-ollama", "both"),
        default="both",
        help="构建目标版本：lite（标准轻量版）、with-ollama（内置 Ollama 版）、both（同时生成两版，默认）",
    )
    args = parser.parse_args()

    pyinstaller_exe = Path(sys.executable).resolve().parent / "pyinstaller.exe"
    cmd_base = [str(pyinstaller_exe)] if pyinstaller_exe.exists() else [sys.executable, "-m", "PyInstaller"]

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

    print(f"=== 开始 PyInstaller V1.5.2 绿色打包 (edition={args.edition}) ===")
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)

    base_dir = PROJECT_DIR / "dist" / APP_NAME
    bin_dir = base_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_src = _resolve_ffmpeg_binary("ffmpeg")
    ffprobe_src = _resolve_ffmpeg_binary("ffprobe")
    for src in (ffmpeg_src, ffprobe_src):
        shutil.copy2(src, bin_dir / src.name)
        print(f"已内置: bin/{src.name}")

    # 内置离线 PP-OCRv5 模型目录
    src_models = PROJECT_DIR / "models" / "ppocr_v5"
    dst_models = base_dir / "models" / "ppocr_v5"
    _sync_dir(src_models, dst_models)
    for m_name in ("ch_PP-OCRv5_det_mobile.onnx", "ch_PP-OCRv5_rec_server.onnx", "ch_PP-OCRv5_rec_mobile.onnx"):
        mp = dst_models / m_name
        if not mp.exists() or mp.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"缺少或损坏的离线 PP-OCRv5 模型: {mp}")
        print(f"已内置模型: models/ppocr_v5/{m_name} ({mp.stat().st_size / 1024 / 1024:.2f} MB)")

    # 预置 models/llm 引导文件（不拷贝任何 .gguf 大模型）
    llm_drop_dst = base_dir / "models" / "llm"
    llm_drop_dst.mkdir(parents=True, exist_ok=True)
    guide_src = PROJECT_DIR / "models" / "llm" / "把下载好的GGUF模型放这里.txt"
    if guide_src.is_file():
        shutil.copy2(guide_src, llm_drop_dst / guide_src.name)

    # 预置多地方言词典、选区预设模板与默认翻译配置（不含任何私钥）
    sys.path.insert(0, str(PROJECT_DIR))
    from core.runtime_paths import DEFAULT_GLOSSARY, DEFAULT_PRESETS, DEFAULT_TRANSLATOR_CONFIG
    cfg_dir = base_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "presets.json").write_text(json.dumps(DEFAULT_PRESETS, ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg_dir / "glossary.txt").write_text(DEFAULT_GLOSSARY.strip() + "\n", encoding="utf-8")
    (cfg_dir / "translator.json").write_text(json.dumps(DEFAULT_TRANSLATOR_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    # 补齐 DirectML.dll
    import onnxruntime
    ort_capi = Path(onnxruntime.__file__).resolve().parent / "capi"
    dst_capi = base_dir / "_internal" / "onnxruntime" / "capi"
    dst_capi.mkdir(parents=True, exist_ok=True)
    for dll_name in ("DirectML.dll", "onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        src_dll = ort_capi / dll_name
        dst_dll = dst_capi / dll_name
        if src_dll.exists() and not dst_dll.exists():
            shutil.copy2(src_dll, dst_dll)
            print(f"补齐 DLL: {dll_name}")
    print("DirectML.dll 就绪:", (dst_capi / "DirectML.dll").exists())

    lite_dir = PROJECT_DIR / "dist" / LITE_DIST_NAME
    with_ollama_dir = PROJECT_DIR / "dist" / WITH_OLLAMA_DIST_NAME

    if args.edition in ("lite", "both"):
        _sync_dir(base_dir, lite_dir)
        (lite_dir / "使用说明.txt").write_text(README_LITE_TEXT, encoding="utf-8")
        print(f"[版本1/2] 标准轻量版就绪: {lite_dir}")

    if args.edition in ("with-ollama", "both"):
        _sync_dir(base_dir, with_ollama_dir)
        _bundle_ollama_runtime(with_ollama_dir)
        (with_ollama_dir / "使用说明.txt").write_text(README_WITH_OLLAMA_TEXT, encoding="utf-8")
        print(f"[版本2/2] 内置 Ollama 全能版就绪: {with_ollama_dir}")

    # 同步覆盖桌面唯一交付目录（双版/全能版优先交付内置 Ollama 全能版）
    desktop_src = with_ollama_dir if args.edition in ("with-ollama", "both") else lite_dir
    subprocess.run(["taskkill", "/F", "/IM", f"{APP_NAME}.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    time.sleep(0.8)
    _sync_dir(desktop_src, DESKTOP_RELEASE_DIR)
    print(f"=== 桌面正式目录已更新 ({desktop_src.name}) -> {DESKTOP_RELEASE_DIR} ===")


if __name__ == "__main__":
    main()
