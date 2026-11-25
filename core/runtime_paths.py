import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_PRESETS: dict[str, dict] = {
    "横屏常规底部 (16:9)": {
        "main_roi": [0.08, 0.81, 0.92, 0.92],
        "main_white_only": True,
        "extract_notes": True,
        "note_roi": [0.00, 0.00, 1.00, 1.00],
        "note_min_height_ratio": 0.038,
    },
    "竖屏常规底部 (9:16 避开抖音上下UI)": {
        "main_roi": [0.05, 0.72, 0.95, 0.88],
        "main_white_only": True,
        "extract_notes": True,
        "note_roi": [0.00, 0.10, 1.00, 0.85],
        "note_min_height_ratio": 0.032,
    },
    "横屏老剪辑偏高字幕 (双行适用)": {
        "main_roi": [0.08, 0.74, 0.92, 0.91],
        "main_white_only": True,
        "extract_notes": True,
        "note_roi": [0.00, 0.00, 1.00, 0.90],
        "note_min_height_ratio": 0.038,
    },
}

DEFAULT_GLOSSARY = """【90年代北方农村 & 剧情专属】
- 戴建国 = Dave Jianguo（或 Dave）
- 戴建国奖 = the "Dave Jianguo Award"
- 向日葵 / 豌豆射手 / 寒冰射手 = Sunflower / Peashooter / Snow Pea
- 赶大集 / 赶集 = village fair / market day
- 串门 = drop by the neighbor's
- 土炕 / 热炕头 = warm kang bed / heated brick bed
- 村大队 / 村长 = village committee / village chief
- 供销社 = local co-op / village supply store
- 整点小烧 / 喝两盅 = pour a cup of baijiu / have a couple of shots

【东北方言高频】
- 扯犊子 = talking nonsense / bullshitting
- 老鼻子了 = tons of / a whole bunch of
- 埋汰 = messy / shabby / filthy
- 消停 = chill out / settle down
- 磨叽 / 吭哧瘪肚 = dragging your feet / beating around the bush
- 稀罕 = really like / fancy
- 可不咋地 = you bet / couldn't agree more

【川渝方言高频】
- 巴适 / 安逸 = awesome / super cozy / hits the spot
- 要不得 = no way / awful / unacceptable
- 勒个是啥子 / 搞毛呢 = what on earth is this? / what's going on?
- 摆龙门阵 = shoot the breeze / chat away
- 想爪子 = what do you want? / looking for trouble?
- 瓜兮兮 = goofy / clueless / dorky

【粤语高频】
- 搞掂 = nailed it / sorted / done deal
- 吹水 = shooting the breeze / bullshitting
- 冇得顶 = top-notch / awesome / second to none
- 大镬 = big trouble / we're screwed
- 蛇王 = slacking off / slacker
- 犀利 = impressive / badass
- 唔该 = thanks / excuse me

【云南方言高频】
- 板扎 = spot on / awesome / great
- 乱哄哩啰 = a hot mess / all over the place
- 扯把子 / 赶马 = full of hot air / talking crap
- 整球不得 = hopeless / no way in hell
- 妥妥滴 = totally nailed it / you got it

【湖南湖北方言高频】
- 搞么介 / 搞么名堂 = what are you up to? / what's the fuss?
- 霸蛮 = stubbornly persisting / forcing it through
- 不服周 = refuse to back down / won't take it lying down
- 满哥 = bro / buddy / dude
- 苕 / 信球 = dummy / dork / dimwit
- 恰饭 = grab a bite / eat"""

DEFAULT_TRANSLATOR_CONFIG: dict = {
    "provider": "ollama",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M",
    "api_base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "api_model": "deepseek-chat",
    "batch_size": 12,
    "temperature": 0.1,
    "skip_existing": True,
    "unload_after_done": True,
    "auto_detect_dialect": True,
}

_LEGACY_GLOSSARY_PREFIX = "- 戴建国 = Dave Jianguo（或 Dave）"


def get_app_dir() -> Path:
    """获取程序运行根目录（冻结 EXE 时为可执行文件所在文件夹，开发态为源码根目录）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_models_dir() -> Path:
    """获取离线 PP-OCRv5 模型目录（优先绿色包/工程根目录 models/ppocr_v5，其次 _MEIPASS）。"""
    app_dir = get_app_dir()
    candidates = [app_dir / "models" / "ppocr_v5"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "models" / "ppocr_v5")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def get_llm_drop_dir() -> Path:
    """获取用户存放自定义本地 GGUF 翻译模型的目录（绿色包根目录 models/llm）。"""
    return get_app_dir() / "models" / "llm"


def get_embedded_ollama_dir() -> Path:
    """获取绿色包内置 Ollama 运行时目录（runtime/ollama）。"""
    return get_app_dir() / "runtime" / "ollama"


def get_embedded_ollama_exe() -> Optional[Path]:
    """若绿色包自带了 runtime/ollama/ollama.exe 则返回其路径，否则返回 None。"""
    exe = get_embedded_ollama_dir() / "ollama.exe"
    return exe if exe.is_file() else None


def get_embedded_ollama_models_dir() -> Path:
    """获取内置 Ollama 专用模型库存储目录（runtime/ollama_models），实现整包完全绿色隔离。"""
    return get_app_dir() / "runtime" / "ollama_models"


def get_ffmpeg_exe() -> str:
    """优先使用绿色包自带的 bin/ffmpeg.exe，其次回退系统 PATH。"""
    app_dir = get_app_dir()
    candidates = [
        app_dir / "bin" / "ffmpeg.exe",
        app_dir / "ffmpeg.exe",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "bin" / "ffmpeg.exe")
    for p in candidates:
        if p.exists():
            return str(p)
    found = shutil.which("ffmpeg")
    return found if found else "ffmpeg"


def get_ffprobe_exe() -> str:
    """优先使用绿色包自带的 bin/ffprobe.exe，其次回退系统 PATH。"""
    app_dir = get_app_dir()
    candidates = [
        app_dir / "bin" / "ffprobe.exe",
        app_dir / "ffprobe.exe",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "bin" / "ffprobe.exe")
    for p in candidates:
        if p.exists():
            return str(p)
    found = shutil.which("ffprobe")
    return found if found else "ffprobe"


def get_creation_flags() -> int:
    """Windows 下隐藏子进程控制台黑框。"""
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def run_silent(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """静默执行子进程命令，防止 --windowed 模式下出现 WinError 6 或闪黑框。"""
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    if os.name == "nt":
        kwargs.setdefault("creationflags", get_creation_flags())
    return subprocess.run(cmd, **kwargs)


def popen_silent(cmd: list[str], **kwargs) -> subprocess.Popen:
    """静默启动可中断的子进程（用于支持用户中途取消 FFmpeg 压制）。"""
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    if os.name == "nt":
        kwargs.setdefault("creationflags", get_creation_flags())
    return subprocess.Popen(cmd, **kwargs)


def get_config_dir() -> Path:
    cfg_dir = get_app_dir() / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def load_presets() -> dict[str, dict]:
    """读取选区预设，首次自动写入默认模板。"""
    p = get_config_dir() / "presets.json"
    if not p.exists():
        save_presets(DEFAULT_PRESETS)
        return json.loads(json.dumps(DEFAULT_PRESETS))
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return json.loads(json.dumps(DEFAULT_PRESETS))


def save_presets(presets: dict[str, dict]) -> None:
    p = get_config_dir() / "presets.json"
    p.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")


def load_glossary() -> str:
    """读取用户可编辑的方言词典（若为旧版默认词典则自动升级为六大区新版词典）。"""
    p = get_config_dir() / "glossary.txt"
    if not p.exists():
        save_glossary(DEFAULT_GLOSSARY)
        return DEFAULT_GLOSSARY
    content = p.read_text(encoding="utf-8")
    if content.strip().startswith(_LEGACY_GLOSSARY_PREFIX) and "【东北方言高频】" not in content and "- 磨叽 = drag your feet / stall" in content:
        save_glossary(DEFAULT_GLOSSARY)
        return DEFAULT_GLOSSARY
    return content


def save_glossary(text: str) -> None:
    p = get_config_dir() / "glossary.txt"
    p.write_text(text.strip() + "\n", encoding="utf-8")


def load_translator_config() -> dict:
    """读取本地 AI 自动翻译配置（Ollama / 云端 OpenAI 兼容 API），缺失字段自动补齐默认值。"""
    p = get_config_dir() / "translator.json"
    cfg = dict(DEFAULT_TRANSLATOR_CONFIG)
    if not p.exists():
        return cfg
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k in DEFAULT_TRANSLATOR_CONFIG:
                if k in raw:
                    cfg[k] = raw[k]
    except Exception:
        pass
    return cfg


def save_translator_config(cfg: dict) -> None:
    """将翻译引擎配置保存至 config/translator.json。"""
    p = get_config_dir() / "translator.json"
    data = {k: cfg.get(k, v) for k, v in DEFAULT_TRANSLATOR_CONFIG.items()}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
