import json
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.runtime_paths import get_ffmpeg_exe, get_ffprobe_exe, popen_silent, run_silent

# ================= 默认多维去重与样式配置 =================
DEFAULT_DEDUP_OPTIONS: dict[str, bool] = {
    "trim_head_tail": True,      # 智能掐头去尾（保护首句和片尾金句）
    "micro_speed": True,         # 全片 1.025x 微变速（音画字幕100%同步）
    "micro_zoom": True,          # 全屏 1.03x 微缩放裁剪（不镜像保护中文）
    "color_grade": True,         # 轻电影感调色 + 锐化
    "film_grain": True,          # 90年代轻胶片颗粒
    "subtle_vignette": True,     # 极轻微电影镜头暗角
    "audio_fingerprint": True,   # 音频防查重（EQ+微空间感重编码）
    "strip_metadata": True,      # 清空老抖音元数据
    "micro_rotate": False,       # 进阶可选：0.6度微旋转补边
    "blur_frame": False,         # 进阶可选：86%居中主画面 + 高斯模糊暗背景 + 顶部标题栏
}

RAW_DEDUP_OPTIONS: dict[str, bool] = {k: False for k in DEFAULT_DEDUP_OPTIONS}

DEDUP_LABELS: list[tuple[str, str]] = [
    ("trim_head_tail", "智能掐头去尾（不吞台词）"),
    ("micro_speed", "1.025x 微变速（防时长查重）"),
    ("micro_zoom", "1.03x 微缩放（不镜像护中文）"),
    ("color_grade", "电影感轻调色 + 锐化"),
    ("film_grain", "90年代怀旧轻胶片颗粒"),
    ("subtle_vignette", "极轻微镜头暗角"),
    ("audio_fingerprint", "音频改声纹（EQ+人声提亮）"),
    ("strip_metadata", "彻底清除原平台元数据"),
    ("micro_rotate", "[进阶] 0.6度微旋转补边"),
    ("blur_frame", "[进阶] 86%居中+模糊暗框+顶标"),
]

# ASS 使用 &H00BBGGRR 顺序，PIL 使用 (R, G, B)
COLOR_PRESETS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "暖白": ("&H00DCF6FF", (255, 246, 220)),
    "纯白": ("&H00FFFFFF", (255, 255, 255)),
    "柔白": ("&H00FFF6DC", (220, 246, 255)),
    "亮金黄": ("&H0080EEFF", (255, 238, 128)),
}

ENCODER_CHOICES: list[tuple[str, str]] = [
    ("auto", "自动识别显卡（推荐）"),
    ("hevc_amf", "AMD 显卡 H.265 (hevc_amf)"),
    ("h264_amf", "AMD 显卡 H.264 (h264_amf)"),
    ("hevc_nvenc", "NVIDIA 显卡 H.265 (hevc_nvenc)"),
    ("h264_nvenc", "NVIDIA 显卡 H.264 (h264_nvenc)"),
    ("x264_cpu", "CPU 视觉无损 (libx264 CRF18)"),
]

AUTO_PLACE_CHOICES: list[tuple[str, str]] = [
    ("auto", "自动防重叠（优先下方，太低放上方）[推荐]"),
    ("always_above", "自动紧贴原中文字幕上方"),
    ("always_below", "自动紧贴原中文字幕下方"),
    ("manual", "手动自由定位 (X, Y 滑块 / 左图拖拽)"),
]

SPEED_FACTOR = 1.025
FONT_NAME = "Arial"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv"}
# ========================================================


class BurnCancelled(Exception):
    """用户主动中止烧录时抛出。"""


@dataclass
class BurnStyleConfig:
    """所见即所得双语字幕排版样式配置（支持主字幕与单条花字自由 X/Y 定位）。"""
    margin_v_ratio: float = 0.052      # 手动模式下距底边高度比（兼容字段，与 pos_y_ratio 联动）
    font_size_ratio: float = 0.036     # 主英文字幕字号比（0.018 ~ 0.070）
    font_color_name: str = "暖白"
    outline_width: float = 3.0         # 描边粗细
    use_semi_bg_box: bool = False      # 半透明黑底框
    note_scale_mult: float = 1.0       # 花字英文字号倍率（0.6 ~ 1.5）
    note_gap_px: int = 8               # 花字英文与原中文的上下间距（px）
    top_banner_text: str = "90s Rural China · Official"
    auto_zoom_margin_comp: bool = True  # 回退高度开启 micro_zoom 时自动微补偿下移
    auto_place_mode: str = "auto"      # "auto" | "always_above" | "always_below" | "manual"
    auto_gap_ratio: float = 0.012      # 自动排版时英文与原中文字幕之间的安全间距比例（默认 1.2%）
    pos_x_ratio: float = 0.50          # 主英文字幕水平中心 X（0.10 ~ 0.90，所有模式均可横移）
    pos_y_ratio: float = 0.948         # 手动自由定位下主英文字幕底边 Y（0.08 ~ 0.985）
    note_pos_overrides: dict[str, list[float]] = field(default_factory=dict)  # 单条花字手动位置 {"1": [cx, cy]}


def detect_best_encoder() -> str:
    """实际探测本机可用硬件编码器（优先 AMD AMF -> NVIDIA NVENC -> CPU x264）。"""
    ffmpeg = get_ffmpeg_exe()
    for enc in ("hevc_amf", "hevc_nvenc"):
        cmd = [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
            "-c:v", enc, "-f", "null", "-",
        ]
        try:
            res = run_silent(cmd, capture_output=True, timeout=8)
            if res.returncode == 0:
                return enc
        except Exception:
            pass
    return "x264_cpu"


def get_video_info(video_path: Path) -> dict:
    """使用内置 ffprobe 读取真实宽高、帧率、时长与音轨状态。"""
    ffprobe = get_ffprobe_exe()
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        str(video_path),
    ]
    res = run_silent(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    data = json.loads(res.stdout)

    v_stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    w, h = int(v_stream["width"]), int(v_stream["height"])
    rotation = 0
    tags = v_stream.get("tags", {})
    if "rotate" in tags:
        rotation = int(tags["rotate"])
    for sd in v_stream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if abs(rotation) in (90, 270):
        w, h = h, w

    w = w - (w % 2)
    h = h - (h % 2)

    fps_str = v_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 30.0
    fps = min(60.0, max(24.0, round(fps)))

    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    return {"w": w, "h": h, "fps": int(fps), "duration": duration, "has_audio": has_audio}


def parse_srt_time_to_sec(t: str) -> float:
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])


def srt_time_to_ass(t: str) -> str:
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    h = int(parts[0])
    m = parts[1]
    s_parts = parts[2].split(".")
    s = s_parts[0]
    cs = (s_parts[1] + "00")[:2] if len(s_parts) > 1 else "00"
    return f"{h}:{m}:{s}.{cs}"


def parse_srt(srt_path: Optional[Path]) -> list[dict]:
    if not srt_path or not srt_path.exists():
        return []
    raw = srt_path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    blocks = re.split(r"\r?\n\s*\r?\n", raw)
    events = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_idx = 1 if "-->" in lines[1] else (0 if "-->" in lines[0] else -1)
        if time_idx == -1:
            continue
        start_s, end_s = lines[time_idx].split("-->")
        text = r"\N".join(lines[time_idx + 1 :])
        if text:
            idx_val = int(lines[0]) if time_idx == 1 and lines[0].isdigit() else (len(events) + 1)
            events.append({
                "index": idx_val,
                "start_sec": parse_srt_time_to_sec(start_s),
                "end_sec": parse_srt_time_to_sec(end_s),
                "start_ass": srt_time_to_ass(start_s),
                "end_ass": srt_time_to_ass(end_s),
                "text": text,
            })
    return events


def _has_main_overlap(n_start: float, n_end: float, main_events: list[dict]) -> bool:
    for ev in main_events:
        if min(n_end, ev["end_sec"]) > max(n_start, ev["start_sec"]):
            return True
    return False


def _estimate_text_width(text: str, font_size: int) -> float:
    units = 0.0
    for ch in text:
        if ch in " ilI.,;:'\"!|()":
            units += 0.34
        elif ch in "mwMW@#":
            units += 0.80
        elif ch.isupper():
            units += 0.68
        else:
            units += 0.56
    return units * font_size


def load_or_detect_zh_band(
    video_path: Path,
    layout_json_path: Optional[Path] = None,
    shared_ocr=None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional[tuple[float, float]]:
    """
    读取已有 *_main.layout.json 或实时采样探测原中文字幕竖直区间 (zh_y0, zh_y1)。
    若通过实时探测命中且提供了 layout_json_path，自动缓存写入磁盘供后续秒读。
    """
    if layout_json_path is not None and layout_json_path.exists():
        try:
            data = json.loads(layout_json_path.read_text(encoding="utf-8"))
            y0 = float(data["zh_y0"])
            y1 = float(data["zh_y1"])
            if 0.35 <= y0 < y1 <= 1.0:
                return (y0, y1)
        except Exception:
            pass

    from core.extractor import detect_zh_subtitle_band

    band = detect_zh_subtitle_band(video_path, shared_ocr=shared_ocr)
    if band is not None:
        if log_cb:
            log_cb(f"[自动测位] {video_path.name} 原中文字幕 Y={band[0]*100:.1f}%~{band[1]*100:.1f}%")
        if layout_json_path is not None:
            try:
                layout_json_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "video_name": video_path.name,
                    "zh_y0": band[0],
                    "zh_y1": band[1],
                    "source": "auto_probe_on_burn",
                }
                layout_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
    return band


def compute_effective_main_layout(
    w: int,
    h: int,
    style: BurnStyleConfig,
    opts: dict[str, bool],
    zh_band: Optional[tuple[float, float]] = None,
) -> dict:
    """
    预览绘制与实际 ASS 烧录共用同一套主英文字幕布局公式，保证所见即所得 1 像素不差。
    支持根据原中文字幕位置及缩放裁剪参数全自动防重叠定位（放下方或自动改放上方）。
    """
    is_blur_frame = bool(opts.get("blur_frame"))
    is_zoom = bool(opts.get("micro_zoom")) or bool(opts.get("micro_rotate"))
    total_scale = (
        (1.03 if opts.get("micro_zoom") else 1.0)
        * (1.025 if opts.get("micro_rotate") else 1.0)
        * (0.86 if is_blur_frame else 1.0)
    )

    f_ratio = style.font_size_ratio * (0.90 if is_blur_frame else 1.0)
    fontsize = max(18, int(round(h * f_ratio)))

    pos_x_ratio = max(0.10, min(0.90, float(style.pos_x_ratio)))
    px = int(round(w * pos_x_ratio))
    safe_side_pad = max(16, int(round(w * 0.03)))
    margin_lr = int(round(w * 0.05))
    sym_half = max(90, min(px - safe_side_pad, w - safe_side_pad - px))
    max_line_w = float(min(w - 2 * margin_lr, 2 * sym_half))

    placement = "manual"
    zy0: Optional[float] = None
    zy1: Optional[float] = None
    en_h_1line = f_ratio * 1.18

    if style.auto_place_mode != "manual" and zh_band is not None:
        zy0 = max(0.0, min(1.0, 0.5 + (zh_band[0] - 0.5) * total_scale))
        zy1 = max(0.0, min(1.0, 0.5 + (zh_band[1] - 0.5) * total_scale))
        gap = max(0.008, float(style.auto_gap_ratio))
        min_bottom_pad = 0.016
        bottom_space = max(0.0, 1.0 - zy1)
        need_below = min_bottom_pad + en_h_1line + gap

        if style.auto_place_mode == "always_above":
            placement = "above"
        elif style.auto_place_mode == "always_below":
            placement = "below"
        else:
            placement = "below" if (bottom_space >= need_below) else "above"

        if placement == "above":
            m_ratio = max(0.020, min(0.60, 1.0 - (zy0 - gap)))
            pos_y_ratio = max(0.08, min(0.985, 1.0 - m_ratio))
            reason = f"原中文(缩放后) {zy0*100:.1f}%~{zy1*100:.1f}%(底空{bottom_space*100:.1f}%) → 自动放原字幕上方(Y={pos_y_ratio*100:.1f}%)"
        else:
            m_ratio = max(min_bottom_pad, min(0.50, 1.0 - (zy1 + gap + en_h_1line)))
            pos_y_ratio = max(0.08, min(0.985, 1.0 - m_ratio))
            reason = f"原中文(缩放后) {zy0*100:.1f}%~{zy1*100:.1f}%(底空{bottom_space*100:.1f}%) → 自动放原字幕下方(Y={pos_y_ratio*100:.1f}%)"
        if abs(pos_x_ratio - 0.50) > 0.003:
            reason += f" [X={pos_x_ratio*100:.1f}%]"
        main_zone_top_px = int(round(h * min(zy0, 1.0 - m_ratio - en_h_1line)))
    else:
        raw_y = float(style.pos_y_ratio)
        raw_mv = float(style.margin_v_ratio)
        if style.auto_place_mode == "manual":
            # 自由定位模式优先读取 pos_y_ratio（若仅改了旧字段 margin_v_ratio 则自动换算兼容）
            if abs(raw_y - 0.948) > 1e-4 or abs(raw_mv - 0.052) <= 1e-4:
                pos_y_ratio = max(0.08, min(0.985, raw_y))
            else:
                pos_y_ratio = max(0.08, min(0.985, 1.0 - raw_mv))
            m_ratio = 1.0 - pos_y_ratio
            reason = f"手动自由定位 X={pos_x_ratio*100:.1f}%, Y={pos_y_ratio*100:.1f}%"
        else:
            if is_blur_frame:
                m_ratio = min(0.024, raw_mv)
            elif is_zoom and style.auto_zoom_margin_comp:
                m_ratio = max(0.010, raw_mv - 0.008)
            else:
                m_ratio = raw_mv
            pos_y_ratio = max(0.08, min(0.985, 1.0 - m_ratio))
            reason = "未检出原中文字幕，回退默认高度"
        main_zone_top_px = int(round(h * max(0.02, pos_y_ratio - en_h_1line * 1.4)))

    py = int(round(h * pos_y_ratio))
    margin_v = max(0, h - py)
    ass_color, rgb_color = COLOR_PRESETS.get(style.font_color_name, COLOR_PRESETS["暖白"])
    outline_px = max(1.0, float(style.outline_width))

    return {
        "fontsize": fontsize,
        "margin_v": margin_v,
        "margin_v_ratio": m_ratio,
        "margin_lr": margin_lr,
        "px": px,
        "py": py,
        "pos_x_ratio": pos_x_ratio,
        "pos_y_ratio": pos_y_ratio,
        "max_line_w": max_line_w,
        "ass_color": ass_color,
        "rgb_color": rgb_color,
        "outline_px": outline_px,
        "is_blur_frame": is_blur_frame,
        "placement": placement,
        "zy0": zy0,
        "zy1": zy1,
        "reason": reason,
        "main_zone_top_px": main_zone_top_px,
    }


def compute_note_layout(
    pos_info: dict,
    en_text: str,
    w: int,
    h: int,
    style: BurnStyleConfig,
    opts: dict[str, bool],
    has_main_overlap: bool,
    main_zone_top_px: Optional[int] = None,
) -> dict:
    """预览与 ASS 共用同一套花字定位与防出画计算公式（支持单条花字手动坐标覆写）。"""
    is_blur_frame = bool(opts.get("blur_frame"))
    total_scale = (
        (1.03 if opts.get("micro_zoom") else 1.0)
        * (1.025 if opts.get("micro_rotate") else 1.0)
        * (0.86 if is_blur_frame else 1.0)
    )
    margin_x = int(round(w * 0.03))
    max_allowed_w = w - 2 * margin_x

    cx_p = w / 2.0 + (float(pos_info["cx"]) - w / 2.0) * total_scale
    y0_p = h / 2.0 + (float(pos_info["y0"]) - h / 2.0) * total_scale
    y1_p = h / 2.0 + (float(pos_info["y1"]) - h / 2.0) * total_scale
    bh_p = float(pos_info["bh"]) * total_scale

    raw_fs = bh_p * 0.44 * style.note_scale_mult
    note_fs = int(round(max(24.0, min(68.0, raw_fs))))
    clean_en = en_text.replace(r"\N", " ").strip()

    est_w = _estimate_text_width(clean_en, note_fs)
    if est_w > max_allowed_w and est_w > 0:
        note_fs = max(20, int(note_fs * (max_allowed_w / est_w)))
        est_w = _estimate_text_width(clean_en, note_fs)

    half_w = est_w / 2.0
    note_idx = int(pos_info.get("index", 0) or 0)
    override = (style.note_pos_overrides or {}).get(str(note_idx))

    if isinstance(override, (list, tuple)) and len(override) >= 2:
        ov_x = max(0.05, min(0.95, float(override[0])))
        ov_y = max(0.04, min(0.96, float(override[1])))
        px = int(round(max(margin_x + half_w, min(w - margin_x - half_w, w * ov_x))))
        py = int(round(max(note_fs * 0.6 + 12, min(h - note_fs * 0.6 - 12, h * ov_y))))
        align = 5
        return {
            "index": note_idx,
            "align": align,
            "px": px,
            "py": py,
            "note_fs": note_fs,
            "text": clean_en,
            "is_manual": True,
            "nx_ratio": round(px / max(1, w), 4),
            "ny_ratio": round(py / max(1, h), 4),
        }

    px = int(round(max(margin_x + half_w, min(w - margin_x - half_w, cx_p))))

    gap = int(style.note_gap_px)
    py_below = int(round(y1_p + gap))
    hits_bottom_edge = (py_below + note_fs) > (h - 20)
    guard_top = min(int(h * 0.81), main_zone_top_px) if main_zone_top_px is not None else int(h * 0.81)
    hits_main_zone = has_main_overlap and ((py_below + note_fs) > guard_top) and (0.20 * w < cx_p < 0.80 * w)

    if hits_bottom_edge or hits_main_zone:
        align = 2
        py = int(round(max(note_fs + 16, y0_p - gap)))
        center_y = py - note_fs * 0.5
    else:
        align = 8
        py = py_below
        center_y = py + note_fs * 0.5

    return {
        "index": note_idx,
        "align": align,
        "px": px,
        "py": py,
        "note_fs": note_fs,
        "text": clean_en,
        "is_manual": False,
        "nx_ratio": round(px / max(1, w), 4),
        "ny_ratio": round(max(0.04, min(0.96, center_y / max(1, h))), 4),
    }


def _fit_en_text_for_width(text: str, base_fs: int, max_line_w: float, allow_wrap: bool = True) -> tuple[str, int]:
    """根据当前水平位置的最大对称行宽自适应缩窄字号与智能折行，防止拖偏左右时文字出画截断。"""
    clean = text.strip()
    if not clean:
        return "", base_fs
    norm = clean.replace("\n", r"\N")
    if r"\N" in norm:
        max_w = max((_estimate_text_width(seg.strip(), base_fs) for seg in norm.split(r"\N")), default=0.0)
        if max_w > max_line_w > 0:
            fit_fs = max(18, int(round(base_fs * 0.68)), int(base_fs * (max_line_w * 0.94 / max_w)))
            return norm, fit_fs
        return norm, base_fs

    est_w = _estimate_text_width(norm, base_fs)
    if est_w <= max_line_w or est_w <= 0:
        return norm, base_fs

    min_fs = max(18, int(round(base_fs * 0.72)))
    fit_fs = max(min_fs, int(base_fs * (max_line_w * 0.94 / est_w)))
    if _estimate_text_width(norm, fit_fs) <= max_line_w or (not allow_wrap):
        return norm, fit_fs

    # 拖到画面靠左/靠右较窄区域时按英文空格均匀断行
    words = norm.split()
    if len(words) < 2:
        return norm, max(16, int(base_fs * (max_line_w * 0.92 / est_w)))

    lines_out: list[str] = []
    cur = words[0]
    for wd in words[1:]:
        cand = f"{cur} {wd}"
        if _estimate_text_width(cand, min_fs) <= max_line_w * 0.96:
            cur = cand
        else:
            lines_out.append(cur)
            cur = wd
    lines_out.append(cur)
    wrapped = r"\N".join(lines_out)
    max_w = max((_estimate_text_width(seg, min_fs) for seg in lines_out), default=0.0)
    if max_w > max_line_w > 0:
        min_fs = max(16, int(min_fs * (max_line_w * 0.94 / max_w)))
    return wrapped, min_fs


def _fit_below_line_fontsize(text: str, base_fs: int, w: int, margin_lr: int) -> int:
    """兼容辅助函数：计算常规居中下方单行自适应字号。"""
    _, fs = _fit_en_text_for_width(text, base_fs, max(100.0, (w - 2 * margin_lr) * 0.96), allow_wrap=False)
    return fs


# ==================== 所见即所得实时预览绘制 ====================

def _load_preview_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc",
    ):
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size_px)
            except Exception:
                pass
    return ImageFont.load_default()


def render_burn_preview_with_meta(
    frame_bgr: np.ndarray,
    style: BurnStyleConfig,
    opts: dict[str, bool],
    sample_en_text: str = "The first Dave Jianguo Award goes to Dave Jianguo!",
    sample_note: Optional[dict] = None,
    zh_band: Optional[tuple[float, float]] = None,
) -> tuple[Image.Image, dict]:
    """在内存帧上所见即所得绘制双语英文字幕与花字，并返回拖拽锚点/边界框元数据。"""
    h, w = frame_bgr.shape[:2]
    work = frame_bgr.copy()

    if opts.get("micro_zoom"):
        wz = int(round(w * 1.03)) // 2 * 2
        hz = int(round(h * 1.03)) // 2 * 2
        resized = cv2.resize(work, (wz, hz), interpolation=cv2.INTER_LINEAR)
        x_off = (wz - w) // 2
        y_off = (hz - h) // 2
        work = resized[y_off : y_off + h, x_off : x_off + w]

    if opts.get("blur_frame"):
        bg = cv2.GaussianBlur(work, (0, 0), sigmaX=18)
        bg = np.clip(bg.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
        fw = int(round(w * 0.86)) // 2 * 2
        fh = int(round(h * 0.86)) // 2 * 2
        fg = cv2.resize(work, (fw, fh), interpolation=cv2.INTER_AREA)
        x_off = (w - fw) // 2
        y_off = (h - fh) // 2
        bg[y_off : y_off + fh, x_off : x_off + fw] = fg
        work = bg

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    layout = compute_effective_main_layout(w, h, style, opts, zh_band=zh_band)
    fs = layout["fontsize"]
    px = layout["px"]
    py = layout["py"]
    rgb_color = layout["rgb_color"]
    stroke_w = max(1, int(round(layout["outline_px"])))

    main_box: Optional[tuple[float, float, float, float]] = None
    main_anchor = (layout["pos_x_ratio"], layout["pos_y_ratio"])

    # 绘制主英文字幕（锚点为主字幕底边中心 (px, py)）
    if sample_en_text.strip():
        allow_wrap = layout["placement"] != "below"
        fit_ass, fs_eff = _fit_en_text_for_width(sample_en_text, fs, layout["max_line_w"], allow_wrap=allow_wrap)
        font = _load_preview_font(fs_eff)
        text = fit_ass.replace(r"\N", "\n").strip()
        bbox = draw.multiline_textbbox((px, 0), text, font=font, anchor="ma", align="center")
        text_h = bbox[3] - bbox[1]
        y_top = py - text_h - bbox[1]
        draw_y = py - bbox[3]

        bx0 = max(0, bbox[0] - 8)
        bx1 = min(w, bbox[2] + 8)
        by0 = max(0, (py - text_h) - 6)
        by1 = min(h, py + 6)
        main_box = (bx0 / max(1, w), by0 / max(1, h), bx1 / max(1, w), by1 / max(1, h))

        if style.use_semi_bg_box and not layout["is_blur_frame"]:
            pad_x = max(12, int(fs_eff * 0.35))
            pad_y = max(6, int(fs_eff * 0.18))
            draw.rounded_rectangle(
                [bbox[0] - pad_x, (py - text_h) - pad_y, bbox[2] + pad_x, py + pad_y],
                radius=8,
                fill=(0, 0, 0, 160),
            )
            draw.multiline_text(
                (px, draw_y), text, font=font, anchor="ma", align="center",
                fill=(*rgb_color, 255),
            )
        else:
            draw.multiline_text(
                (px, draw_y), text, font=font, anchor="ma", align="center",
                fill=(*rgb_color, 255),
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 255),
            )

    note_box: Optional[tuple[float, float, float, float]] = None
    note_anchor: Optional[tuple[float, float]] = None
    note_index: Optional[int] = None
    note_layout_meta: Optional[dict] = None

    # 若当前帧有对应花字注释，同时画出花字位置预览并记录可拖拽包围盒
    if sample_note and sample_note.get("pos_info") and sample_note.get("text_en"):
        nl = compute_note_layout(
            sample_note["pos_info"],
            sample_note["text_en"],
            w, h, style, opts,
            has_main_overlap=bool(sample_en_text.strip()),
            main_zone_top_px=layout["main_zone_top_px"],
        )
        n_font = _load_preview_font(nl["note_fs"])
        anchor = "mm" if nl["align"] == 5 else ("ma" if nl["align"] == 8 else "mb")
        draw.text(
            (nl["px"], nl["py"]),
            nl["text"],
            font=n_font,
            anchor=anchor,
            fill=(255, 238, 128, 255),
            stroke_width=max(2, stroke_w),
            stroke_fill=(0, 0, 0, 255),
        )
        nb = draw.textbbox((nl["px"], nl["py"]), nl["text"], font=n_font, anchor=anchor)
        note_box = (
            max(0, nb[0] - 8) / max(1, w),
            max(0, nb[1] - 6) / max(1, h),
            min(w, nb[2] + 8) / max(1, w),
            min(h, nb[3] + 6) / max(1, h),
        )
        note_anchor = (nl["nx_ratio"], nl["ny_ratio"])
        note_index = nl["index"]
        note_layout_meta = nl

    # 若启用画框模式，画顶部官方标语
    if layout["is_blur_frame"] and style.top_banner_text.strip():
        b_fs = max(18, int(round(h * 0.026)))
        b_font = _load_preview_font(b_fs)
        b_y = int(round(h * 0.018))
        draw.text(
            (w // 2, b_y),
            style.top_banner_text.strip(),
            font=b_font,
            anchor="ma",
            fill=(255, 244, 232, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    out_img = Image.alpha_composite(img, overlay).convert("RGB")
    meta = {
        "layout": layout,
        "main_box": main_box,
        "main_anchor": main_anchor,
        "note_box": note_box,
        "note_anchor": note_anchor,
        "note_index": note_index,
        "note_layout": note_layout_meta,
    }
    return out_img, meta


def render_burn_preview_pil(
    frame_bgr: np.ndarray,
    style: BurnStyleConfig,
    opts: dict[str, bool],
    sample_en_text: str = "The first Dave Jianguo Award goes to Dave Jianguo!",
    sample_note: Optional[dict] = None,
    zh_band: Optional[tuple[float, float]] = None,
) -> Image.Image:
    """兼容原有调用：返回所见即所得 PIL RGB Image。"""
    img, _ = render_burn_preview_with_meta(
        frame_bgr,
        style,
        opts,
        sample_en_text=sample_en_text,
        sample_note=sample_note,
        zh_band=zh_band,
    )
    return img


# ==================== ASS 生成与 FFmpeg 烧录 ====================

def build_bilingual_ass(
    main_events: list[dict],
    note_events: list[dict],
    pos_items: list[dict],
    out_ass: Path,
    w: int,
    h: int,
    style: BurnStyleConfig,
    opts: dict[str, bool],
    duration: float,
    zh_band: Optional[tuple[float, float]] = None,
) -> dict:
    layout = compute_effective_main_layout(w, h, style, opts, zh_band=zh_band)
    fontsize = layout["fontsize"]
    margin_v = layout["margin_v"]
    margin_lr = layout["margin_lr"]
    px = layout["px"]
    py = layout["py"]
    max_line_w = layout["max_line_w"]
    ass_color = layout["ass_color"]
    is_blur_frame = layout["is_blur_frame"]
    use_pos_tag = (style.auto_place_mode == "manual") or (abs(layout["pos_x_ratio"] - 0.50) > 0.002)

    banner_size = max(18, int(round(h * 0.026)))
    banner_margin_v = int(round(h * 0.018))

    if style.use_semi_bg_box and not is_blur_frame:
        border_style = 3
        outline = max(4, int(round(fontsize * 0.16)))
        shadow = 0
        outline_color = "&H60000000"
        back_color = "&H60000000"
    else:
        border_style = 1
        outline = round(layout["outline_px"], 1)
        shadow = max(1, int(round(fontsize * 0.04)))
        outline_color = "&H00000000"
        back_color = "&H70000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: EnSub,{FONT_NAME},{fontsize},{ass_color},&H000000FF,{outline_color},{back_color},1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1
Style: NoteSub,{FONT_NAME},36,&H0080EEFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{max(2.0, round(layout['outline_px'], 1))},1,8,24,24,20,1
Style: TopBanner,{FONT_NAME},{banner_size},&H00E8F4FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,8,{margin_lr},{margin_lr},{banner_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    if is_blur_frame and style.top_banner_text.strip():
        end_h = int(duration // 3600) + 1
        lines.append(f"Dialogue: 0,0:00:00.00,{end_h}:00:00.00,TopBanner,,0,0,0,,{style.top_banner_text.strip()}\n")

    allow_wrap = layout["placement"] != "below"
    for ev in main_events:
        ev_txt, fit_fs = _fit_en_text_for_width(ev["text"], fontsize, max_line_w, allow_wrap=allow_wrap)
        tags: list[str] = []
        if use_pos_tag:
            tags.append(rf"\an2\pos({px},{py})")
        if fit_fs != fontsize:
            tags.append(rf"\fs{fit_fs}")
        prefix = "{" + "".join(tags) + "}" if tags else ""
        lines.append(f"Dialogue: 0,{ev['start_ass']},{ev['end_ass']},EnSub,,0,0,0,,{prefix}{ev_txt}\n")

    if note_events and pos_items:
        pos_by_idx = {int(item["index"]): item for item in pos_items}
        for i, ev in enumerate(note_events):
            pos_info = pos_by_idx.get(ev["index"]) or (pos_items[i] if i < len(pos_items) else None)
            if not pos_info:
                continue
            has_overlap = _has_main_overlap(ev["start_sec"], ev["end_sec"], main_events)
            nl = compute_note_layout(
                pos_info, ev["text"], w, h, style, opts, has_overlap,
                main_zone_top_px=layout["main_zone_top_px"],
            )
            lines.append(
                f"Dialogue: 1,{ev['start_ass']},{ev['end_ass']},NoteSub,,0,0,0,,"
                f"{{\\q2\\an{nl['align']}\\fs{nl['note_fs']}\\pos({nl['px']},{nl['py']})}}{nl['text']}\n"
            )

    out_ass.write_text("".join(lines), encoding="utf-8-sig")
    return layout


def get_encoder_args(mode: str) -> list[str]:
    color_args = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-pix_fmt", "yuv420p"]
    if mode == "hevc_amf":
        return ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20", "-tag:v", "hvc1"] + color_args
    elif mode == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "18"] + color_args
    elif mode == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "22", "-b:v", "0", "-tag:v", "hvc1"] + color_args
    elif mode == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "20", "-b:v", "0"] + color_args
    else:
        return ["-c:v", "libx264", "-crf", "18", "-preset", "medium"] + color_args


def compute_safe_trim_bounds(main_events: list[dict], note_events: list[dict], duration: float) -> tuple[float, float]:
    all_events = main_events + note_events
    if duration <= 2.5 or not all_events:
        return 0.0, duration
    earliest_start = min(ev["start_sec"] for ev in all_events)
    latest_end = max(ev["end_sec"] for ev in all_events)
    t_start = min(0.25, max(0.0, earliest_start - 0.03))
    t_end = min(duration, max(latest_end + 0.10, duration - 0.35))
    if t_end <= t_start + 1.0:
        return 0.0, duration
    return t_start, t_end


def scan_burn_matches(video_dir: Path, zh_dir: Path, en_dir: Path) -> list[dict]:
    """扫描待烧录视频与英文字幕、花字坐标的匹配与条数校验状态。"""
    if not video_dir.exists():
        return []
    videos = sorted([p for p in video_dir.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    results: list[dict] = []

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        res_str = f"{w}x{h}" if (w > 0 and h > 0) else "未知"

        en_main = en_dir / f"{vid.stem}.srt"
        zh_main = zh_dir / f"{vid.stem}.srt"
        en_notes = en_dir / f"{vid.stem}_notes.srt"
        pos_json = zh_dir / f"{vid.stem}_notes.pos.json"
        layout_json = zh_dir / f"{vid.stem}_main.layout.json"

        en_main_cnt = len(parse_srt(en_main)) if en_main.exists() else 0
        zh_main_cnt = len(parse_srt(zh_main)) if zh_main.exists() else 0
        en_notes_cnt = len(parse_srt(en_notes)) if en_notes.exists() else 0

        pos_cnt = 0
        if pos_json.exists():
            try:
                pos_cnt = len(json.loads(pos_json.read_text(encoding="utf-8")).get("items", []))
            except Exception:
                pos_cnt = 0

        main_str = f"{en_main_cnt} 条" if en_main.exists() else "缺失"
        if pos_cnt > 0:
            notes_str = f"英文{en_notes_cnt} / 坐标{pos_cnt}" if en_notes.exists() else f"缺英文(坐标{pos_cnt})"
        elif en_notes.exists():
            notes_str = f"英文{en_notes_cnt}(无坐标)"
        else:
            notes_str = "无花字"

        if not en_main.exists() or en_main_cnt == 0:
            status = "❌ 缺主英文SRT"
            ready = False
        elif zh_main_cnt > 0 and en_main_cnt != zh_main_cnt:
            status = f"⚠ 主条数不一致({en_main_cnt}/{zh_main_cnt})"
            ready = True
        elif pos_cnt > 0 and not en_notes.exists():
            status = "⚠ 缺花字英文(仅烧主字幕)"
            ready = True
        elif pos_cnt > 0 and en_notes_cnt != pos_cnt:
            status = f"⚠ 花字条数不一致({en_notes_cnt}/{pos_cnt})"
            ready = True
        elif pos_cnt > 0 and en_notes_cnt == pos_cnt:
            status = "✅ 双轨就绪"
            ready = True
        else:
            status = "✅ 主字幕就绪"
            ready = True

        results.append({
            "video_path": vid,
            "stem": vid.stem,
            "resolution": res_str,
            "en_main_path": en_main if en_main.exists() else None,
            "en_main_count": en_main_cnt,
            "main_str": main_str,
            "en_notes_path": en_notes if en_notes.exists() else None,
            "en_notes_count": en_notes_cnt,
            "pos_json_path": pos_json if pos_json.exists() else None,
            "layout_json_path": layout_json,
            "pos_count": pos_cnt,
            "notes_str": notes_str,
            "status": status,
            "ready": ready,
        })

    return results


def burn_video(
    video_path: Path,
    main_srt_path: Path,
    out_path: Path,
    notes_srt_path: Optional[Path] = None,
    pos_json_path: Optional[Path] = None,
    mode: str = "auto",
    dedup_options: Optional[dict[str, bool]] = None,
    style: Optional[BurnStyleConfig] = None,
    test_sec: int = 0,
    log_cb: Optional[Callable[[str], None]] = None,
    item_progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    layout_json_path: Optional[Path] = None,
    zh_band: Optional[tuple[float, float]] = None,
    shared_ocr=None,
) -> bool:
    """使用内置 ffmpeg 烧录双轨英文字幕与多维去重，支持全自动防重叠排版、实时压制进度与安全中止。"""

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def iprog(pct: float, msg: str):
        if item_progress_cb:
            item_progress_cb(max(0.0, min(100.0, pct)), msg)

    opts = dict(DEFAULT_DEDUP_OPTIONS)
    if dedup_options is not None:
        opts.update(dedup_options)
    style_cfg = style or BurnStyleConfig()

    ffmpeg = get_ffmpeg_exe()
    enc_mode = detect_best_encoder() if mode == "auto" else mode
    info = get_video_info(video_path)
    w, h, fps, duration, has_audio = info["w"], info["h"], info["fps"], info["duration"], info["has_audio"]

    main_events = parse_srt(main_srt_path)
    note_events = parse_srt(notes_srt_path)
    pos_items: list[dict] = []
    if pos_json_path and pos_json_path.exists():
        pos_items = json.loads(pos_json_path.read_text(encoding="utf-8")).get("items", [])

    if not main_events and not note_events:
        raise RuntimeError(f"主台词和花字字幕均为空: {main_srt_path}")

    resolved_zh_band = zh_band
    if style_cfg.auto_place_mode != "manual" and resolved_zh_band is None:
        resolved_zh_band = load_or_detect_zh_band(
            video_path,
            layout_json_path=layout_json_path,
            shared_ocr=shared_ocr,
            log_cb=log_cb,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    speed = SPEED_FACTOR if opts.get("micro_speed") else 1.0
    do_trim = bool(opts.get("trim_head_tail"))
    do_audio_fp = bool(opts.get("audio_fingerprint"))
    do_blur_frame = bool(opts.get("blur_frame"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ass = Path(tmpdir) / "sub_v3.ass"
        layout = build_bilingual_ass(
            main_events, note_events, pos_items, tmp_ass, w, h, style_cfg, opts, max(duration, 600.0),
            zh_band=resolved_zh_band,
        )
        if style_cfg.auto_place_mode != "manual":
            log(f"[自动防撞排版] {video_path.name} | {layout['reason']}")
        else:
            log(f"[手动自由定位] {video_path.name} | {layout['reason']}")

        pre_filters: list[str] = []
        if opts.get("micro_zoom"):
            wz = int(round(w * 1.03)) // 2 * 2
            hz = int(round(h * 1.03)) // 2 * 2
            pre_filters.append(f"scale={wz}:{hz},crop={w}:{h},setsar=1")

        if opts.get("micro_rotate"):
            wz_r = int(round(w * 1.025)) // 2 * 2
            hz_r = int(round(h * 1.025)) // 2 * 2
            pre_filters.append(f"scale={wz_r}:{hz_r},rotate=0.6*PI/180:bilinear=1,crop={w}:{h},setsar=1")

        if opts.get("color_grade"):
            pre_filters.append("eq=contrast=1.04:brightness=0.01:saturation=1.07,colorbalance=rs=0.02:gs=0.01:bs=-0.02,unsharp=3:3:0.35")

        if opts.get("film_grain"):
            pre_filters.append("noise=alls=3:allf=t+u")

        if opts.get("subtle_vignette"):
            pre_filters.append("vignette=PI/7")

        pre_chain = ",".join(pre_filters)

        if do_blur_frame:
            fw = int(round(w * 0.86)) // 2 * 2
            fh = int(round(h * 0.86)) // 2 * 2
            fg_chain = f"scale={fw}:{fh},setsar=1" + (f",{pre_chain}" if pre_chain else "")
            video_base = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={w}:{h},setsar=1,gblur=sigma=18,eq=brightness=-0.22:contrast=1.05[bg_blur];"
                f"[fg]{fg_chain}[fg_main];"
                f"[bg_blur][fg_main]overlay=(W-w)/2:(H-h)/2,setsar=1,ass=sub_v3.ass"
            )
        else:
            video_base = "[0:v]" + (f"{pre_chain}," if pre_chain else "") + "setsar=1,ass=sub_v3.ass"

        t_start, t_end = (0.0, duration)
        if do_trim:
            t_start, t_end = compute_safe_trim_bounds(main_events, note_events, duration)
        if test_sec > 0 and duration > 0:
            t_end = min(t_end, t_start + test_sec * speed)

        expected_out_dur = max(1.0, (t_end - t_start) / speed) if (t_end > t_start) else max(1.0, float(test_sec or duration or 10.0))

        need_time_filter = do_trim or (test_sec > 0) or (abs(speed - 1.0) > 1e-6)
        if need_time_filter:
            trim_v = f"trim=start={t_start:.3f}:end={t_end:.3f},"
            pts_v = f"setpts=(PTS-STARTPTS)/{speed}" if abs(speed - 1.0) > 1e-6 else "setpts=PTS-STARTPTS"
            vf_full = f"{video_base},{trim_v}{pts_v},fps={fps}[vout]"
        else:
            vf_full = f"{video_base}[vout]"

        need_audio_reencode = has_audio and (need_time_filter or do_audio_fp)
        cmd = [ffmpeg, "-y", "-nostats", "-progress", "pipe:1", "-i", str(video_path.resolve())]

        if has_audio and need_audio_reencode:
            a_parts: list[str] = []
            if do_trim or (test_sec > 0):
                a_parts.append(f"atrim=start={t_start:.3f}:end={t_end:.3f},asetpts=PTS-STARTPTS")
            if abs(speed - 1.0) > 1e-6:
                a_parts.append(f"atempo={speed}")
            if do_audio_fp:
                a_parts.append("highpass=f=80,equalizer=f=2800:width_type=h:width=1000:g=1.5,aecho=0.8:0.88:12:0.05")
            af_full = ";[0:a]" + ",".join(a_parts) + "[aout]"
            cmd += ["-filter_complex", vf_full + af_full, "-map", "[vout]", "-map", "[aout]"]
            cmd += get_encoder_args(enc_mode)
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-filter_complex", vf_full, "-map", "[vout]"]
            if has_audio:
                cmd += ["-map", "0:a", "-c:a", "copy"]
            cmd += get_encoder_args(enc_mode)

        if opts.get("strip_metadata"):
            cmd += ["-map_metadata", "-1"]

        tmp_out = out_path.with_name(f".{out_path.stem}.building.mp4")
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass

        cmd += ["-movflags", "+faststart", str(tmp_out.resolve())]

        active_cnt = sum(1 for v in opts.values() if v)
        iprog(0.0, f"启动编码 ({enc_mode})...")
        log(f"[烧录中] {video_path.name} (主:{len(main_events)}条, 花字:{len(note_events)}条, 编码:{enc_mode}, 去重项:{active_cnt}个)...")

        proc = popen_silent(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        stderr_chunks: list[str] = []

        def _drain_stderr():
            if proc.stderr:
                for line in proc.stderr:
                    stderr_chunks.append(line)
                    if len(stderr_chunks) > 200:
                        stderr_chunks.pop(0)

        def _read_progress_stdout():
            if not proc.stdout:
                return
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        if us > 0:
                            cur_s = us / 1_000_000.0
                            pct = min(99.5, (cur_s / expected_out_dur) * 100.0)
                            cm, cs = divmod(int(cur_s), 60)
                            tm, ts = divmod(int(expected_out_dur), 60)
                            iprog(pct, f"压制中 {pct:.1f}% ({cm:02d}:{cs:02d}/{tm:02d}:{ts:02d})")
                    except ValueError:
                        pass
                elif line == "progress=end":
                    iprog(100.0, "压制完成")

        t_err = threading.Thread(target=_drain_stderr, daemon=True)
        t_prog = threading.Thread(target=_read_progress_stdout, daemon=True)
        t_err.start()
        t_prog.start()

        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                t_err.join(timeout=1)
                t_prog.join(timeout=1)
                if tmp_out.exists():
                    try:
                        tmp_out.unlink()
                    except Exception:
                        pass
                raise BurnCancelled(f"用户中止烧录: {video_path.name}")
            threading.Event().wait(0.2)

        t_err.join(timeout=1)
        t_prog.join(timeout=1)
        if proc.returncode != 0:
            err_tail = "".join(stderr_chunks)[-800:]
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except Exception:
                    pass
            log(f"[失败] {video_path.name}:\n{err_tail}")
            return False

        if not tmp_out.exists() or tmp_out.stat().st_size <= 10240:
            log(f"[失败] {video_path.name}: 临时输出文件缺失或体积异常")
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except Exception:
                    pass
            return False

        os.replace(tmp_out, out_path)
        iprog(100.0, "压制完成")
        log(f"[完成] -> {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True


def burn_batch(
    match_items: list[dict],
    out_dir: Path,
    mode: str = "auto",
    dedup_options: Optional[dict[str, bool]] = None,
    style: Optional[BurnStyleConfig] = None,
    test_sec: int = 0,
    skip_existing: bool = True,
    progress_cb: Optional[Callable[[int, int, float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    per_video_styles: Optional[dict[str, BurnStyleConfig]] = None,
) -> dict:
    """批量烧录匹配就绪的视频列表，支持每个视频独立字幕样式、日志自动落盘与实时进度回传。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "_burn_log.txt"

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    ready_items = [it for it in match_items if it.get("ready")]
    total = len(ready_items)
    done = 0
    skipped = 0
    failed = 0
    cancelled = False

    resolved_mode = detect_best_encoder() if mode == "auto" else mode
    default_style = style or BurnStyleConfig()
    log(f"开始批量烧录: 共 {total} 个就绪视频 | 编码器: {resolved_mode}")

    # 若存在需要自动防重叠排版且尚未保存 _main.layout.json 的视频，预先复用同一个 DirectML OCR 引擎
    shared_ocr = None
    need_probe = any(
        ((per_video_styles or {}).get(it["video_path"].name, default_style).auto_place_mode != "manual")
        and (it.get("layout_json_path") is None or not Path(it["layout_json_path"]).exists())
        for it in ready_items
    )
    if need_probe and ready_items:
        from core.extractor import create_ocr_engine
        shared_ocr = create_ocr_engine("auto")

    for idx, item in enumerate(ready_items, 1):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        vid: Path = item["video_path"]
        has_custom_style = bool(per_video_styles and vid.name in per_video_styles)
        vid_style = per_video_styles[vid.name] if has_custom_style else default_style
        if has_custom_style and vid_style is not None:
            log(f"[独立样式] {vid.name} (模式={vid_style.auto_place_mode}, 距底={vid_style.margin_v_ratio*100:.1f}%, 字号={vid_style.font_size_ratio*100:.1f}%)")

        if progress_cb:
            progress_cb(idx, total, 0.0, f"[{vid.name}] 准备压制...")

        suffix = f"_试压{test_sec}秒.mp4" if test_sec > 0 else "_双语成品.mp4"
        out_mp4 = out_dir / f"{vid.stem}{suffix}"

        if skip_existing and test_sec == 0 and out_mp4.exists():
            log(f"[跳过] 已存在成品: {out_mp4.name}")
            skipped += 1
            if progress_cb:
                progress_cb(idx, total, 100.0, f"[{vid.name}] 已跳过")
            continue

        try:
            def _item_prog(pct: float, stage: str):
                if progress_cb:
                    progress_cb(idx, total, pct, f"[{vid.name}] {stage}")

            ok = burn_video(
                vid,
                item["en_main_path"],
                out_mp4,
                notes_srt_path=item["en_notes_path"],
                pos_json_path=item["pos_json_path"],
                mode=resolved_mode,
                dedup_options=dedup_options,
                style=vid_style,
                test_sec=test_sec,
                log_cb=log,
                item_progress_cb=_item_prog,
                cancel_event=cancel_event,
                layout_json_path=item.get("layout_json_path"),
                shared_ocr=shared_ocr,
            )
            if ok:
                done += 1
            else:
                failed += 1
        except BurnCancelled:
            cancelled = True
            log(f"[已取消] 烧录已在 {vid.name} 处中止")
            break
        except Exception as exc:
            failed += 1
            log(f"[异常] {vid.name}: {exc}")

    if not cancelled and progress_cb:
        progress_cb(total, total, 100.0, "批量烧录全部完成")

    return {"done": done, "skipped": skipped, "failed": failed, "cancelled": cancelled}
