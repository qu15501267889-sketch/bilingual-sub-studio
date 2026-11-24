import difflib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

from core.runtime_paths import get_models_dir

# ================= 固定算法阈值配置 =================
COARSE_SAMPLE_FPS = 6.0
SIMILARITY_MERGE = 0.72
SIMILARITY_BOUNDARY = 0.70
GAP_BRIDGE_SEC = 0.40
MIN_MAIN_DURATION = 0.22

NOTE_SAMPLE_FPS = 3.0
NOTE_MIN_CONFIDENCE = 0.62
NOTE_SIMILARITY_CLUSTER = 0.65
NOTE_GAP_BRIDGE_SEC = 0.60
NOTE_MIN_DURATION = 0.35

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv"}
WATERMARK_KEYWORDS = (
    "AI视频", "仅供娱乐", "AI生成", "纯属虚构", "请勿模仿", "切勿模仿",
    "剧情演绎", "虚拟演绎", "内容虚构", "抖音", "TikTok", "Douyin",
    "快手", "小红书", "西瓜视频", "剪映", "即梦", "可灵", "豆包",
)
# ===================================================

OCR_PROFILE_CHOICES: list[tuple[str, str]] = [
    ("v5_server", "PP-OCRv5 高精度 (推荐·抗花哨字体)"),
    ("v5_mobile", "PP-OCRv5 极速版 (常规清晰字体)"),
]

_STRIP_PUNCT_RE = re.compile(r"""[\s,，.。!！?？、;；:：'"“”‘’()（）\[\]【】『』—…·~～\-]+""")
_LEADING_NOISE_RE = re.compile(r"""^[\s,，.。、;；:：!！?？"”’\]】』·~～]+""")
_TRAILING_NOISE_RE = re.compile(r"""[\s·~～`]+$""")
_VALID_CHAR_RE = re.compile(r"""[\u4e00-\u9fffA-Za-z0-9]""")
_WATERMARK_KW_CMPS = tuple(_STRIP_PUNCT_RE.sub("", kw).lower() for kw in WATERMARK_KEYWORDS)

_thread_local = threading.local()


class ExtractCancelled(Exception):
    """用户主动取消提取时抛出的异常。"""


@dataclass
class ExtractConfig:
    """可自由调节的字幕提取区域、过滤、OCR 精度与硬件加速配置。"""
    main_roi: tuple[float, float, float, float] = (0.08, 0.81, 0.92, 0.92)  # (x1, y1, x2, y2)
    main_white_only: bool = True
    extract_notes: bool = True
    note_roi: tuple[float, float, float, float] = (0.00, 0.00, 1.00, 1.00)  # (nx1, ny1, nx2, ny2)
    note_min_height_ratio: float = 0.038
    filter_watermark: bool = True
    skip_existing: bool = True
    accel_mode: str = "auto"  # "auto" | "directml" | "cpu"
    ocr_profile: str = "v5_server"  # "v5_server" | "v5_mobile"


def _matches_watermark_keyword(text: str) -> bool:
    """判断文本去标点转小写后是否包含常见平台或声明水印关键词。"""
    cmp_t = _STRIP_PUNCT_RE.sub("", text).lower()
    if not cmp_t:
        return False
    return any(kw in cmp_t for kw in _WATERMARK_KW_CMPS if kw)


def detect_ocr_accelerator() -> tuple[str, str]:
    """探测当前环境是否支持 DirectML 显卡加速（AMD / NVIDIA / Intel 通用）。"""
    providers = ort.get_available_providers()
    if "DmlExecutionProvider" in providers:
        return "directml", "PP-OCRv5 DirectML 显卡加速 (AMD/NVIDIA通用)"
    return "cpu", "PP-OCRv5 CPU 限流防卡顿模式"


def _build_ort_session(model_path: str, use_dml: bool) -> ort.InferenceSession:
    """按加速模式构造防卡顿且兼容动态尺寸的 ONNX Runtime 会话。"""
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.enable_cpu_mem_arena = False
    opt.inter_op_num_threads = 1

    if use_dml:
        # DirectML 动态尺寸推理必须关闭 mem_pattern，且严格顺序执行防显存冲突
        opt.enable_mem_pattern = False
        opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opt.intra_op_num_threads = 2
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        return ort.InferenceSession(model_path, sess_options=opt, providers=providers)

    # CPU 模式主动限流（最多占用约 1/4 核心），保障电脑同时顺畅操作不卡顿
    cpu_cnt = os.cpu_count() or 4
    opt.intra_op_num_threads = max(1, min(4, cpu_cnt // 4))
    return ort.InferenceSession(model_path, sess_options=opt, providers=["CPUExecutionProvider"])


class _RapidOCRV5Adapter:
    """
    离线 PP-OCRv5 统一适配器（基于 RapidOCR 3.x），屏蔽新版输出对象差异并接管 DirectML/CPU 会话，
    同时内置 DirectML 运行异常时的 CPU 自动兜底机制。
    """

    def __init__(self, accel_mode: str = "auto", ocr_profile: str = "v5_server"):
        models_dir = get_models_dir()
        det_path = models_dir / "ch_PP-OCRv5_det_mobile.onnx"
        rec_name = "ch_PP-OCRv5_rec_mobile.onnx" if ocr_profile == "v5_mobile" else "ch_PP-OCRv5_rec_server.onnx"
        rec_path = models_dir / rec_name

        if not det_path.exists() or not rec_path.exists():
            raise FileNotFoundError(
                f"缺少离线 PP-OCRv5 模型文件（期望位于 {models_dir}），请检查绿色包 models 目录是否完整。"
            )

        self.det_path = str(det_path)
        self.rec_path = str(rec_path)
        self.ocr_profile = ocr_profile
        self._fallback_done = False

        self._engine = RapidOCR(
            params={
                "Global.use_cls": False,
                "Global.width_height_ratio": -1,
                "Global.log_level": "error",
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Det.model_path": self.det_path,
                "Det.limit_type": "max",
                "Det.limit_side_len": 960,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE if ocr_profile == "v5_mobile" else ModelType.SERVER,
                "Rec.model_path": self.rec_path,
            }
        )
        # 显式提前加载检测与识别子模型，接管 ORT 会话选项
        self._engine._load_det_model()
        self._engine._load_rec_model()

        want_dml = accel_mode in ("auto", "directml") and ("DmlExecutionProvider" in ort.get_available_providers())
        self.backend = "cpu"
        if want_dml:
            try:
                self._engine.text_det.session.session = _build_ort_session(self.det_path, use_dml=True)
                self._engine.text_rec.session.session = _build_ort_session(self.rec_path, use_dml=True)
                self.backend = "directml"
                return
            except Exception:
                pass

        try:
            self._engine.text_det.session.session = _build_ort_session(self.det_path, use_dml=False)
            self._engine.text_rec.session.session = _build_ort_session(self.rec_path, use_dml=False)
        except Exception:
            pass

    def _fallback_to_cpu(self, err_msg: str) -> None:
        print(f"[OCR兜底提示] DirectML 推理回退至 CPU 限流模式: {err_msg}")
        self._engine.text_det.session.session = _build_ort_session(self.det_path, use_dml=False)
        self._engine.text_rec.session.session = _build_ort_session(self.rec_path, use_dml=False)
        self.backend = "cpu"
        self._fallback_done = True

    def __call__(self, img: np.ndarray, text_score: float = 0.55, box_thresh: float = 0.35):
        try:
            out = self._engine(img, use_cls=False, text_score=text_score, box_thresh=box_thresh)
        except Exception as exc:
            if self.backend == "directml" and not self._fallback_done:
                self._fallback_to_cpu(str(exc))
                out = self._engine(img, use_cls=False, text_score=text_score, box_thresh=box_thresh)
            else:
                raise

        items: list[tuple[list, str, float]] = []
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)
        if boxes is not None and txts is not None and scores is not None:
            for b, t, s in zip(boxes, txts, scores):
                pts = b.tolist() if hasattr(b, "tolist") else b
                items.append((pts, str(t), float(s)))
        return items, getattr(out, "elapse_list", None)


def create_ocr_engine(accel_mode: str = "auto", ocr_profile: str = "v5_server") -> tuple[_RapidOCRV5Adapter, str]:
    """创建离线 PP-OCRv5 识别引擎（修复宽条 ROI 漏检，支持 DirectML 显卡加速与防卡顿 CPU 限流）。"""
    adapter = _RapidOCRV5Adapter(accel_mode=accel_mode, ocr_profile=ocr_profile)
    return adapter, adapter.backend


def _get_cpu_worker_ocr(ocr_profile: str = "v5_server") -> _RapidOCRV5Adapter:
    if not hasattr(_thread_local, "ocr_map"):
        _thread_local.ocr_map = {}
    if ocr_profile not in _thread_local.ocr_map:
        _thread_local.ocr_map[ocr_profile], _ = create_ocr_engine("cpu", ocr_profile=ocr_profile)
    return _thread_local.ocr_map[ocr_profile]


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "'\"")


def _white_pixel_ratio(img: np.ndarray, box: list) -> float:
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    x0, x1 = max(0, min(xs)), min(img.shape[1], max(xs))
    y0, y1 = max(0, min(ys)), min(img.shape[0], max(ys))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    b, g, r = cv2.split(crop)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    pure_white = (mn > 195) & ((mx - mn) < 55)
    return float(np.mean(pure_white))


def read_roi_text_with_boxes(
    ocr: _RapidOCRV5Adapter,
    roi_img: np.ndarray,
    filter_white_only: bool = True,
) -> tuple[str, list[tuple[float, float]]]:
    """识别主台词 ROI 区域并按行/列有序拼接，同时返回保留文字框在 ROI 内的纵向坐标列表 [(y0_px, y1_px), ...]。"""
    if roi_img is None or roi_img.size == 0:
        return "", []
    res, _ = ocr(roi_img, text_score=0.55, box_thresh=0.35)
    if not res:
        return "", []

    items = []
    kept_y_boxes: list[tuple[float, float]] = []
    for box, txt, score in res:
        if float(score) < 0.55:
            continue
        clean = txt.strip()
        if not clean:
            continue
        if filter_white_only and _white_pixel_ratio(roi_img, box) < 0.04:
            continue
        ys = [float(p[1]) for p in box]
        xs = [float(p[0]) for p in box]
        y_center = sum(ys) / 4.0
        box_h = max(ys) - min(ys)
        x_left = min(xs)
        items.append((y_center, box_h, x_left, clean))
        kept_y_boxes.append((min(ys), max(ys)))

    if not items:
        return "", []

    items.sort(key=lambda x: x[0])
    lines: list[list[tuple[float, float, float, str]]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        cur_line = lines[-1]
        avg_y = sum(x[0] for x in cur_line) / len(cur_line)
        avg_h = sum(x[1] for x in cur_line) / len(cur_line)
        if abs(item[0] - avg_y) <= max(15.0, avg_h * 0.65):
            cur_line.append(item)
        else:
            lines.append([item])

    line_strs = []
    for line in lines:
        line.sort(key=lambda x: x[2])
        joined = ""
        for _, _, _, part in line:
            if not joined:
                joined = part
            elif _is_ascii_word_char(joined[-1]) and _is_ascii_word_char(part[0]):
                joined += " " + part
            else:
                joined += part
        line_strs.append(joined)

    return " ".join(line_strs).strip(), kept_y_boxes


def read_roi_text(ocr: _RapidOCRV5Adapter, roi_img: np.ndarray, filter_white_only: bool = True) -> str:
    """识别主台词 ROI 区域并按行/列有序拼接。"""
    return read_roi_text_with_boxes(ocr, roi_img, filter_white_only=filter_white_only)[0]


def clean_for_compare(text: str) -> str:
    return _STRIP_PUNCT_RE.sub("", text).strip()


def clean_output_text(text: str) -> str:
    t = text.strip()
    t = _LEADING_NOISE_RE.sub("", t)
    t = _TRAILING_NOISE_RE.sub("", t)
    if t.endswith("，") or t.endswith(","):
        t = t[:-1].rstrip()
    return t.strip()


def clean_note_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"""^[\s,，.。、;；:：·~～`#]+""", "", t)
    t = re.sub(r"""[\s·~～`]+$""", "", t)
    if t.startswith("“") and "”" not in t:
        t = t + "”"
    elif t.endswith("”") and "“" not in t:
        t = "“" + t
    if t.startswith("（") and "）" not in t:
        t = t + "）"
    elif t.endswith("）") and "（" not in t:
        t = "（" + t
    return t.strip()


def text_similarity(a: str, b: str) -> float:
    ca = clean_for_compare(a)
    cb = clean_for_compare(b)
    if not ca and not cb:
        return 1.0 if (not a.strip() and not b.strip()) else 0.0
    if not ca or not cb:
        return 0.0
    return difflib.SequenceMatcher(None, ca, cb).ratio()


def pick_representative_text(texts: list[str], is_note: bool = False) -> str:
    cleaner = clean_note_text if is_note else clean_output_text
    cleaned_pairs = []
    for t in texts:
        out_t = cleaner(t)
        cmp_t = clean_for_compare(out_t)
        if cmp_t:
            cleaned_pairs.append((cmp_t, out_t))
    if not cleaned_pairs:
        return ""

    key_counts = Counter(cmp_t for cmp_t, _ in cleaned_pairs)
    best_cmp, _ = max(key_counts.items(), key=lambda kv: (kv[1], len(kv[0])))

    candidates = [out_t for cmp_t, out_t in cleaned_pairs if cmp_t == best_cmp]
    out_counts = Counter(candidates)
    best_out, _ = max(out_counts.items(), key=lambda kv: (kv[1], len(kv[0])))
    return best_out


def format_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _read_single_frame_crop(
    cap: cv2.VideoCapture,
    frame_idx: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame[y1:y2, x1:x2]


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ExtractCancelled("用户取消了提取操作")


# ==================== 通用主字幕位置探测与红框定位 ====================

def _probe_main_subtitle_band(
    video_path: Path,
    shared_ocr: Optional[tuple[_RapidOCRV5Adapter, str]] = None,
    accel_mode: str = "auto",
    main_white_only: bool = True,
    ocr_profile: str = "v5_server",
) -> Optional[dict]:
    """
    通用主台词竖直区间探测（按视频真实比例多点采样并做纵向密度聚类，适配任意横竖屏与字幕高度）。
    返回包含 zh_y0, zh_y1, med_h, hits, w, h 的字典；未命中返回 None。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w < 10 or h < 10:
            return None

        ocr_engine, _ = shared_ocr if shared_ocr else create_ocr_engine(accel_mode, ocr_profile=ocr_profile)
        crop_y0 = int(h * 0.45)
        visited_frames: set[int] = set()
        candidates: list[dict] = []

        def _scan_ratios(ratios: tuple[float, ...]) -> None:
            for r in ratios:
                f_idx = max(0, min(total_frames - 1, int(round(total_frames * r))))
                if f_idx in visited_frames:
                    continue
                visited_frames.add(f_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                crop = frame[crop_y0:h, 0:w]
                if crop.size == 0:
                    continue
                res, _ = ocr_engine(crop, text_score=0.58, box_thresh=0.35)
                for box, txt, sc in (res or []):
                    if float(sc) < 0.58:
                        continue
                    clean = txt.strip()
                    if not _VALID_CHAR_RE.search(clean):
                        continue
                    if _matches_watermark_keyword(clean):
                        continue
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) + crop_y0 for p in box]
                    rx0, rx1 = min(xs) / w, max(xs) / w
                    ry0, ry1 = min(ys) / h, max(ys) / h
                    rcx = (rx0 + rx1) / 2.0
                    rcy = (ry0 + ry1) / 2.0
                    rbh = ry1 - ry0
                    if not (0.20 <= rcx <= 0.80):
                        continue
                    if not (0.020 <= rbh <= 0.16):
                        continue
                    if main_white_only and _white_pixel_ratio(crop, box) < 0.04:
                        continue
                    candidates.append({
                        "ry0": ry0,
                        "ry1": ry1,
                        "rcy": rcy,
                        "rbh": rbh,
                    })

        _scan_ratios((0.12, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82))
        if len(candidates) < 2:
            _scan_ratios((0.06, 0.17, 0.28, 0.52, 0.64, 0.76, 0.88, 0.94))
    finally:
        cap.release()

    if not candidates:
        return None

    best_cluster: list[dict] = []
    best_key = (-1, -1.0)
    for c in candidates:
        cluster = [o for o in candidates if abs(o["rcy"] - c["rcy"]) <= 0.045]
        med_cy = float(np.median([o["rcy"] for o in cluster]))
        key = (len(cluster), med_cy)
        if key > best_key:
            best_key = key
            best_cluster = cluster

    return {
        "zh_y0": round(float(np.percentile([b["ry0"] for b in best_cluster], 15)), 4),
        "zh_y1": round(float(np.percentile([b["ry1"] for b in best_cluster], 85)), 4),
        "med_h": round(float(np.median([b["rbh"] for b in best_cluster])), 4),
        "hits": len(best_cluster),
        "w": w,
        "h": h,
    }


def detect_zh_subtitle_band(
    video_path: Path,
    shared_ocr: Optional[tuple[_RapidOCRV5Adapter, str]] = None,
    accel_mode: str = "auto",
    main_white_only: bool = True,
    ocr_profile: str = "v5_server",
) -> Optional[tuple[float, float]]:
    """探测视频原中文字幕真实占据的画面纵向比例范围 (zh_y0, zh_y1)，供烧录阶段全自动防重叠排版使用。"""
    band = _probe_main_subtitle_band(
        video_path,
        shared_ocr=shared_ocr,
        accel_mode=accel_mode,
        main_white_only=main_white_only,
        ocr_profile=ocr_profile,
    )
    if not band:
        return None
    return (band["zh_y0"], band["zh_y1"])


def auto_detect_main_roi(
    video_path: Path,
    shared_ocr: Optional[tuple[_RapidOCRV5Adapter, str]] = None,
    accel_mode: str = "auto",
    main_white_only: bool = True,
    fallback_roi: tuple[float, float, float, float] = (0.06, 0.80, 0.94, 0.95),
    ocr_profile: str = "v5_server",
) -> tuple[tuple[float, float, float, float], bool, int]:
    """
    通用主台词底部红框自动探测（基于 _probe_main_subtitle_band 增加上下安全留白与左右安全全宽）。
    返回: (roi_tuple(x1, y1, x2, y2), 是否命中字幕, 主簇命中框数)
    """
    band = _probe_main_subtitle_band(
        video_path,
        shared_ocr=shared_ocr,
        accel_mode=accel_mode,
        main_white_only=main_white_only,
        ocr_profile=ocr_profile,
    )
    if not band:
        return fallback_roi, False, 0

    pad_y = max(0.022, band["med_h"] * 0.42)
    y1 = round(max(0.42, band["zh_y0"] - pad_y), 3)
    y2 = round(min(0.995, band["zh_y1"] + pad_y), 3)
    if y2 <= y1 + 0.03:
        y2 = round(min(0.995, y1 + 0.08), 3)
    return (0.06, y1, 0.94, y2), True, band["hits"]


def auto_detect_batch_rois(
    video_paths: list[Path],
    accel_mode: str = "auto",
    main_white_only: bool = True,
    fallback_roi: tuple[float, float, float, float] = (0.06, 0.80, 0.94, 0.95),
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    ocr_profile: str = "v5_server",
) -> dict[str, tuple[tuple[float, float, float, float], bool, int]]:
    """批量通用自动探测全部视频的主台词红框初稿（复用同一 PP-OCRv5 引擎，支持随时取消）。"""
    shared_ocr = create_ocr_engine(accel_mode, ocr_profile=ocr_profile)
    total = len(video_paths)
    results: dict[str, tuple[tuple[float, float, float, float], bool, int]] = {}
    for idx, vid in enumerate(video_paths, 1):
        _check_cancel(cancel_event)
        if progress_cb:
            progress_cb(idx, total, f"正在探测 [{vid.name}]...")
        results[vid.name] = auto_detect_main_roi(
            vid,
            shared_ocr=shared_ocr,
            accel_mode=accel_mode,
            main_white_only=main_white_only,
            fallback_roi=fallback_roi,
            ocr_profile=ocr_profile,
        )
    return results


# ==================== 轨道 1：自由选区主台词提取 ====================

def extract_main_track(
    video_path: Path,
    out_srt: Path,
    config: ExtractConfig,
    log_cb: Optional[Callable[[str], None]] = None,
    item_progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    shared_ocr: Optional[tuple[_RapidOCRV5Adapter, str]] = None,
) -> tuple[list[dict], dict]:
    """按用户自定义 main_roi 精确提取底部主台词，精度 ±1 帧，支持 PP-OCRv5 显卡加速与实时进度。"""
    t0 = time.time()

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def iprog(pct: float, msg: str):
        if item_progress_cb:
            item_progress_cb(max(0.0, min(100.0, pct)), msg)

    ocr_engine, actual_backend = shared_ocr if shared_ocr else create_ocr_engine(config.accel_mode, ocr_profile=config.ocr_profile)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rx1, ry1, rx2, ry2 = config.main_roi
    x1 = max(0, min(w - 2, int(w * min(rx1, rx2))))
    x2 = max(x1 + 2, min(w, int(w * max(rx1, rx2))))
    y1 = max(0, min(h - 2, int(h * min(ry1, ry2))))
    y2 = max(y1 + 2, min(h, int(h * max(ry1, ry2))))

    coarse_step = max(1, int(round(fps / COARSE_SAMPLE_FPS)))
    sampled_rois: list[tuple[int, np.ndarray]] = []
    frame_idx = 0
    iprog(1.0, "正在读取视频帧...")
    while True:
        if frame_idx % 60 == 0:
            _check_cancel(cancel_event)
            iprog(1.0 + 11.0 * min(1.0, frame_idx / total_frames), f"读帧采样 {min(100, int(frame_idx * 100 / total_frames))}%")
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % coarse_step == 0:
            sampled_rois.append((frame_idx, frame[y1:y2, x1:x2].copy()))
        frame_idx += 1

    actual_total_frames = max(frame_idx, total_frames)
    cap.release()

    backend_label = "DirectML显卡" if actual_backend == "directml" else "CPU限流"
    log(f"[轨道1-主台词] {video_path.name} ({w}x{h}, {fps:.2f}fps, {backend_label}) 粗扫 {len(sampled_rois)} 帧...")

    coarse_results: list[tuple[int, str]] = []
    all_y0s: list[float] = []
    all_y1s: list[float] = []
    n_samples = max(1, len(sampled_rois))

    if actual_backend == "directml":
        # DirectML 下单线程共享 GPU 会话，快且 CPU 极低不卡顿
        for idx, (f_i, roi_img) in enumerate(sampled_rois, 1):
            _check_cancel(cancel_event)
            txt, y_boxes = read_roi_text_with_boxes(ocr_engine, roi_img, filter_white_only=config.main_white_only)
            coarse_results.append((f_i, txt))
            for by0, by1 in y_boxes:
                all_y0s.append((y1 + by0) / h)
                all_y1s.append((y1 + by1) / h)
            if idx % 10 == 0 or idx == n_samples:
                pct = 12.0 + 66.0 * (idx / n_samples)
                iprog(pct, f"轨道1粗扫 {idx}/{n_samples} 帧 ({backend_label})")
    else:
        def _worker(item: tuple[int, np.ndarray]) -> tuple[int, str, list[tuple[float, float]]]:
            f_i, roi_img = item
            txt, y_boxes = read_roi_text_with_boxes(_get_cpu_worker_ocr(config.ocr_profile), roi_img, filter_white_only=config.main_white_only)
            return f_i, txt, y_boxes

        chunk_size = 20
        with ThreadPoolExecutor(max_workers=2) as pool:
            for idx in range(0, len(sampled_rois), chunk_size):
                _check_cancel(cancel_event)
                batch = sampled_rois[idx : idx + chunk_size]
                for f_i, txt, y_boxes in pool.map(_worker, batch):
                    coarse_results.append((f_i, txt))
                    for by0, by1 in y_boxes:
                        all_y0s.append((y1 + by0) / h)
                        all_y1s.append((y1 + by1) / h)
                done_cnt = min(n_samples, idx + len(batch))
                pct = 12.0 + 66.0 * (done_cnt / n_samples)
                iprog(pct, f"轨道1粗扫 {done_cnt}/{n_samples} 帧 ({backend_label})")

    frame_text_cache: dict[int, str] = {f_i: txt for f_i, txt in coarse_results}

    raw_segments: list[dict] = []
    for f_i, txt in coarse_results:
        if not clean_for_compare(txt):
            continue
        if (
            raw_segments
            and (f_i - raw_segments[-1]["coarse_end"]) <= coarse_step
            and text_similarity(raw_segments[-1]["texts"][-1], txt) >= SIMILARITY_MERGE
        ):
            raw_segments[-1]["coarse_end"] = f_i
            raw_segments[-1]["texts"].append(txt)
        else:
            raw_segments.append({"coarse_start": f_i, "coarse_end": f_i, "texts": [txt]})

    bridged_segments: list[dict] = []
    for seg in raw_segments:
        seg["rep_text"] = pick_representative_text(seg["texts"], is_note=False)
        if not seg["rep_text"]:
            continue
        if bridged_segments:
            prev = bridged_segments[-1]
            gap_sec = (seg["coarse_start"] - prev["coarse_end"]) / fps
            if gap_sec <= GAP_BRIDGE_SEC and text_similarity(prev["rep_text"], seg["rep_text"]) >= SIMILARITY_MERGE:
                prev["coarse_end"] = seg["coarse_end"]
                prev["texts"].extend(seg["texts"])
                prev["rep_text"] = pick_representative_text(prev["texts"], is_note=False)
                continue
        bridged_segments.append(seg)

    cap = cv2.VideoCapture(str(video_path))
    boundary_calls = 0

    def get_frame_text(f_i: int) -> str:
        nonlocal boundary_calls
        if f_i in frame_text_cache:
            return frame_text_cache[f_i]
        roi = _read_single_frame_crop(cap, f_i, x1, y1, x2, y2)
        if roi is None:
            frame_text_cache[f_i] = ""
            return ""
        txt = read_roi_text(ocr_engine, roi, filter_white_only=config.main_white_only)
        boundary_calls += 1
        frame_text_cache[f_i] = txt
        return txt

    def matches_text(f_i: int, target: str) -> bool:
        return text_similarity(get_frame_text(f_i), target) >= SIMILARITY_BOUNDARY

    n_segs = max(1, len(bridged_segments))
    try:
        for s_idx, seg in enumerate(bridged_segments, 1):
            _check_cancel(cancel_event)
            rep = seg["rep_text"]

            lo = max(0, seg["coarse_start"] - coarse_step)
            hi = seg["coarse_start"]
            exact_start = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                if matches_text(mid, rep):
                    exact_start = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            seg["start_frame"] = exact_start

            lo = seg["coarse_end"]
            hi = min(actual_total_frames - 1, seg["coarse_end"] + coarse_step)
            last_match = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                if matches_text(mid, rep):
                    last_match = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            seg["end_frame"] = min(actual_total_frames, last_match + 1)

            if s_idx % 3 == 0 or s_idx == n_segs:
                iprog(78.0 + 18.0 * (s_idx / n_segs), f"轨道1边界精定位 {s_idx}/{n_segs} 条")

        for i in range(len(bridged_segments) - 1):
            cur_seg = bridged_segments[i]
            nxt_seg = bridged_segments[i + 1]
            if cur_seg["end_frame"] >= nxt_seg["start_frame"] or (nxt_seg["coarse_start"] - cur_seg["coarse_end"]) <= coarse_step:
                search_lo = max(cur_seg["start_frame"] + 1, min(cur_seg["coarse_end"], nxt_seg["start_frame"]))
                search_hi = min(nxt_seg["end_frame"] - 1, max(cur_seg["end_frame"], nxt_seg["coarse_start"]))
                if search_lo <= search_hi:
                    lo, hi = search_lo, search_hi
                    cut_frame = None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if matches_text(mid, nxt_seg["rep_text"]):
                            cut_frame = mid
                            hi = mid - 1
                        else:
                            lo = mid + 1
                    if cut_frame is not None and (cur_seg["end_frame"] >= nxt_seg["start_frame"] or (cut_frame - cur_seg["end_frame"]) <= 2):
                        cur_seg["end_frame"] = cut_frame
                        nxt_seg["start_frame"] = cut_frame

            gap_frames = nxt_seg["start_frame"] - cur_seg["end_frame"]
            if 0 < gap_frames <= 2 or cur_seg["end_frame"] > nxt_seg["start_frame"]:
                cur_seg["end_frame"] = nxt_seg["start_frame"]
    finally:
        cap.release()

    valid_segments = []
    for seg in bridged_segments:
        dur = (seg["end_frame"] - seg["start_frame"]) / fps
        if dur >= MIN_MAIN_DURATION and seg["end_frame"] > seg["start_frame"]:
            valid_segments.append({
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "start_sec": seg["start_frame"] / fps,
                "end_sec": seg["end_frame"] / fps,
                "text": seg["rep_text"],
            })

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    with out_srt.open("w", encoding="utf-8") as f:
        for idx, seg in enumerate(valid_segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_time(seg['start_sec'])} --> {format_srt_time(seg['end_sec'])}\n")
            f.write(f"{seg['text']}\n\n")

    zh_band: Optional[tuple[float, float]] = None
    if all_y0s and all_y1s:
        zh_y0 = round(float(np.percentile(all_y0s, 15)), 4)
        zh_y1 = round(float(np.percentile(all_y1s, 85)), 4)
        if 0.35 <= zh_y0 < zh_y1 <= 1.0:
            zh_band = (zh_y0, zh_y1)
            layout_path = out_srt.with_name(f"{out_srt.stem}_main.layout.json")
            layout_payload = {
                "video_name": video_path.name,
                "video_w": w,
                "video_h": h,
                "zh_y0": zh_y0,
                "zh_y1": zh_y1,
                "samples": len(all_y0s),
                "source": "extract_main_track",
            }
            layout_path.write_text(json.dumps(layout_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    iprog(100.0, f"轨道1完成 ({len(valid_segments)}条)")
    band_info = f", 原中文位置={zh_band[0]*100:.1f}%~{zh_band[1]*100:.1f}%" if zh_band else ""
    log(f"[轨道1完成] 提取 {len(valid_segments)} 条主台词 (耗时 {elapsed:.2f}s, {backend_label}, 精定位 {boundary_calls} 帧{band_info}) -> {out_srt.name}")
    return valid_segments, {
        "count": len(valid_segments),
        "elapsed": elapsed,
        "backend": actual_backend,
        "w": w, "h": h, "fps": fps,
        "zh_band": zh_band,
    }


# ==================== 轨道 2：自由蓝框全屏花字 & 背景大字提取 ====================

def _is_main_sub_active(t_sec: float, main_segments: list[dict], pad: float = 0.12) -> bool:
    for seg in main_segments:
        if (seg["start_sec"] - pad) <= t_sec <= (seg["end_sec"] + pad):
            return True
    return False


def _is_subseq_or_fade(cmp_a: str, cmp_b: str) -> bool:
    if not cmp_a or not cmp_b:
        return False
    shorter, longer = (cmp_a, cmp_b) if len(cmp_a) <= len(cmp_b) else (cmp_b, cmp_a)
    return len(shorter) >= 3 and shorter in longer


def _merge_boxes_into_note_tracks(
    frame_hits: list[tuple[int, list[dict]]],
    fps: float,
    w: int,
    h: int,
) -> list[dict]:
    diag = math.hypot(w, h)
    active_tracks: list[dict] = []
    finished_tracks: list[dict] = []

    for f_i, boxes in frame_hits:
        t_sec = f_i / fps
        still_active = []
        for tr in active_tracks:
            if (t_sec - tr["last_t"]) > NOTE_GAP_BRIDGE_SEC:
                finished_tracks.append(tr)
            else:
                still_active.append(tr)
        active_tracks = still_active

        used_tracks = set()
        for b in boxes:
            best_idx = -1
            best_score = -1.0
            cmp_b = clean_for_compare(b["text"])

            for idx, tr in enumerate(active_tracks):
                if idx in used_tracks:
                    continue
                last_obs = tr["obs"][-1]
                cmp_last = clean_for_compare(last_obs["text"])

                dist = math.hypot(b["cx"] - last_obs["cx"], b["cy"] - last_obs["cy"])
                sim = text_similarity(b["text"], last_obs["text"])
                is_same_row = abs(b["cy"] - last_obs["cy"]) <= (0.12 * h)
                horiz_overlap = min(b["x1"], last_obs["x1"]) > max(b["x0"], last_obs["x0"]) - (0.05 * w)

                match_normal = (sim >= NOTE_SIMILARITY_CLUSTER) and (dist <= 0.16 * diag)
                match_progressive = is_same_row and horiz_overlap and _is_subseq_or_fade(cmp_last, cmp_b)

                if match_normal or match_progressive:
                    score = sim * 2.0 - (dist / diag)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx >= 0:
                tr = active_tracks[best_idx]
                tr["obs"].append(b)
                tr["last_f"] = f_i
                tr["last_t"] = t_sec
                used_tracks.add(best_idx)
            else:
                active_tracks.append({
                    "first_f": f_i,
                    "last_f": f_i,
                    "last_t": t_sec,
                    "obs": [b],
                })
                used_tracks.add(len(active_tracks) - 1)

    finished_tracks.extend(active_tracks)

    tracks_summary = []
    for tr in finished_tracks:
        obs = tr["obs"]
        cmp_list = [clean_for_compare(o["text"]) for o in obs if clean_for_compare(o["text"])]
        if not cmp_list:
            continue

        max_len = max(len(c) for c in cmp_list)
        min_len = min(len(c) for c in cmp_list)
        progressive_mode = (max_len >= min_len + 3) and any(
            (c in max(cmp_list, key=len)) for c in cmp_list
        )

        if progressive_mode:
            long_texts = [o["text"] for o in obs if len(clean_for_compare(o["text"])) >= max_len - 1]
            rep_text = pick_representative_text(long_texts, is_note=True)
        else:
            rep_text = pick_representative_text([o["text"] for o in obs], is_note=True)

        if not rep_text:
            continue

        coord_obs = [o for o in obs if text_similarity(o["text"], rep_text) >= 0.70] or obs
        cx = float(np.median([o["cx"] for o in coord_obs]))
        cy = float(np.median([o["cy"] for o in coord_obs]))
        x0 = float(np.median([o["x0"] for o in coord_obs]))
        y0 = float(np.median([o["y0"] for o in coord_obs]))
        x1 = float(np.median([o["x1"] for o in coord_obs]))
        y1 = float(np.median([o["y1"] for o in coord_obs]))
        bw = float(np.median([o["bw"] for o in coord_obs]))
        bh = float(np.median([o["bh"] for o in coord_obs]))
        is_white = float(np.median([o["wr"] for o in coord_obs])) >= 0.04

        union_x0 = max(0, int(min(o["x0"] for o in obs) - 140))
        union_y0 = max(0, int(min(o["y0"] for o in obs) - 90))
        union_x1 = min(w, int(max(o["x1"] for o in obs) + 140))
        union_y1 = min(h, int(max(o["y1"] for o in obs) + 90))

        tracks_summary.append({
            "coarse_start": tr["first_f"],
            "coarse_end": tr["last_f"],
            "rep_text": rep_text,
            "all_cmps": list(set(cmp_list)),
            "cx": round(cx, 1), "cy": round(cy, 1),
            "x0": round(x0, 1), "y0": round(y0, 1),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "bw": round(bw, 1), "bh": round(bh, 1),
            "is_white": is_white,
            "search_box": (union_x0, union_y0, union_x1, union_y1),
        })

    return tracks_summary


def _filter_watermark_tracks(
    tracks: list[dict],
    fps: float,
    total_frames: int,
    w: int,
    h: int,
    log_cb: Callable[[str], None],
) -> list[dict]:
    """过滤固定角落常驻水印与平台声明（如 AI视频 / 仅供娱乐），保留真实剧情注释花字。"""
    video_dur = total_frames / max(1.0, fps)
    diag = math.hypot(w, h)

    def _in_corner(tr: dict) -> bool:
        cx, cy = float(tr["cx"]), float(tr["cy"])
        return (cy < 0.25 * h) or (cy > 0.80 * h) or (cx < 0.25 * w) or (cx > 0.75 * w)

    drop_reasons: dict[int, str] = {}

    # 规则 A（角标关键词）与规则 B（单段超长常驻）
    for i, tr in enumerate(tracks):
        if _in_corner(tr) and _matches_watermark_keyword(tr["rep_text"]):
            drop_reasons[i] = "角标关键词"
            continue
        dur = (tr["coarse_end"] - tr["coarse_start"]) / max(1.0, fps)
        if video_dur >= 15.0 and dur >= max(10.0, video_dur * 0.60):
            drop_reasons[i] = f"单段常驻{dur:.1f}s"

    # 规则 C（跨黑屏/转场同位置同字反复出现的累计常驻水印）
    if video_dur >= 15.0:
        corner_indices = [i for i, tr in enumerate(tracks) if i not in drop_reasons and _in_corner(tr)]
        visited: set[int] = set()
        for idx_a in corner_indices:
            if idx_a in visited:
                continue
            tr_a = tracks[idx_a]
            group = [idx_a]
            visited.add(idx_a)
            for idx_b in corner_indices:
                if idx_b in visited:
                    continue
                tr_b = tracks[idx_b]
                dist = math.hypot(tr_a["cx"] - tr_b["cx"], tr_a["cy"] - tr_b["cy"])
                if dist <= 0.10 * diag and text_similarity(tr_a["rep_text"], tr_b["rep_text"]) >= 0.72:
                    group.append(idx_b)
                    visited.add(idx_b)
            tot_dur = sum((tracks[k]["coarse_end"] - tracks[k]["coarse_start"]) / max(1.0, fps) for k in group)
            if tot_dur >= max(10.0, video_dur * 0.55):
                for k in group:
                    drop_reasons[k] = f"累计常驻{tot_dur:.1f}s"

    kept: list[dict] = []
    for i, tr in enumerate(tracks):
        if i in drop_reasons:
            log_cb(f"[轨道2过滤水印] 已跳过固定水印 ({drop_reasons[i]}): \"{tr['rep_text']}\"")
        else:
            kept.append(tr)
    return kept


def extract_notes_track(
    video_path: Path,
    main_segments: list[dict],
    out_notes_srt: Path,
    out_pos_json: Path,
    config: ExtractConfig,
    log_cb: Optional[Callable[[str], None]] = None,
    item_progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    shared_ocr: Optional[tuple[_RapidOCRV5Adapter, str]] = None,
) -> tuple[list[dict], dict]:
    """在用户选定的 note_roi 蓝框内提取画面花字/背景大字，支持 PP-OCRv5 显卡加速与实时进度。"""
    t0 = time.time()

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    def iprog(pct: float, msg: str):
        if item_progress_cb:
            item_progress_cb(max(0.0, min(100.0, pct)), msg)

    ocr_engine, actual_backend = shared_ocr if shared_ocr else create_ocr_engine(config.accel_mode, ocr_profile=config.ocr_profile)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mx1, my1, mx2, my2 = config.main_roi
    main_x1_p, main_x2_p = w * min(mx1, mx2), w * max(mx1, mx2)
    main_y1_p, main_y2_p = h * min(my1, my2), h * max(my1, my2)

    nx1, ny1, nx2, ny2 = config.note_roi
    crop_x0 = max(0, min(w - 2, int(w * min(nx1, nx2))))
    crop_x1 = max(crop_x0 + 2, min(w, int(w * max(nx1, nx2))))
    crop_y0 = max(0, min(h - 2, int(h * min(ny1, ny2))))
    crop_y1 = max(crop_y0 + 2, min(h, int(h * max(ny1, ny2))))

    note_step = max(1, int(round(fps / NOTE_SAMPLE_FPS)))
    approx_samples = max(1, total_frames // note_step)
    min_box_h = h * config.note_min_height_ratio

    def _parse_ocr_boxes(f_i: int, note_img: np.ndarray, res) -> tuple[int, list[dict]]:
        kept = []
        t_sec = f_i / fps
        main_active = _is_main_sub_active(t_sec, main_segments, pad=0.12)

        for box, txt, sc in (res or []):
            if float(sc) < NOTE_MIN_CONFIDENCE:
                continue
            raw_t = txt.strip()
            if not _VALID_CHAR_RE.search(raw_t):
                continue

            xs = [float(p[0]) + crop_x0 for p in box]
            ys = [float(p[1]) + crop_y0 for p in box]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            bw = x1 - x0
            bh = y1 - y0

            if bh < min_box_h:
                continue

            wr = _white_pixel_ratio(note_img, box)
            in_main_box = (main_y1_p <= cy <= main_y2_p) and (main_x1_p <= cx <= main_x2_p)
            if in_main_box and (not config.main_white_only or wr >= 0.04):
                continue

            if main_active and (cy > max(main_y2_p - 0.04 * h, 0.88 * h)) and (0.25 * w <= cx <= 0.75 * w):
                continue

            kept.append({
                "text": raw_t,
                "score": float(sc),
                "cx": cx, "cy": cy,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "bw": bw, "bh": bh,
                "wr": wr,
            })
        return f_i, kept

    all_hits: list[tuple[int, list[dict]]] = []
    sampled_count = 0
    frame_idx = 0
    iprog(0.0, "准备全屏花字扫描...")

    if actual_backend == "directml":
        while True:
            if frame_idx % 30 == 0:
                _check_cancel(cancel_event)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % note_step == 0:
                note_img = frame[crop_y0:crop_y1, crop_x0:crop_x1]
                res, _ = ocr_engine(note_img, text_score=NOTE_MIN_CONFIDENCE, box_thresh=0.35)
                all_hits.append(_parse_ocr_boxes(frame_idx, note_img, res))
                sampled_count += 1
                if sampled_count % 8 == 0 or sampled_count == approx_samples:
                    pct = 80.0 * min(1.0, sampled_count / approx_samples)
                    iprog(pct, f"轨道2花字扫描 {sampled_count}/{approx_samples} 帧 (DirectML显卡)")
            frame_idx += 1
    else:
        def _detect_cpu(item: tuple[int, np.ndarray]) -> tuple[int, list[dict]]:
            f_i, note_img = item
            ocr = _get_cpu_worker_ocr(config.ocr_profile)
            res, _ = ocr(note_img, text_score=NOTE_MIN_CONFIDENCE, box_thresh=0.35)
            return _parse_ocr_boxes(f_i, note_img, res)

        chunk_items: list[tuple[int, np.ndarray]] = []
        chunk_size = 24
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                if frame_idx % 30 == 0:
                    _check_cancel(cancel_event)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % note_step == 0:
                    chunk_items.append((frame_idx, frame[crop_y0:crop_y1, crop_x0:crop_x1].copy()))
                    sampled_count += 1
                    if len(chunk_items) >= chunk_size:
                        _check_cancel(cancel_event)
                        all_hits.extend(list(pool.map(_detect_cpu, chunk_items)))
                        chunk_items.clear()
                        iprog(80.0 * min(1.0, sampled_count / approx_samples), f"轨道2花字扫描 {sampled_count}/{approx_samples} 帧 (CPU限流)")
                frame_idx += 1
            if chunk_items:
                _check_cancel(cancel_event)
                all_hits.extend(list(pool.map(_detect_cpu, chunk_items)))
                chunk_items.clear()

    actual_total_frames = max(frame_idx, total_frames)
    cap.release()

    log(f"[轨道2-花字/背景大字] 粗扫 {sampled_count} 帧完毕，正在聚类与精定位...")
    tracks = _merge_boxes_into_note_tracks(all_hits, fps, w, h)
    if config.filter_watermark and tracks:
        tracks = _filter_watermark_tracks(tracks, fps, actual_total_frames, w, h, log)

    cap = cv2.VideoCapture(str(video_path))
    boundary_calls = 0
    valid_notes: list[dict] = []
    n_tracks = max(1, len(tracks))

    try:
        for t_idx, tr in enumerate(tracks, 1):
            _check_cancel(cancel_event)
            bx0, by0, bx1, by1 = tr["search_box"]
            local_cache: dict[int, list[str]] = {}

            def _check_frame_in_local_box(f_i: int) -> bool:
                nonlocal boundary_calls
                if f_i not in local_cache:
                    crop = _read_single_frame_crop(cap, f_i, bx0, by0, bx1, by1)
                    if crop is None or crop.size == 0:
                        local_cache[f_i] = []
                    else:
                        res, _ = ocr_engine(crop, text_score=0.55, box_thresh=0.35)
                        boundary_calls += 1
                        txts = [t.strip() for _, t, s in (res or []) if float(s) >= 0.55 and t.strip()]
                        if txts:
                            txts.append("".join(txts))
                        local_cache[f_i] = txts

                for cand in local_cache[f_i]:
                    cmp_cand = clean_for_compare(cand)
                    if not cmp_cand:
                        continue
                    for known_cmp in tr["all_cmps"]:
                        if difflib.SequenceMatcher(None, cmp_cand, known_cmp).ratio() >= 0.62 or _is_subseq_or_fade(cmp_cand, known_cmp):
                            return True
                return False

            lo = max(0, tr["coarse_start"] - note_step)
            hi = tr["coarse_start"]
            exact_start = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                if _check_frame_in_local_box(mid):
                    exact_start = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            lo = tr["coarse_end"]
            hi = min(actual_total_frames - 1, tr["coarse_end"] + note_step)
            last_match = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                if _check_frame_in_local_box(mid):
                    last_match = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            end_frame = min(actual_total_frames, last_match + 1)
            dur = (end_frame - exact_start) / fps
            if dur >= NOTE_MIN_DURATION and end_frame > exact_start:
                valid_notes.append({
                    "start_frame": exact_start,
                    "end_frame": end_frame,
                    "start_sec": round(exact_start / fps, 3),
                    "end_sec": round(end_frame / fps, 3),
                    "text_zh": tr["rep_text"],
                    "cx": tr["cx"], "cy": tr["cy"],
                    "x0": tr["x0"], "y0": tr["y0"], "x1": tr["x1"], "y1": tr["y1"],
                    "bw": tr["bw"], "bh": tr["bh"],
                    "is_white": tr["is_white"],
                })
            iprog(80.0 + 18.0 * (t_idx / n_tracks), f"轨道2局部精定位 {t_idx}/{n_tracks} 条")
    finally:
        cap.release()

    valid_notes.sort(key=lambda x: (x["start_frame"], x["y0"]))

    if valid_notes:
        out_notes_srt.parent.mkdir(parents=True, exist_ok=True)
        with out_notes_srt.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(valid_notes, 1):
                item["index"] = idx
                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(item['start_sec'])} --> {format_srt_time(item['end_sec'])}\n")
                f.write(f"{item['text_zh']}\n\n")

        pos_payload = {
            "video_name": video_path.name,
            "video_w": w,
            "video_h": h,
            "fps": fps,
            "items": [
                {
                    "index": item["index"],
                    "start_sec": item["start_sec"],
                    "end_sec": item["end_sec"],
                    "text_zh": item["text_zh"],
                    "cx": item["cx"], "cy": item["cy"],
                    "x0": item["x0"], "y0": item["y0"], "x1": item["x1"], "y1": item["y1"],
                    "bw": item["bw"], "bh": item["bh"],
                    "is_white": item["is_white"],
                }
                for item in valid_notes
            ],
        }
        out_pos_json.write_text(json.dumps(pos_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    iprog(100.0, f"轨道2完成 ({len(valid_notes)}条)")
    log(f"[轨道2完成] 提取 {len(valid_notes)} 条花字/背景大字 (耗时 {elapsed:.2f}s, 精定位 {boundary_calls} 帧)")
    return valid_notes, {"count": len(valid_notes), "elapsed": elapsed, "boundary_calls": boundary_calls}


# ==================== Agent 提示词生成与批量调度 ====================

def generate_agent_prompt(
    zh_dir: Path,
    en_dir: Path,
    extracted_files: list[str],
    glossary_text: str,
    out_prompt_path: Path,
) -> str:
    """生成带多地方言自识别、整句通顺度校对铁律与代表词典的 Agent 翻译提示词。"""
    file_list_str = "\n".join(f"  - `{name}`" for name in extracted_files) if extracted_files else "  - （当前目录中的全部 .srt 文件）"

    prompt_text = f"""请你作为精通中国各地方言俚语（涵盖80/90年代北方农村、东北、川渝、粤语、云南、湖南湖北等民俗口语及网络二创热梗）且母语为地道英语口语的影视字幕译者，帮我批量校对 OCR 错别字并翻译以下双语字幕文件。

==================== 一、文件夹路径与强制命名规范 ====================
1. 待翻译中文 SRT 文件夹：`{zh_dir.resolve()}`
   （注：同目录下的 `*.json` 文件（`*.pos.json` / `*_main.layout.json` / `_video_*.json`）是程序自动保存的画面坐标与排版账本，**绝对不要修改、删除或翻译任何 JSON 文件**！）
2. 翻译后英文 SRT 输出文件夹：`{en_dir.resolve()}`
3. 本次待翻译文件清单：
{file_list_str}
4. **文件名铁律（非常重要！必须严格执行）**：
   - 输出到英文文件夹时，**必须与原中文 SRT 文件名 100% 完全一致**（包括 `_notes` 后缀，不得擅自改名或加 `_en`）：
     - 例：`测试1.srt`        → 保存为 `{en_dir.resolve()}\\测试1.srt`
     - 例：`测试1_notes.srt`  → 保存为 `{en_dir.resolve()}\\测试1_notes.srt`
   - 所有输出文件均采用 **UTF-8** 编码。

==================== 二、先判别方言年代 + 整句通顺度校对铁律（核心要求！） ====================
1. **全局方言与年代自适应判别**：
   - 在翻译每个 `.srt` 文件前，请先通读全片字幕上下文，**自行判断这集视频属于哪个地域方言和年代背景**（例如：90年代北方农村怀旧、东北喜剧、川渝市井、港风粤语、滇味/云贵、湘鄂方言或现代二创），据此统一锁定与之匹配的英文口语语调（质朴接地气、诙谐泼辣或港片地道俚语）。
2. **OCR 错字纠正铁律（严禁死板逐字替换！）**：
   - OCR 识别容易出现形近字、同音方言字或漏字错别字（例如把“咦这豆子还甜嘞”认成“睫这豆子还甜”，把“自己人”认成“自已人”，或把川渝/东北/粤语同音土话拆错字）。
   - **纠正步骤**：你必须结合**前后 2~3 句完整剧情 + 当地方言读音（同音/近音）+ 汉字笔画形近特征**进行还原；**改完之后必须默读一遍，检查还原后的中文是不是一句完整、顺畅、符合当地人说话逻辑的自然口语！**
   - 切忌生硬死板地按字面硬抠，若某字明显不通顺（哪怕看起来像常用字），也要大胆按语境与方言发音还原为真正通顺的意思后再做英文意译。
3. **严禁改动格式与时间轴**：
   - 严禁增删字幕序号（1, 2, 3...），严禁修改任何时间轴（`00:00:00,100 --> 00:00:00,900`）！中文有几条，英文就必须严丝合缝对应几条。
   - 只输出纯英文译文（原视频已有中文），不要夹杂中文，不要输出解释废话。

==================== 三、双轨差异化翻译风格 ====================
1. **普通对白文件（`*.srt`）**：
   - 风格：贴合该片年代与地域气息的自然美式口语，重神韵和情绪张力，**绝不机翻直译方言词面**。
   - 长度：单条尽量控制在 1 行以内（短小精悍），方便短视频观众扫一眼看懂。
2. **画面注释/背景大字文件（`*_notes.srt`）**：
   - 物体/功能注释（如 `“向日葵照明”`、`“冰鲜蔬菜”`）：用 **2~4 个英文单词的极简屏幕标签风**，保留引号感；
   - 转场提示（如 `（不久之前）`）：用常见影视写法如 `(A Little While Ago)`；
   - 片尾升华金句：翻出短促有力、温暖大气的电影海报金句感。

==================== 四、全国高频方言意译标杆词典（请举一反三、重神韵不直译） ====================
{glossary_text.strip()}
"""
    out_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    out_prompt_path.write_text(prompt_text, encoding="utf-8")
    return prompt_text


def extract_batch(
    video_paths: list[Path],
    zh_dir: Path,
    en_dir: Path,
    config: ExtractConfig,
    glossary_text: str,
    prompt_path: Path,
    progress_cb: Optional[Callable[[int, int, float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    per_video_configs: Optional[dict[str, ExtractConfig]] = None,
    auto_detect_unset: bool = False,
    on_auto_roi_cb: Optional[Callable[[str, tuple[float, float, float, float]], None]] = None,
) -> dict:
    """批量执行多视频的双轨中文 SRT 提取，支持 PP-OCRv5 切换、逐集独立选区、日志自动落盘与实时双进度。"""
    zh_dir.mkdir(parents=True, exist_ok=True)
    en_dir.mkdir(parents=True, exist_ok=True)
    log_file = zh_dir / "_extract_log.txt"

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

    total = len(video_paths)
    processed = 0
    skipped = 0
    cancelled = False
    extracted_names: list[str] = []

    log(f"=== 开始批量提取 (共 {total} 个视频 | 模型={config.ocr_profile} | 加速={config.accel_mode}) ===")

    # 预先初始化一次 OCR 引擎（DirectML 下单视频之间复用，极大加快速度）
    shared_ocr = create_ocr_engine(config.accel_mode, ocr_profile=config.ocr_profile)

    for idx, vid in enumerate(video_paths, 1):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        has_custom = bool(per_video_configs and vid.name in per_video_configs)
        vid_cfg = per_video_configs[vid.name] if has_custom else config

        out_main = zh_dir / f"{vid.stem}.srt"
        out_notes = zh_dir / f"{vid.stem}_notes.srt"
        out_pos = zh_dir / f"{vid.stem}_notes.pos.json"

        if vid_cfg.skip_existing and out_main.exists() and (not vid_cfg.extract_notes or out_notes.exists() or out_pos.exists()):
            log(f"[跳过] 已存在提取文件: {out_main.name}")
            extracted_names.append(out_main.name)
            if vid_cfg.extract_notes and out_notes.exists():
                extracted_names.append(out_notes.name)
            skipped += 1
            if progress_cb:
                progress_cb(idx, total, 100.0, f"[{vid.name}] 已跳过")
            continue

        if (not has_custom) and auto_detect_unset:
            if progress_cb:
                progress_cb(idx, total, 0.0, f"[{vid.name}] 正在自动探测红框位置...")
            det_roi, found, hits = auto_detect_main_roi(
                vid,
                shared_ocr=shared_ocr,
                accel_mode=vid_cfg.accel_mode,
                main_white_only=vid_cfg.main_white_only,
                fallback_roi=vid_cfg.main_roi,
                ocr_profile=vid_cfg.ocr_profile,
            )
            if found:
                vid_cfg = replace(vid_cfg, main_roi=det_roi)
                tag = f"自动红框(命中{hits})"
                if on_auto_roi_cb:
                    on_auto_roi_cb(vid.name, det_roi)
            else:
                tag = "默认配置(未探出)"
        else:
            tag = "独立配置" if has_custom else "默认配置"

        mx1, my1, mx2, my2 = vid_cfg.main_roi
        log(f"[视频 {idx}/{total}] {vid.name} | {tag} 红框 Y={my1*100:.1f}%~{my2*100:.1f}%, X={mx1*100:.1f}%~{mx2*100:.1f}%")

        if progress_cb:
            progress_cb(idx, total, 0.0, f"[{vid.name}] 准备提取...")

        try:
            def _track1_prog(pct: float, stage: str):
                mapped = (pct * 0.55) if vid_cfg.extract_notes else pct
                if progress_cb:
                    progress_cb(idx, total, mapped, f"[{vid.name}] {stage}")

            main_segs, _ = extract_main_track(
                vid, out_main, vid_cfg,
                log_cb=log,
                item_progress_cb=_track1_prog,
                cancel_event=cancel_event,
                shared_ocr=shared_ocr,
            )
            extracted_names.append(out_main.name)

            if vid_cfg.extract_notes:
                def _track2_prog(pct: float, stage: str):
                    mapped = 55.0 + pct * 0.45
                    if progress_cb:
                        progress_cb(idx, total, mapped, f"[{vid.name}] {stage}")

                notes_list, _ = extract_notes_track(
                    vid, main_segs, out_notes, out_pos, vid_cfg,
                    log_cb=log,
                    item_progress_cb=_track2_prog,
                    cancel_event=cancel_event,
                    shared_ocr=shared_ocr,
                )
                if notes_list:
                    extracted_names.append(out_notes.name)

            if progress_cb:
                progress_cb(idx, total, 100.0, f"[{vid.name}] 提取完成")
            processed += 1
        except ExtractCancelled:
            cancelled = True
            log(f"[已取消] 提取流程在 {vid.name} 处中止")
            break
        except Exception as exc:
            log(f"[错误] 处理 {vid.name} 失败: {exc}")

    if not cancelled and progress_cb:
        progress_cb(total, total, 100.0, "提取全部完成")

    all_zh_srts = sorted([p.name for p in zh_dir.glob("*.srt")]) if zh_dir.exists() else extracted_names
    prompt_text = generate_agent_prompt(zh_dir, en_dir, all_zh_srts, glossary_text, prompt_path)
    log(f"[完成] 提示词已更新保存 -> {prompt_path}")

    return {
        "processed": processed,
        "skipped": skipped,
        "cancelled": cancelled,
        "prompt_text": prompt_text,
        "prompt_path": prompt_path,
    }
