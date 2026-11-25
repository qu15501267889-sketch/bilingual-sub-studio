import gc
import json
import os
import threading
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Optional

import cv2
from PIL import Image

from core.burner import (
    AUTO_PLACE_CHOICES,
    COLOR_PRESETS,
    DEDUP_LABELS,
    DEFAULT_DEDUP_OPTIONS,
    ENCODER_CHOICES,
    BurnStyleConfig,
    burn_batch,
    compute_effective_main_layout,
    detect_best_encoder,
    load_or_detect_zh_band,
    parse_srt,
    render_burn_preview_with_meta,
    scan_burn_matches,
)
from core.extractor import (
    OCR_PROFILE_CHOICES,
    VIDEO_EXTS,
    ExtractCancelled,
    ExtractConfig,
    auto_detect_batch_rois,
    detect_ocr_accelerator,
    extract_batch,
    generate_agent_prompt,
)
from core.runtime_paths import (
    get_app_dir,
    get_ffmpeg_exe,
    load_glossary,
    load_presets,
    load_translator_config,
    save_glossary,
    save_presets,
    save_translator_config,
)
from core.translator import (
    SrtBlock,
    TranslateCancelled,
    TranslatorConfig,
    list_ollama_models,
    parse_srt_blocks,
    test_translator_connection,
    translate_batch,
    unload_ollama_model,
    write_srt_blocks,
)
from gui.preview_canvas import PreviewCanvas


class ScrollableFrame(ttk.Frame):
    """带垂直滚动条的通用容器，保障高缩放或小屏幕下全部控件可完整滚动访问。"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win_id, width=e.width))

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.inner.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def _fmt_sec(sec: float) -> str:
    sec_i = max(0, int(round(sec)))
    m, s = divmod(sec_i, 60)
    return f"{m:02d}:{s:02d}"


class BilingualSubStudioApp(tk.Tk):
    """双语字幕工坊 V1.5（一条龙自动翻译 · PP-OCRv5 · 自由(X,Y)拖拽排版）。"""

    def __init__(self):
        super().__init__()
        self.title("双语字幕工坊 V1.5（一条龙自动翻译 · PP-OCRv5 · 自由(X,Y)拖拽排版）")
        sw = max(1220, self.winfo_screenwidth())
        sh = max(820, self.winfo_screenheight())
        self.geometry(f"{min(1440, sw - 50)}x{min(940, sh - 70)}")
        self.minsize(1180, 760)

        self.style = ttk.Style(self)
        if "vista" in self.style.theme_names():
            self.style.theme_use("vista")
        self.option_add("*Font", ("Microsoft YaHei UI", 9))

        # 共用路径变量（三页智能联动）
        self.var_video_dir = tk.StringVar()
        self.var_zh_dir = tk.StringVar()
        self.var_en_dir = tk.StringVar()
        self.var_out_dir = tk.StringVar()
        self._last_auto_vdir: Optional[Path] = None

        # 运行任务状态
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._is_busy = False

        # 提取页状态
        self._ext_videos: list[Path] = []
        self._ext_default_cfg: dict = self._make_default_ext_cfg()
        self._ext_video_configs: dict[str, dict] = {}
        self._ext_current_video_name: Optional[str] = None
        self._ext_loaded_video_path: Optional[Path] = None
        self._ext_loading_ui: bool = False
        self._ext_cap: Optional[cv2.VideoCapture] = None
        self._ext_fps: float = 30.0
        self._ext_total_frames: int = 0
        self._ext_seek_after_id: Optional[str] = None

        # 翻译页校对状态
        self._trans_review_files: list[str] = []
        self._trans_cur_zh_blocks: list[SrtBlock] = []
        self._trans_cur_en_blocks: list[SrtBlock] = []

        # 烧录页预览、匹配与逐集独立样式状态
        self._burn_matches: list[dict] = []
        self._burn_selected_match: Optional[dict] = None
        self._burn_default_style: dict = asdict(BurnStyleConfig())
        self._burn_video_styles: dict[str, dict] = {}
        self._cur_note_pos_overrides: dict[str, list[float]] = {}
        self._burn_loading_ui: bool = False
        self._burn_frame_bgr = None
        self._burn_fps: float = 30.0
        self._burn_total_frames: int = 0
        self._burn_seek_after_id: Optional[str] = None
        self._burn_cur_en_srt: list[dict] = []
        self._burn_cur_notes_srt: list[dict] = []
        self._burn_cur_pos_items: list[dict] = []
        self._burn_zh_bands: dict[str, Optional[tuple[float, float]]] = {}
        self._burn_probing_videos: set[str] = set()
        self._burn_last_note_index: Optional[int] = None

        self._presets = load_presets()
        self._glossary_text = load_glossary()
        self._trans_cfg_dict = load_translator_config()
        self._last_prompt_text = ""

        self._build_header()
        self._build_pipeline_bar()
        self._build_notebook()
        self._probe_hardware_async()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _make_default_ext_cfg() -> dict:
        return {
            "main_roi": [0.08, 0.81, 0.92, 0.92],
            "main_white_only": True,
            "extract_notes": True,
            "note_roi": [0.00, 0.00, 1.00, 1.00],
            "note_min_height_ratio": 0.038,
            "filter_watermark": True,
            "source": "default",
        }

    # ==================== 顶部状态条与全自动一条龙快速入口 ====================

    def _build_header(self) -> None:
        bar = ttk.Frame(self, padding=(12, 6, 12, 4))
        bar.pack(fill=tk.X)

        self.var_hw_status = tk.StringVar(value="正在检测内置 FFmpeg / 显卡 / Ollama...")
        ttk.Label(
            bar,
            textvariable=self.var_hw_status,
            foreground="#0b6bcb",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.RIGHT)

        ttk.Label(bar, text="双语字幕工坊 V1.5", font=("Microsoft YaHei UI", 13, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            bar,
            text="  |  一键全自动一条龙（提取 → AI翻译 → 烧录） · 亦保留手动 Agent 与逐帧校对",
            foreground="#555555",
        ).pack(side=tk.LEFT)

    def _build_pipeline_bar(self) -> None:
        pbar = ttk.LabelFrame(self, text="全自动一条龙工作台（支持内置 AI 直出成片，也可照旧仅提取后手动交外部 Agent 翻译）", padding=(10, 6))
        pbar.pack(fill=tk.X, padx=10, pady=(0, 6))

        ttk.Button(pbar, text="📁 选择原视频文件夹", command=self._choose_video_dir).pack(side=tk.LEFT, padx=(0, 8))

        self.btn_pipeline = ttk.Button(
            pbar,
            text="🚀 一键全自动一条龙（提取 → AI翻译 → 烧录出片）",
            command=self._start_full_pipeline,
        )
        self.btn_pipeline.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop_pipeline = ttk.Button(pbar, text="停止", width=7, command=self._stop_task, state=tk.DISABLED)
        self.btn_stop_pipeline.pack(side=tk.LEFT, padx=(0, 10))

        self.var_pipeline_pause_review = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pbar,
            text="翻译后暂停供抽查（不勾选则全自动直出成片）",
            variable=self.var_pipeline_pause_review,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self.var_pipeline_status = tk.StringVar(value="待命：请先选择原视频文件夹")
        ttk.Label(
            pbar,
            textvariable=self.var_pipeline_status,
            foreground="#15803d",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.RIGHT)

    def _probe_hardware_async(self) -> None:
        def _worker():
            ff = get_ffmpeg_exe()
            ocr_code, _ = detect_ocr_accelerator()
            enc = detect_best_encoder()
            ff_tag = "内置FFmpeg" if str(get_app_dir()) in str(ff) else "系统FFmpeg"
            ocr_short = "PP-OCRv5(显卡)" if ocr_code == "directml" else "PP-OCRv5(CPU)"
            enc_short = {"hevc_amf": "AMD硬编", "hevc_nvenc": "NVENC", "x264_cpu": "CPU软编"}.get(enc, enc)

            ollama_url = self._trans_cfg_dict.get("ollama_url", "http://127.0.0.1:11434")
            models: list[str] = []
            try:
                models = list_ollama_models(ollama_url)
                ol_tag = f"Ollama({len(models)}模型)"
            except Exception:
                ol_tag = "Ollama(未启动/使用API)"

            def _apply():
                self.var_hw_status.set(f"{ff_tag} | {ocr_short} | {ol_tag} | {enc_short}")
                if models and hasattr(self, "cmb_ollama_model"):
                    self.cmb_ollama_model["values"] = models

            self.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    # ==================== 主三标签页 ====================

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.tab_extract = ttk.Frame(self.notebook, padding=8)
        self.tab_translate = ttk.Frame(self.notebook, padding=8)
        self.tab_burn = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_extract, text="  第 1 步：批量提取中文 SRT + 选区  ")
        self.notebook.add(self.tab_translate, text="  第 2 步：内置 AI 自动翻译 & 校对（可选外部 Agent）  ")
        self.notebook.add(self.tab_burn, text="  第 3 步：自由(X,Y)预览调位 & 批量烧录  ")

        self._build_extract_tab()
        self._build_translate_tab()
        self._build_burn_tab()

    # ==================== Tab 1：提取页 ====================

    def _build_extract_tab(self) -> None:
        path_box = ttk.LabelFrame(self.tab_extract, text="目录配置（选原视频目录后会自动填充推荐子目录）", padding=8)
        path_box.pack(fill=tk.X, pady=(0, 8))

        self._make_dir_row(path_box, 0, "原视频文件夹：", self.var_video_dir, self._choose_video_dir)
        self._make_dir_row(path_box, 1, "中文SRT输出：", self.var_zh_dir, lambda: self._browse_dir(self.var_zh_dir, on_done=self._on_zh_dir_changed))
        self._make_dir_row(path_box, 2, "英文SRT目录：", self.var_en_dir, lambda: self._browse_dir(self.var_en_dir, on_done=self._on_en_dir_changed))
        path_box.columnconfigure(1, weight=1)

        paned = ttk.PanedWindow(self.tab_extract, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左半区 ----
        left_frame = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left_frame, weight=5)

        tree_box = ttk.Frame(left_frame)
        tree_box.pack(fill=tk.X, pady=(0, 6))
        cols = ("status", "name", "main_roi", "note_info")
        self.tree_ext_videos = ttk.Treeview(tree_box, columns=cols, show="headings", height=5, selectmode="browse")
        for col, title, width, anchor in (
            ("status", "选区状态", 92, "center"),
            ("name", "批量视频文件名（点选逐个设置）", 175, "w"),
            ("main_roi", "主台词红框范围", 155, "center"),
            ("note_info", "花字/水印", 95, "center"),
        ):
            self.tree_ext_videos.heading(col, text=title)
            self.tree_ext_videos.column(col, width=width, anchor=anchor)
        ext_ysb = ttk.Scrollbar(tree_box, orient="vertical", command=self.tree_ext_videos.yview)
        self.tree_ext_videos.configure(yscrollcommand=ext_ysb.set)
        self.tree_ext_videos.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ext_ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_ext_videos.bind("<<TreeviewSelect>>", lambda _e: self._on_ext_tree_selected())

        ctrl_row1 = ttk.Frame(left_frame)
        ctrl_row1.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(ctrl_row1, text="◀ 上一个", width=9, command=lambda: self._step_ext_video(-1)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(ctrl_row1, text="下一个 ▶", width=9, command=lambda: self._step_ext_video(1)).pack(side=tk.LEFT, padx=(0, 8))
        self.var_ext_quota = tk.StringVar(value="请先选择原视频文件夹（列表将显示全部视频，可逐个单独框选）")
        ttk.Label(ctrl_row1, textvariable=self.var_ext_quota, foreground="#0b6bcb", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ctrl_row2 = ttk.Frame(left_frame)
        ctrl_row2.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(ctrl_row2, text="左图拖拽：").pack(side=tk.LEFT)
        self.var_drag_target = tk.StringVar(value="main")
        ttk.Radiobutton(
            ctrl_row2, text="红框(主台词)", variable=self.var_drag_target, value="main",
            command=lambda: self.ext_canvas.set_active_box("main"),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(
            ctrl_row2, text="蓝框(花字区)", variable=self.var_drag_target, value="note",
            command=lambda: self.ext_canvas.set_active_box("note"),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl_row2, text="恢复默认框", width=10, command=self._reset_current_video_roi).pack(side=tk.RIGHT)
        ttk.Button(ctrl_row2, text="↔ 红框拉满宽度", width=14, command=self._widen_current_main_roi).pack(side=tk.RIGHT, padx=(0, 6))

        self.ext_canvas = PreviewCanvas(left_frame, interactive_roi=True, on_roi_changed=self._on_canvas_roi_dragged)
        self.ext_canvas.pack(fill=tk.BOTH, expand=True)

        seek_bar = ttk.Frame(left_frame, padding=(0, 6, 0, 0))
        seek_bar.pack(fill=tk.X)
        self.var_ext_time = tk.DoubleVar(value=0.0)
        self.var_ext_time_lbl = tk.StringVar(value="00:00 / 00:00")
        self.scl_ext_time = ttk.Scale(seek_bar, from_=0.0, to=100.0, variable=self.var_ext_time, command=self._on_ext_seek)
        self.scl_ext_time.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(seek_bar, textvariable=self.var_ext_time_lbl, width=13).pack(side=tk.LEFT)
        for ratio, lbl in ((0.10, "10%"), (0.42, "42%"), (0.72, "72%"), (0.95, "片尾")):
            ttk.Button(seek_bar, text=lbl, width=5, command=lambda r=ratio: self._jump_ext_ratio(r)).pack(side=tk.LEFT, padx=2)

        # ---- 右半区 ----
        right_scroll = ScrollableFrame(paned)
        paned.add(right_scroll, weight=4)
        rf = right_scroll.inner

        preset_box = ttk.LabelFrame(rf, text="选区预设模板（仅应用到当前选中视频，不覆盖他人）", padding=8)
        preset_box.pack(fill=tk.X, pady=(0, 6))
        self.var_preset_name = tk.StringVar(value=next(iter(self._presets.keys())))
        self.cmb_presets = ttk.Combobox(preset_box, textvariable=self.var_preset_name, values=list(self._presets.keys()), state="readonly")
        self.cmb_presets.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.cmb_presets.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        ttk.Button(preset_box, text="保存为预设", command=self._save_current_preset).pack(side=tk.LEFT)

        batch_roi_box = ttk.LabelFrame(rf, text="批量选区快捷工具（支持每集不同位置）", padding=8)
        batch_roi_box.pack(fill=tk.X, pady=(0, 6))
        self.btn_auto_detect = ttk.Button(
            batch_roi_box,
            text="✨ 自动识别全部视频红框(通用初稿)",
            command=self._start_auto_detect_rois,
        )
        self.btn_auto_detect.pack(fill=tk.X, pady=(0, 5))

        sync_row = ttk.Frame(batch_roi_box)
        sync_row.pack(fill=tk.X)
        ttk.Button(sync_row, text="当前框套用到未定制视频", command=lambda: self._apply_current_roi_batch(only_unset=True)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(sync_row, text="当前框套用到全部视频", command=lambda: self._apply_current_roi_batch(only_unset=False)).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.var_m_y1 = tk.DoubleVar(value=0.81)
        self.var_m_y2 = tk.DoubleVar(value=0.92)
        self.var_m_x1 = tk.DoubleVar(value=0.08)
        self.var_m_x2 = tk.DoubleVar(value=0.92)
        self.var_main_white_only = tk.BooleanVar(value=True)

        main_box = ttk.LabelFrame(rf, text="当前视频主台词区（红框：可直接在左图拖拽）", padding=8)
        main_box.pack(fill=tk.X, pady=(0, 6))
        self._make_ratio_slider(main_box, 0, "上边界 Y1：", self.var_m_y1, self._sync_ext_rois)
        self._make_ratio_slider(main_box, 1, "下边界 Y2：", self.var_m_y2, self._sync_ext_rois)
        self._make_ratio_slider(main_box, 2, "左边界 X1：", self.var_m_x1, self._sync_ext_rois)
        self._make_ratio_slider(main_box, 3, "右边界 X2：", self.var_m_x2, self._sync_ext_rois)
        ttk.Checkbutton(
            main_box,
            text="仅抓纯白主字幕（防红字道具；彩字请取消）",
            variable=self.var_main_white_only,
            command=self._sync_ext_rois,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        main_box.columnconfigure(1, weight=1)

        self.var_extract_notes = tk.BooleanVar(value=True)
        self.var_n_y1 = tk.DoubleVar(value=0.00)
        self.var_n_y2 = tk.DoubleVar(value=1.00)
        self.var_n_x1 = tk.DoubleVar(value=0.00)
        self.var_n_x2 = tk.DoubleVar(value=1.00)
        self.var_note_min_h = tk.DoubleVar(value=0.038)
        self.var_filter_watermark = tk.BooleanVar(value=True)

        note_box = ttk.LabelFrame(rf, text="画面花字 / 背景大字区（蓝框 + 水印过滤）", padding=8)
        note_box.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(
            note_box,
            text="同时提取画面花字/背景大字（输出 *_notes.srt）",
            variable=self.var_extract_notes,
            command=self._sync_ext_rois,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self._make_ratio_slider(note_box, 1, "上边界 Y1：", self.var_n_y1, self._sync_ext_rois)
        self._make_ratio_slider(note_box, 2, "下边界 Y2：", self.var_n_y2, self._sync_ext_rois)
        self._make_ratio_slider(note_box, 3, "左边界 X1：", self.var_n_x1, self._sync_ext_rois)
        self._make_ratio_slider(note_box, 4, "右边界 X2：", self.var_n_x2, self._sync_ext_rois)
        self._make_ratio_slider(note_box, 5, "最小字高过滤：", self.var_note_min_h, self._sync_ext_rois, min_v=0.015, max_v=0.100)
        ttk.Checkbutton(
            note_box,
            text="自动过滤常驻固定水印（AI视频/仅供娱乐等）",
            variable=self.var_filter_watermark,
            command=self._sync_ext_rois,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        note_box.columnconfigure(1, weight=1)

        act_box = ttk.LabelFrame(rf, text="批量提取与提示词生成", padding=8)
        act_box.pack(fill=tk.BOTH, expand=True)

        opt_row = ttk.Frame(act_box)
        opt_row.pack(fill=tk.X)
        self.var_ext_skip = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_row, text="跳过已提取", variable=self.var_ext_skip).pack(side=tk.LEFT)
        ttk.Label(opt_row, text="OCR加速：").pack(side=tk.LEFT, padx=(8, 2))
        self._ocr_accel_labels = ["自动显卡加速 (DirectML)", "仅 CPU (限流防卡顿)"]
        self.var_ocr_accel = tk.StringVar(value=self._ocr_accel_labels[0])
        ttk.Combobox(opt_row, textvariable=self.var_ocr_accel, values=self._ocr_accel_labels, state="readonly", width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)

        opt_row2 = ttk.Frame(act_box)
        opt_row2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(opt_row2, text="OCR模型：").pack(side=tk.LEFT)
        self._ocr_prof_codes = [c for c, _ in OCR_PROFILE_CHOICES]
        self._ocr_prof_labels = [lbl for _, lbl in OCR_PROFILE_CHOICES]
        self.var_ocr_profile_label = tk.StringVar(value=self._ocr_prof_labels[0])
        ttk.Combobox(opt_row2, textvariable=self.var_ocr_profile_label, values=self._ocr_prof_labels, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)

        opt_row3 = ttk.Frame(act_box)
        opt_row3.pack(fill=tk.X, pady=(3, 0))
        self.var_auto_detect_unset = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_row3, text="提取前自动识别未定制视频红框（免人工逐条框选）", variable=self.var_auto_detect_unset).pack(side=tk.LEFT)

        btn_row1 = ttk.Frame(act_box, padding=(0, 6))
        btn_row1.pack(fill=tk.X)
        self.btn_run_extract = ttk.Button(btn_row1, text="▶ 仅执行第1步：批量提取中文 SRT", command=self._start_extract)
        self.btn_run_extract.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_stop_extract = ttk.Button(btn_row1, text="停止", width=8, command=self._stop_task, state=tk.DISABLED)
        self.btn_stop_extract.pack(side=tk.LEFT)

        btn_row2 = ttk.Frame(act_box, padding=(0, 0, 0, 6))
        btn_row2.pack(fill=tk.X)
        ttk.Button(btn_row2, text="编辑多地方言词典", command=self._edit_glossary_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_row2, text="📋 手动复制 Agent 提示词", command=self._show_prompt_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_row2, text="打开中文目录", command=lambda: self._open_folder(self.var_zh_dir.get())).pack(side=tk.LEFT)

        self.var_ext_cur_prog = tk.DoubleVar(value=0.0)
        self.var_ext_cur_msg = tk.StringVar(value="当前视频：等待开始...")
        self.var_ext_tot_prog = tk.DoubleVar(value=0.0)
        self.var_ext_tot_msg = tk.StringVar(value="总进度：0 / 0")

        ttk.Label(act_box, textvariable=self.var_ext_cur_msg, foreground="#0b6bcb").pack(anchor="w")
        ttk.Progressbar(act_box, variable=self.var_ext_cur_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 4))
        ttk.Label(act_box, textvariable=self.var_ext_tot_msg, foreground="#333333").pack(anchor="w")
        ttk.Progressbar(act_box, variable=self.var_ext_tot_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 6))

        self.txt_ext_log = tk.Text(act_box, height=5, wrap=tk.WORD, bg="#f9fafb", relief=tk.SOLID, borderwidth=1)
        self.txt_ext_log.pack(fill=tk.BOTH, expand=True)

        self._sync_ext_rois(persist=False)

    # ==================== Tab 2：内置 AI 自动翻译 & 校对页 ====================

    def _build_translate_tab(self) -> None:
        top_dir = ttk.LabelFrame(self.tab_translate, text="字幕目录联动（若改用外部 Agent 手动翻译，可直接点右侧按钮复制提示词）", padding=8)
        top_dir.pack(fill=tk.X, pady=(0, 8))
        self._make_dir_row(top_dir, 0, "中文SRT目录：", self.var_zh_dir, lambda: self._browse_dir(self.var_zh_dir, on_done=self._on_zh_dir_changed))
        self._make_dir_row(top_dir, 1, "英文SRT输出：", self.var_en_dir, lambda: self._browse_dir(self.var_en_dir, on_done=self._on_en_dir_changed))
        ttk.Button(top_dir, text="📋 改用手动 Agent 提示词", command=self._show_prompt_dialog).grid(row=0, column=3, rowspan=2, padx=(8, 0), sticky="ns")
        top_dir.columnconfigure(1, weight=1)

        paned = ttk.PanedWindow(self.tab_translate, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左半区：中英对照校对表 ----
        left_pane = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left_pane, weight=5)

        sel_bar = ttk.Frame(left_pane)
        sel_bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(sel_bar, text="校对字幕文件：", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        self.var_trans_review_file = tk.StringVar()
        self.cmb_trans_review_file = ttk.Combobox(sel_bar, textvariable=self.var_trans_review_file, state="readonly", width=28)
        self.cmb_trans_review_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.cmb_trans_review_file.bind("<<ComboboxSelected>>", lambda _e: self._load_review_file_rows())
        ttk.Button(sel_bar, text="🔄 刷新清单", command=self._refresh_translate_review_list).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_bar, text="打开英文目录", command=lambda: self._open_folder(self.var_en_dir.get())).pack(side=tk.LEFT)

        tree_frame = ttk.Frame(left_pane)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        cols = ("idx", "time", "zh", "en")
        self.tree_trans_lines = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        for col, title, width, anchor in (
            ("idx", "#", 45, "center"),
            ("time", "时间轴", 145, "center"),
            ("zh", "原中文（OCR提取）", 230, "w"),
            ("en", "英文译文（双击本行可直接改字保存）", 290, "w"),
        ):
            self.tree_trans_lines.heading(col, text=title)
            self.tree_trans_lines.column(col, width=width, anchor=anchor)
        tysb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree_trans_lines.yview)
        self.tree_trans_lines.configure(yscrollcommand=tysb.set)
        self.tree_trans_lines.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_trans_lines.bind("<Double-1>", lambda _e: self._edit_selected_trans_line())

        ttk.Label(
            left_pane,
            text="提示：自动翻译完成后可直接在此核对各集，双击任意一行即可秒改英文并自动同步到烧录页。",
            foreground="#555555",
        ).pack(anchor="w", pady=(6, 0))

        # ---- 右半区：模型引擎配置与执行 ----
        right_scroll = ScrollableFrame(paned)
        paned.add(right_scroll, weight=4)
        tf = right_scroll.inner

        eng_box = ttk.LabelFrame(tf, text="AI 翻译引擎配置（本地显存模型 / 云端 API 二选一）", padding=8)
        eng_box.pack(fill=tk.X, pady=(0, 8))

        self.var_trans_provider = tk.StringVar(value=str(self._trans_cfg_dict.get("provider", "ollama")))
        p_row = ttk.Frame(eng_box)
        p_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(p_row, text="推理方式：").pack(side=tk.LEFT)
        ttk.Radiobutton(p_row, text="本机 Ollama 模型 (默认 B站9B)", variable=self.var_trans_provider, value="ollama", command=self._save_trans_cfg_from_ui).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(p_row, text="云端 OpenAI 兼容 API", variable=self.var_trans_provider, value="openai", command=self._save_trans_cfg_from_ui).pack(side=tk.LEFT, padx=6)

        # Ollama 区块
        ol_box = ttk.LabelFrame(eng_box, text="本机 Ollama 设置（支持 B站 Index-Translate-9B / Hy-MT / Qwen）", padding=6)
        ol_box.pack(fill=tk.X, pady=(0, 6))
        self.var_ollama_url = tk.StringVar(value=str(self._trans_cfg_dict.get("ollama_url", "http://127.0.0.1:11434")))
        self.var_ollama_model = tk.StringVar(value=str(self._trans_cfg_dict.get("ollama_model", "hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M")))
        ttk.Label(ol_box, text="服务地址：").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(ol_box, textvariable=self.var_ollama_url).grid(row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(ol_box, text="选择模型：").grid(row=1, column=0, sticky="w", pady=2)
        self.cmb_ollama_model = ttk.Combobox(ol_box, textvariable=self.var_ollama_model, values=[self.var_ollama_model.get()])
        self.cmb_ollama_model.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(ol_box, text="🔄 刷新列表", width=10, command=self._refresh_ollama_models_ui).grid(row=1, column=2, pady=2)
        ol_box.columnconfigure(1, weight=1)

        # 云端 API 区块
        api_box = ttk.LabelFrame(eng_box, text="云端 OpenAI 兼容 API 设置（DeepSeek / 百炼 / 硅基流动等）", padding=6)
        api_box.pack(fill=tk.X, pady=(0, 6))
        self.var_api_base = tk.StringVar(value=str(self._trans_cfg_dict.get("api_base_url", "https://api.deepseek.com/v1")))
        self.var_api_key = tk.StringVar(value=str(self._trans_cfg_dict.get("api_key", "")))
        self.var_api_model = tk.StringVar(value=str(self._trans_cfg_dict.get("api_model", "deepseek-chat")))
        ttk.Label(api_box, text="Base URL：").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(api_box, textvariable=self.var_api_base).grid(row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(api_box, text="API Key：").grid(row=1, column=0, sticky="w", pady=2)
        self._api_key_entry = ttk.Entry(api_box, textvariable=self.var_api_key, show="*")
        self._api_key_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        self._api_key_shown = False
        self._btn_toggle_key = ttk.Button(api_box, text="显示", width=6, command=self._toggle_api_key_mask)
        self._btn_toggle_key.grid(row=1, column=2, pady=2)
        ttk.Label(api_box, text="模型名称：").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(api_box, textvariable=self.var_api_model).grid(row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        api_box.columnconfigure(1, weight=1)

        t_btn_row = ttk.Frame(eng_box)
        t_btn_row.pack(fill=tk.X)
        self.btn_test_trans = ttk.Button(t_btn_row, text="🔍 测试连通性并试翻一句", command=self._test_translator_async)
        self.btn_test_trans.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(t_btn_row, text="💾 保存翻译配置", command=lambda: self._save_trans_cfg_from_ui(show_tip=True)).pack(side=tk.LEFT)

        # 翻译质量与显存选项
        pol_box = ttk.LabelFrame(tf, text="翻译质量与显存管理策略", padding=8)
        pol_box.pack(fill=tk.X, pady=(0, 8))

        self.var_trans_batch_size = tk.IntVar(value=int(self._trans_cfg_dict.get("batch_size", 12)))
        self.var_trans_skip = tk.BooleanVar(value=bool(self._trans_cfg_dict.get("skip_existing", True)))
        self.var_trans_dialect = tk.BooleanVar(value=bool(self._trans_cfg_dict.get("auto_detect_dialect", True)))
        self.var_trans_unload = tk.BooleanVar(value=bool(self._trans_cfg_dict.get("unload_after_done", True)))

        r_b = ttk.Frame(pol_box)
        r_b.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(r_b, text="每批条数（滑动窗口）：").pack(side=tk.LEFT)
        ttk.Spinbox(r_b, from_=6, to=20, textvariable=self.var_trans_batch_size, width=6).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(r_b, text="跳过已翻完整文件", variable=self.var_trans_skip).pack(side=tk.LEFT)

        r_b2 = ttk.Frame(pol_box)
        r_b2.pack(fill=tk.X)
        ttk.Checkbutton(r_b2, text="阶段A自动判别方言年代", variable=self.var_trans_dialect).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r_b2, text="译完自动卸载显存(防8G爆显存)", variable=self.var_trans_unload).pack(side=tk.LEFT)

        # 执行区
        run_t_box = ttk.LabelFrame(tf, text="执行批量 AI 翻译", padding=8)
        run_t_box.pack(fill=tk.BOTH, expand=True)

        tr_btns = ttk.Frame(run_t_box, padding=(0, 0, 0, 6))
        tr_btns.pack(fill=tk.X)
        self.btn_run_translate = ttk.Button(tr_btns, text="▶ 一键 AI 批量翻译全部中文 SRT", command=self._start_translate)
        self.btn_run_translate.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_stop_translate = ttk.Button(tr_btns, text="停止", width=7, command=self._stop_task, state=tk.DISABLED)
        self.btn_stop_translate.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tr_btns, text="编辑方言词典", command=self._edit_glossary_dialog).pack(side=tk.LEFT)

        self.var_trans_cur_prog = tk.DoubleVar(value=0.0)
        self.var_trans_cur_msg = tk.StringVar(value="当前文件：等待开始...")
        self.var_trans_tot_prog = tk.DoubleVar(value=0.0)
        self.var_trans_tot_msg = tk.StringVar(value="总进度：0 / 0")

        ttk.Label(run_t_box, textvariable=self.var_trans_cur_msg, foreground="#0b6bcb").pack(anchor="w")
        ttk.Progressbar(run_t_box, variable=self.var_trans_cur_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 4))
        ttk.Label(run_t_box, textvariable=self.var_trans_tot_msg, foreground="#333333").pack(anchor="w")
        ttk.Progressbar(run_t_box, variable=self.var_trans_tot_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 6))

        self.txt_trans_log = tk.Text(run_t_box, height=6, wrap=tk.WORD, bg="#f9fafb", relief=tk.SOLID, borderwidth=1)
        self.txt_trans_log.pack(fill=tk.BOTH, expand=True)

    # ==================== Tab 3：自由 (X, Y) 预览调位与烧录页 ====================

    def _build_burn_tab(self) -> None:
        p_box = ttk.LabelFrame(self.tab_burn, text="目录联动与英文 SRT 匹配校验", padding=8)
        p_box.pack(fill=tk.X, pady=(0, 8))
        self._make_dir_row(p_box, 0, "原视频文件夹：", self.var_video_dir, self._choose_video_dir)
        self._make_dir_row(p_box, 1, "中文/坐标文件夹：", self.var_zh_dir, lambda: self._browse_dir(self.var_zh_dir, on_done=self._on_zh_dir_changed))
        self._make_dir_row(p_box, 2, "翻译好的英文SRT：", self.var_en_dir, lambda: self._browse_dir(self.var_en_dir, on_done=self._on_en_dir_changed))
        self._make_dir_row(p_box, 3, "成品输出文件夹：", self.var_out_dir, lambda: self._browse_dir(self.var_out_dir))
        ttk.Button(p_box, text="🔄 扫描匹配状态", command=self._refresh_burn_matches).grid(row=0, column=3, rowspan=4, padx=(8, 0), sticky="ns")
        p_box.columnconfigure(1, weight=1)

        paned = ttk.PanedWindow(self.tab_burn, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_pane = ttk.Frame(paned, padding=(0, 0, 8, 0))
        paned.add(left_pane, weight=5)

        cols = ("name", "res", "main", "notes", "status")
        tree_wrap = ttk.Frame(left_pane)
        tree_wrap.pack(fill=tk.X, pady=(0, 6))
        self.tree_matches = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=5, selectmode="browse")
        for col, title, width in (
            ("name", "视频文件名", 170),
            ("res", "分辨率", 85),
            ("main", "主英文SRT", 80),
            ("notes", "花字英文/坐标", 110),
            ("status", "匹配校验状态", 165),
        ):
            self.tree_matches.heading(col, text=title)
            self.tree_matches.column(col, width=width, anchor="w" if col in ("name", "status") else "center")
        ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree_matches.yview)
        self.tree_matches.configure(yscrollcommand=ysb.set)
        self.tree_matches.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_matches.bind("<<TreeviewSelect>>", lambda _e: self._on_burn_row_selected())

        prev_bar = ttk.Frame(left_pane)
        prev_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(prev_bar, text="实时双语预览：", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        self.var_burn_drag_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            prev_bar,
            text="左图拖拽调整(X,Y)",
            variable=self.var_burn_drag_enable,
            command=self._update_burn_preview,
        ).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(prev_bar, text="🔍 跳到花字帧", command=self._jump_to_next_note_frame).pack(side=tk.LEFT, padx=4)
        self.var_burn_sample_text = tk.StringVar(value="The first Dave Jianguo Award goes to Dave Jianguo!")
        ttk.Entry(prev_bar, textvariable=self.var_burn_sample_text, width=26).pack(side=tk.RIGHT)
        ttk.Label(prev_bar, text="示例：").pack(side=tk.RIGHT)

        self.burn_canvas = PreviewCanvas(
            left_pane,
            interactive_roi=False,
            interactive_sub_pos=True,
            on_sub_dragged=self._on_burn_sub_dragged,
        )
        self.burn_canvas.pack(fill=tk.BOTH, expand=True)

        b_seek = ttk.Frame(left_pane, padding=(0, 6, 0, 0))
        b_seek.pack(fill=tk.X)
        self.var_burn_time = tk.DoubleVar(value=0.0)
        self.var_burn_time_lbl = tk.StringVar(value="00:00 / 00:00")
        self.scl_burn_time = ttk.Scale(b_seek, from_=0.0, to=100.0, variable=self.var_burn_time, command=self._on_burn_seek)
        self.scl_burn_time.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(b_seek, textvariable=self.var_burn_time_lbl, width=13).pack(side=tk.LEFT)
        for ratio, lbl in ((0.02, "开头"), (0.43, "43%"), (0.72, "72%"), (0.98, "片尾")):
            ttk.Button(b_seek, text=lbl, width=5, command=lambda r=ratio: self._jump_burn_ratio(r)).pack(side=tk.LEFT, padx=2)

        right_scroll = ScrollableFrame(paned)
        paned.add(right_scroll, weight=4)
        bf = right_scroll.inner

        sty_box = ttk.LabelFrame(bf, text="英文字幕位置 (X, Y) 与外观调节（支持自动防重叠 + 左图自由拖拽）", padding=8)
        sty_box.pack(fill=tk.X, pady=(0, 8))

        self.var_burn_per_video = tk.BooleanVar(value=False)
        top_srow = ttk.Frame(sty_box)
        top_srow.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            top_srow,
            text="单独微调当前选中视频（不影响其他）",
            variable=self.var_burn_per_video,
            command=self._on_toggle_burn_per_video,
        ).pack(side=tk.LEFT)
        ttk.Button(top_srow, text="套用当前样式到全部", command=self._apply_burn_style_to_all).pack(side=tk.RIGHT)

        self._auto_place_codes = [c for c, _ in AUTO_PLACE_CHOICES]
        self._auto_place_labels = [lbl for _, lbl in AUTO_PLACE_CHOICES]
        self.var_auto_place_label = tk.StringVar(value=self._auto_place_labels[0])
        auto_row = ttk.Frame(sty_box)
        auto_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 2))
        ttk.Label(auto_row, text="英文定位模式：").pack(side=tk.LEFT)
        cmb_auto = ttk.Combobox(auto_row, textvariable=self.var_auto_place_label, values=self._auto_place_labels, state="readonly")
        cmb_auto.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cmb_auto.bind("<<ComboboxSelected>>", lambda _e: self._on_burn_style_edited())

        self.var_auto_place_info = tk.StringVar(value="选择视频后将自动检测原中文字幕位置并防撞排版")
        ttk.Label(sty_box, textvariable=self.var_auto_place_info, foreground="#0b6bcb", font=("Microsoft YaHei UI", 8, "bold")).grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 2))

        self.var_burn_coord_info = tk.StringVar(value="主字幕坐标: X=50.0%, Y=94.8% | 可直接左图按住文字拖动")
        ttk.Label(sty_box, textvariable=self.var_burn_coord_info, foreground="#15803d", font=("Microsoft YaHei UI", 8)).grid(row=3, column=0, columnspan=4, sticky="w", pady=(0, 4))

        self.var_pos_x = tk.DoubleVar(value=0.50)
        self.var_pos_y = tk.DoubleVar(value=0.948)
        self.var_auto_gap = tk.DoubleVar(value=0.012)
        self.var_font_size = tk.DoubleVar(value=0.036)
        self.var_color_name = tk.StringVar(value="暖白")
        self.var_outline_w = tk.DoubleVar(value=3.0)
        self.var_semi_box = tk.BooleanVar(value=False)
        self.var_note_mult = tk.DoubleVar(value=1.00)
        self.var_note_gap = tk.IntVar(value=8)
        self.var_zoom_comp = tk.BooleanVar(value=True)

        self._make_ratio_slider_with_btn(sty_box, 4, "主字幕水平 X：", self.var_pos_x, self._on_burn_style_edited, 0.10, 0.90, "居中", self._center_main_x)
        self._make_ratio_slider_with_btn(sty_box, 5, "主字幕垂直 Y：", self.var_pos_y, self._on_pos_y_slider_moved, 0.08, 0.985, "自动", self._restore_main_auto_place)
        self._make_ratio_slider(sty_box, 6, "中英安全间距：", self.var_auto_gap, self._on_burn_style_edited, min_v=0.005, max_v=0.040)
        self._make_ratio_slider(sty_box, 7, "主英文字号大小：", self.var_font_size, self._on_burn_style_edited, min_v=0.018, max_v=0.065)

        row8 = ttk.Frame(sty_box)
        row8.grid(row=8, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Label(row8, text="字体颜色：").pack(side=tk.LEFT)
        cmb_col = ttk.Combobox(row8, textvariable=self.var_color_name, values=list(COLOR_PRESETS.keys()), state="readonly", width=8)
        cmb_col.pack(side=tk.LEFT, padx=(0, 8))
        cmb_col.bind("<<ComboboxSelected>>", lambda _e: self._on_burn_style_edited())
        ttk.Checkbutton(row8, text="半透明黑底框", variable=self.var_semi_box, command=self._on_burn_style_edited).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(row8, text="缩放高度补偿", variable=self.var_zoom_comp, command=self._on_burn_style_edited).pack(side=tk.LEFT)

        self._make_num_slider(sty_box, 9, "描边粗细(px)：", self.var_outline_w, self._on_burn_style_edited, 1.0, 6.0, "{:.1f}")
        self._make_num_slider(sty_box, 10, "花字大小倍率：", self.var_note_mult, self._on_burn_style_edited, 0.60, 1.50, "{:.2f}x")
        self._make_num_slider(sty_box, 11, "花字垂直间距：", self.var_note_gap, self._on_burn_style_edited, 2, 24, "{:.0f}px")

        note_ctrl_row = ttk.Frame(sty_box)
        note_ctrl_row.grid(row=12, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.var_cur_note_info = tk.StringVar(value="当前帧花字：无（可点左图「跳到花字帧」后直接拖动）")
        ttk.Label(note_ctrl_row, textvariable=self.var_cur_note_info, foreground="#b45309", font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(note_ctrl_row, text="复位本条", width=8, command=self._reset_cur_note_override).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(note_ctrl_row, text="复位本集花字", width=11, command=self._reset_all_note_overrides).pack(side=tk.RIGHT)
        sty_box.columnconfigure(1, weight=1)

        dedup_box = ttk.LabelFrame(bf, text="多维防查重引擎（已默认勾选 8 项无损感去重）", padding=8)
        dedup_box.pack(fill=tk.X, pady=(0, 8))

        self.dedup_vars: dict[str, tk.BooleanVar] = {}
        for idx, (key, label) in enumerate(DEDUP_LABELS):
            var = tk.BooleanVar(value=DEFAULT_DEDUP_OPTIONS[key])
            self.dedup_vars[key] = var
            cb = ttk.Checkbutton(
                dedup_box,
                text=label,
                variable=var,
                command=self._update_burn_preview if key in ("micro_zoom", "blur_frame") else None,
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=4, pady=2)

        d_btn_row = ttk.Frame(dedup_box)
        d_btn_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        ttk.Button(d_btn_row, text="恢复默认8项去重", command=lambda: self._set_dedup_preset(True)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(d_btn_row, text="全部取消(原画质)", command=lambda: self._set_dedup_preset(False)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(d_btn_row, text="画框顶标：").pack(side=tk.LEFT)
        self.var_top_banner = tk.StringVar(value="90s Rural China · Official")
        ent_banner = ttk.Entry(d_btn_row, textvariable=self.var_top_banner)
        ent_banner.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ent_banner.bind("<FocusOut>", lambda _e: self._on_burn_style_edited())
        dedup_box.columnconfigure(0, weight=1)
        dedup_box.columnconfigure(1, weight=1)

        run_box = ttk.LabelFrame(bf, text="硬件编码与批量压制", padding=8)
        run_box.pack(fill=tk.BOTH, expand=True)

        enc_row = ttk.Frame(run_box)
        enc_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(enc_row, text="编码器：").pack(side=tk.LEFT)
        self._enc_labels = [lbl for _, lbl in ENCODER_CHOICES]
        self._enc_codes = [code for code, _ in ENCODER_CHOICES]
        self.var_enc_label = tk.StringVar(value=self._enc_labels[0])
        ttk.Combobox(enc_row, textvariable=self.var_enc_label, values=self._enc_labels, state="readonly", width=24).pack(side=tk.LEFT, padx=(0, 8))
        self.var_burn_skip = tk.BooleanVar(value=True)
        ttk.Checkbutton(enc_row, text="跳过已压制成品", variable=self.var_burn_skip).pack(side=tk.LEFT)

        b_act1 = ttk.Frame(run_box, padding=(0, 2, 0, 6))
        b_act1.pack(fill=tk.X)
        self.btn_test_burn = ttk.Button(b_act1, text="试压选中前15秒", command=lambda: self._start_burn(test_sec=15))
        self.btn_test_burn.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_run_burn = ttk.Button(b_act1, text="▶ 一键批量烧录全部", command=lambda: self._start_burn(test_sec=0))
        self.btn_run_burn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_stop_burn = ttk.Button(b_act1, text="停止", width=7, command=self._stop_task, state=tk.DISABLED)
        self.btn_stop_burn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(b_act1, text="打开成品目录", command=lambda: self._open_folder(self.var_out_dir.get())).pack(side=tk.LEFT)

        self.var_burn_cur_prog = tk.DoubleVar(value=0.0)
        self.var_burn_cur_msg = tk.StringVar(value="当前视频：等待开始...")
        self.var_burn_tot_prog = tk.DoubleVar(value=0.0)
        self.var_burn_tot_msg = tk.StringVar(value="总进度：0 / 0")

        ttk.Label(run_box, textvariable=self.var_burn_cur_msg, foreground="#0b6bcb").pack(anchor="w")
        ttk.Progressbar(run_box, variable=self.var_burn_cur_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 4))
        ttk.Label(run_box, textvariable=self.var_burn_tot_msg, foreground="#333333").pack(anchor="w")
        ttk.Progressbar(run_box, variable=self.var_burn_tot_prog, maximum=100.0).pack(fill=tk.X, pady=(1, 6))

        self.txt_burn_log = tk.Text(run_box, height=5, wrap=tk.WORD, bg="#f9fafb", relief=tk.SOLID, borderwidth=1)
        self.txt_burn_log.pack(fill=tk.BOTH, expand=True)

    # ==================== 通用界面辅助控件与目录持久化 ====================

    def _make_dir_row(self, parent, row: int, label: str, var: tk.StringVar, browse_cmd) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(parent, text="浏览...", width=8, command=browse_cmd).grid(row=row, column=2, pady=2)

    def _make_ratio_slider(self, parent, row: int, label: str, var: tk.DoubleVar, on_change, min_v: float = 0.0, max_v: float = 1.0) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        val_lbl = ttk.Label(parent, text=f"{var.get() * 100:.1f}%", width=7, anchor="e")
        val_lbl.grid(row=row, column=2, sticky="e", padx=(6, 0))

        def _cb(_val=None):
            val_lbl.configure(text=f"{var.get() * 100:.1f}%")
            on_change()

        ttk.Scale(parent, from_=min_v, to=max_v, variable=var, command=_cb).grid(row=row, column=1, sticky="ew", padx=4)
        var.trace_add("write", lambda *_: val_lbl.configure(text=f"{var.get() * 100:.1f}%"))

    def _make_ratio_slider_with_btn(
        self, parent, row: int, label: str, var: tk.DoubleVar, on_change,
        min_v: float, max_v: float, btn_text: str, btn_cmd,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        val_lbl = ttk.Label(parent, text=f"{var.get() * 100:.1f}%", width=7, anchor="e")
        val_lbl.grid(row=row, column=2, sticky="e", padx=(4, 4))

        def _cb(_val=None):
            val_lbl.configure(text=f"{var.get() * 100:.1f}%")
            on_change()

        ttk.Scale(parent, from_=min_v, to=max_v, variable=var, command=_cb).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text=btn_text, width=5, command=btn_cmd).grid(row=row, column=3, pady=2)
        var.trace_add("write", lambda *_: val_lbl.configure(text=f"{var.get() * 100:.1f}%"))

    def _make_num_slider(self, parent, row: int, label: str, var, on_change, min_v: float, max_v: float, fmt: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        val_lbl = ttk.Label(parent, text=fmt.format(var.get()), width=7, anchor="e")
        val_lbl.grid(row=row, column=2, sticky="e", padx=(6, 0))

        def _cb(_val=None):
            val_lbl.configure(text=fmt.format(var.get()))
            on_change()

        ttk.Scale(parent, from_=min_v, to=max_v, variable=var, command=_cb).grid(row=row, column=1, sticky="ew", padx=4)
        var.trace_add("write", lambda *_: val_lbl.configure(text=fmt.format(var.get())))

    def _browse_dir(self, var: tk.StringVar, on_done=None) -> None:
        cur = var.get().strip() or str(Path.home() / "Desktop")
        chosen = filedialog.askdirectory(initialdir=cur if os.path.isdir(cur) else str(Path.home() / "Desktop"))
        if chosen:
            var.set(chosen)
            if on_done:
                on_done()

    def _choose_video_dir(self) -> None:
        cur = self.var_video_dir.get().strip() or str(Path.home() / "Desktop")
        chosen = filedialog.askdirectory(initialdir=cur if os.path.isdir(cur) else str(Path.home() / "Desktop"))
        if not chosen:
            return
        vdir = Path(chosen)
        old_auto = self._last_auto_vdir

        def _sync_sub(var: tk.StringVar, sub_name: str):
            val = var.get().strip()
            if not val or (old_auto is not None and Path(val) == old_auto / sub_name):
                var.set(str(vdir / sub_name))

        self.var_video_dir.set(str(vdir))
        _sync_sub(self.var_zh_dir, "zh_srt")
        _sync_sub(self.var_en_dir, "en_srt")
        _sync_sub(self.var_out_dir, "output_bilingual")
        self._last_auto_vdir = vdir
        self._burn_zh_bands.clear()

        self._load_video_roi_store()
        self._load_burn_style_store()
        self._load_extract_video_list(vdir)
        self._refresh_translate_review_list()
        self._refresh_burn_matches()
        self.var_pipeline_status.set(f"已加载 {len(self._ext_videos)} 个视频，可点「🚀 一键全自动一条龙」直接出片")

    def _on_zh_dir_changed(self) -> None:
        self._burn_zh_bands.clear()
        self._load_video_roi_store()
        self._load_burn_style_store()
        self._refresh_ext_tree_all()
        if self._ext_current_video_name:
            self._apply_video_cfg_to_ui(self._ext_current_video_name)
        self._refresh_translate_review_list()
        self._refresh_burn_matches()

    def _on_en_dir_changed(self) -> None:
        self._refresh_translate_review_list()
        self._refresh_burn_matches()

    def _open_folder(self, folder_str: str) -> None:
        p = Path(folder_str.strip()) if folder_str.strip() else None
        if p and p.exists():
            os.startfile(str(p))
        else:
            messagebox.showinfo("提示", "该目录尚未生成，请先运行对应操作或确认路径。")

    # ==================== Tab 1 逐视频选区存储与逻辑 ====================

    def _get_roi_store_path(self) -> Optional[Path]:
        zh_s = self.var_zh_dir.get().strip()
        return (Path(zh_s) / "_video_rois.json") if zh_s else None

    def _normalize_ext_cfg(self, raw: Optional[dict], default_source: str = "manual") -> dict:
        base = self._make_default_ext_cfg()
        if isinstance(raw, dict):
            m = raw.get("main_roi")
            if isinstance(m, (list, tuple)) and len(m) == 4:
                base["main_roi"] = [round(float(v), 3) for v in m]
            n = raw.get("note_roi")
            if isinstance(n, (list, tuple)) and len(n) == 4:
                base["note_roi"] = [round(float(v), 3) for v in n]
            if "main_white_only" in raw:
                base["main_white_only"] = bool(raw["main_white_only"])
            if "extract_notes" in raw:
                base["extract_notes"] = bool(raw["extract_notes"])
            if "note_min_height_ratio" in raw:
                base["note_min_height_ratio"] = round(float(raw["note_min_height_ratio"]), 3)
            if "filter_watermark" in raw:
                base["filter_watermark"] = bool(raw["filter_watermark"])
            base["source"] = str(raw.get("source", default_source))
        return base

    def _load_video_roi_store(self) -> None:
        self._ext_default_cfg = self._make_default_ext_cfg()
        self._ext_video_configs = {}
        store_path = self._get_roi_store_path()
        if not store_path or not store_path.exists():
            return
        try:
            payload = json.loads(store_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("default"), dict):
                self._ext_default_cfg = self._normalize_ext_cfg(payload["default"], default_source="default")
            videos_raw = payload.get("videos", {})
            if isinstance(videos_raw, dict):
                for name, cfg in videos_raw.items():
                    self._ext_video_configs[name] = self._normalize_ext_cfg(cfg, default_source="manual")
        except Exception:
            pass

    def _save_video_roi_store(self) -> None:
        store_path = self._get_roi_store_path()
        if not store_path:
            return
        try:
            store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"default": self._ext_default_cfg, "videos": self._ext_video_configs}
            store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _read_ui_ext_cfg_dict(self, source: str = "manual") -> dict:
        return {
            "main_roi": [
                round(min(self.var_m_x1.get(), self.var_m_x2.get()), 3),
                round(min(self.var_m_y1.get(), self.var_m_y2.get()), 3),
                round(max(self.var_m_x1.get(), self.var_m_x2.get()), 3),
                round(max(self.var_m_y1.get(), self.var_m_y2.get()), 3),
            ],
            "main_white_only": bool(self.var_main_white_only.get()),
            "extract_notes": bool(self.var_extract_notes.get()),
            "note_roi": [
                round(min(self.var_n_x1.get(), self.var_n_x2.get()), 3),
                round(min(self.var_n_y1.get(), self.var_n_y2.get()), 3),
                round(max(self.var_n_x1.get(), self.var_n_x2.get()), 3),
                round(max(self.var_n_y1.get(), self.var_n_y2.get()), 3),
            ],
            "note_min_height_ratio": round(float(self.var_note_min_h.get()), 3),
            "filter_watermark": bool(self.var_filter_watermark.get()),
            "source": source,
        }

    def _get_selected_ocr_profile(self) -> str:
        lbl = self.var_ocr_profile_label.get()
        idx = self._ocr_prof_labels.index(lbl) if lbl in self._ocr_prof_labels else 0
        return self._ocr_prof_codes[idx]

    def _dict_to_extract_config(self, cfg_dict: dict) -> ExtractConfig:
        accel = "auto" if "DirectML" in self.var_ocr_accel.get() else "cpu"
        mx1, my1, mx2, my2 = cfg_dict["main_roi"]
        nx1, ny1, nx2, ny2 = cfg_dict["note_roi"]
        return ExtractConfig(
            main_roi=(mx1, my1, mx2, my2),
            main_white_only=bool(cfg_dict["main_white_only"]),
            extract_notes=bool(cfg_dict["extract_notes"]),
            note_roi=(nx1, ny1, nx2, ny2),
            note_min_height_ratio=float(cfg_dict["note_min_height_ratio"]),
            filter_watermark=bool(cfg_dict.get("filter_watermark", True)),
            skip_existing=bool(self.var_ext_skip.get()),
            accel_mode=accel,
            ocr_profile=self._get_selected_ocr_profile(),
        )

    def _apply_video_cfg_to_ui(self, video_name: Optional[str]) -> None:
        cfg = self._ext_video_configs.get(video_name, self._ext_default_cfg) if video_name else self._ext_default_cfg
        self._ext_loading_ui = True
        try:
            mx1, my1, mx2, my2 = cfg["main_roi"]
            self.var_m_x1.set(mx1)
            self.var_m_y1.set(my1)
            self.var_m_x2.set(mx2)
            self.var_m_y2.set(my2)
            self.var_main_white_only.set(bool(cfg.get("main_white_only", True)))
            self.var_extract_notes.set(bool(cfg.get("extract_notes", True)))
            nx1, ny1, nx2, ny2 = cfg.get("note_roi", [0.0, 0.0, 1.0, 1.0])
            self.var_n_x1.set(nx1)
            self.var_n_y1.set(ny1)
            self.var_n_x2.set(nx2)
            self.var_n_y2.set(ny2)
            self.var_note_min_h.set(float(cfg.get("note_min_height_ratio", 0.038)))
            self.var_filter_watermark.set(bool(cfg.get("filter_watermark", True)))
        finally:
            self._ext_loading_ui = False
        self._sync_ext_rois(persist=False)
        self._update_ext_quota_label()

    def _format_ext_row_values(self, vid_name: str) -> tuple[str, str, str, str]:
        custom = self._ext_video_configs.get(vid_name)
        if custom is not None:
            src = custom.get("source", "manual")
            status_lbl = "✨ 自动识别" if src == "auto" else "★ 已自定义"
            cfg = custom
        else:
            status_lbl = "○ 默认区域"
            cfg = self._ext_default_cfg

        mx1, my1, mx2, my2 = cfg["main_roi"]
        roi_str = f"Y:{my1*100:.0f}~{my2*100:.0f}% X:{mx1*100:.0f}~{mx2*100:.0f}%"
        if cfg.get("extract_notes", True):
            note_str = "花字+去水印" if cfg.get("filter_watermark", True) else "提取花字"
        else:
            note_str = "仅主字幕"
        return (status_lbl, vid_name, roi_str, note_str)

    def _refresh_ext_tree_all(self) -> None:
        if not hasattr(self, "tree_ext_videos"):
            return
        cur_sel = self._ext_current_video_name
        for row in self.tree_ext_videos.get_children():
            self.tree_ext_videos.delete(row)
        target_iid = None
        for i, p in enumerate(self._ext_videos):
            iid = str(i)
            self.tree_ext_videos.insert("", tk.END, iid=iid, values=self._format_ext_row_values(p.name))
            if cur_sel and p.name == cur_sel:
                target_iid = iid
        if target_iid is not None:
            self.tree_ext_videos.selection_set(target_iid)
            self.tree_ext_videos.see(target_iid)
        self._update_ext_quota_label()

    def _refresh_ext_tree_row(self, vid_name: str) -> None:
        for i, p in enumerate(self._ext_videos):
            if p.name == vid_name:
                iid = str(i)
                if self.tree_ext_videos.exists(iid):
                    self.tree_ext_videos.item(iid, values=self._format_ext_row_values(vid_name))
                break
        self._update_ext_quota_label()

    def _update_ext_quota_label(self) -> None:
        tot = len(self._ext_videos)
        if tot == 0:
            self.var_ext_quota.set("请先选择原视频文件夹（列表将显示全部视频，可逐个单独框选）")
            return
        custom_cnt = sum(1 for p in self._ext_videos if p.name in self._ext_video_configs)
        cur_name = self._ext_current_video_name or "-"
        cur_tag = "已独立记忆" if cur_name in self._ext_video_configs else "默认区域"
        self.var_ext_quota.set(f"当前：{cur_name}（{cur_tag}） | 已独立设置 {custom_cnt}/{tot} 个")

    def _load_extract_video_list(self, vdir: Path) -> None:
        self._ext_videos = sorted([p for p in vdir.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        self._ext_current_video_name = self._ext_videos[0].name if self._ext_videos else None
        self._ext_loaded_video_path = None
        self._refresh_ext_tree_all()
        if self._ext_videos:
            self.tree_ext_videos.selection_set("0")
            self._on_ext_tree_selected()
        else:
            if self._ext_cap:
                self._ext_cap.release()
                self._ext_cap = None
            self.ext_canvas.set_image(None)
            self._update_ext_quota_label()

    def _on_ext_tree_selected(self) -> None:
        sel = self.tree_ext_videos.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._ext_videos):
            return
        vid = self._ext_videos[idx]
        if self._ext_loaded_video_path == vid and self._ext_cap is not None:
            return
        self._ext_loaded_video_path = vid
        self._ext_current_video_name = vid.name
        self._apply_video_cfg_to_ui(vid.name)

        if self._ext_cap:
            self._ext_cap.release()
        self._ext_cap = cv2.VideoCapture(str(vid))
        self._ext_fps = float(self._ext_cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._ext_total_frames = max(1, int(self._ext_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        total_sec = self._ext_total_frames / self._ext_fps
        self.scl_ext_time.configure(to=max(1.0, total_sec))
        init_sec = min(total_sec, max(0.2, total_sec * 0.42))
        self.var_ext_time.set(init_sec)
        self._render_extract_frame(init_sec)

    def _step_ext_video(self, delta: int) -> None:
        if not self._ext_videos:
            return
        sel = self.tree_ext_videos.selection()
        cur_idx = int(sel[0]) if sel else 0
        new_idx = max(0, min(len(self._ext_videos) - 1, cur_idx + delta))
        iid = str(new_idx)
        self.tree_ext_videos.selection_set(iid)
        self.tree_ext_videos.see(iid)
        self._on_ext_tree_selected()

    def _on_ext_seek(self, _val=None) -> None:
        if self._ext_seek_after_id:
            self.after_cancel(self._ext_seek_after_id)
        self._ext_seek_after_id = self.after(50, lambda: self._render_extract_frame(self.var_ext_time.get()))

    def _jump_ext_ratio(self, ratio: float) -> None:
        if not self._ext_cap:
            return
        total_sec = self._ext_total_frames / max(1.0, self._ext_fps)
        sec = max(0.1, min(total_sec - 0.1, total_sec * ratio))
        self.var_ext_time.set(sec)
        self._render_extract_frame(sec)

    def _render_extract_frame(self, sec: float) -> None:
        if not self._ext_cap:
            return
        total_sec = self._ext_total_frames / max(1.0, self._ext_fps)
        self.var_ext_time_lbl.set(f"{_fmt_sec(sec)} / {_fmt_sec(total_sec)}")
        f_idx = max(0, min(self._ext_total_frames - 1, int(round(sec * self._ext_fps))))
        self._ext_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = self._ext_cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.ext_canvas.set_image(Image.fromarray(rgb))

    def _sync_ext_rois(self, persist: bool = True) -> None:
        m_roi = (
            min(self.var_m_x1.get(), self.var_m_x2.get()),
            min(self.var_m_y1.get(), self.var_m_y2.get()),
            max(self.var_m_x1.get(), self.var_m_x2.get()),
            max(self.var_m_y1.get(), self.var_m_y2.get()),
        )
        n_roi = (
            min(self.var_n_x1.get(), self.var_n_x2.get()),
            min(self.var_n_y1.get(), self.var_n_y2.get()),
            max(self.var_n_x1.get(), self.var_n_x2.get()),
            max(self.var_n_y1.get(), self.var_n_y2.get()),
        )
        self.ext_canvas.set_rois(m_roi, n_roi, show_note_roi=self.var_extract_notes.get())

        if persist and (not self._ext_loading_ui):
            if self._ext_current_video_name:
                self._ext_video_configs[self._ext_current_video_name] = self._read_ui_ext_cfg_dict(source="manual")
                self._refresh_ext_tree_row(self._ext_current_video_name)
                self._save_video_roi_store()
            else:
                self._ext_default_cfg = self._read_ui_ext_cfg_dict(source="default")

    def _on_canvas_roi_dragged(self, which: str, rect: tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = rect
        self._ext_loading_ui = True
        try:
            if which == "main":
                self.var_m_x1.set(x1)
                self.var_m_y1.set(y1)
                self.var_m_x2.set(x2)
                self.var_m_y2.set(y2)
            else:
                self.var_n_x1.set(x1)
                self.var_n_y1.set(y1)
                self.var_n_x2.set(x2)
                self.var_n_y2.set(y2)
        finally:
            self._ext_loading_ui = False
        self._sync_ext_rois(persist=True)

    def _widen_current_main_roi(self) -> None:
        self._ext_loading_ui = True
        try:
            self.var_m_x1.set(0.06)
            self.var_m_x2.set(0.94)
        finally:
            self._ext_loading_ui = False
        self._sync_ext_rois(persist=True)

    def _reset_current_video_roi(self) -> None:
        if not self._ext_current_video_name:
            return
        self._ext_video_configs.pop(self._ext_current_video_name, None)
        self._apply_video_cfg_to_ui(self._ext_current_video_name)
        self._refresh_ext_tree_row(self._ext_current_video_name)
        self._save_video_roi_store()

    def _apply_preset(self) -> None:
        name = self.var_preset_name.get()
        p = self._presets.get(name)
        if not p:
            return
        self._ext_loading_ui = True
        try:
            mx1, my1, mx2, my2 = p["main_roi"]
            self.var_m_x1.set(mx1)
            self.var_m_y1.set(my1)
            self.var_m_x2.set(mx2)
            self.var_m_y2.set(my2)
            self.var_main_white_only.set(bool(p.get("main_white_only", True)))
            self.var_extract_notes.set(bool(p.get("extract_notes", True)))
            nx1, ny1, nx2, ny2 = p.get("note_roi", [0.0, 0.0, 1.0, 1.0])
            self.var_n_x1.set(nx1)
            self.var_n_y1.set(ny1)
            self.var_n_x2.set(nx2)
            self.var_n_y2.set(ny2)
            self.var_note_min_h.set(float(p.get("note_min_height_ratio", 0.038)))
        finally:
            self._ext_loading_ui = False
        self._sync_ext_rois(persist=True)

    def _save_current_preset(self) -> None:
        name = simpledialog.askstring("保存选区预设", "请给当前红框/蓝框预设起一个名字（如：老王剪辑竖屏）：", parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        self._presets[name] = {
            "main_roi": [round(self.var_m_x1.get(), 3), round(self.var_m_y1.get(), 3), round(self.var_m_x2.get(), 3), round(self.var_m_y2.get(), 3)],
            "main_white_only": bool(self.var_main_white_only.get()),
            "extract_notes": bool(self.var_extract_notes.get()),
            "note_roi": [round(self.var_n_x1.get(), 3), round(self.var_n_y1.get(), 3), round(self.var_n_x2.get(), 3), round(self.var_n_y2.get(), 3)],
            "note_min_height_ratio": round(self.var_note_min_h.get(), 3),
        }
        save_presets(self._presets)
        self.cmb_presets["values"] = list(self._presets.keys())
        self.var_preset_name.set(name)
        messagebox.showinfo("成功", f"预设「{name}」已保存，下次打开自动加载！")

    def _apply_current_roi_batch(self, only_unset: bool) -> None:
        if not self._ext_videos:
            messagebox.showinfo("提示", "请先选择包含视频的文件夹。")
            return
        cur_cfg = self._read_ui_ext_cfg_dict(source="manual")
        if only_unset:
            self._ext_default_cfg = dict(cur_cfg, source="default")
            applied = 0
            for p in self._ext_videos:
                if p.name not in self._ext_video_configs:
                    self._ext_video_configs[p.name] = dict(cur_cfg)
                    applied += 1
            msg = f"已将当前选区套用到 {applied} 个未定制视频（已定制的视频保持不变）。"
        else:
            if len(self._ext_videos) > 1 and not messagebox.askyesno("确认套用全部", f"确定要用当前选区覆盖全部 {len(self._ext_videos)} 个视频的设置吗？"):
                return
            self._ext_default_cfg = dict(cur_cfg, source="default")
            for p in self._ext_videos:
                self._ext_video_configs[p.name] = dict(cur_cfg)
            msg = f"已将当前选区套用到全部 {len(self._ext_videos)} 个视频。"

        self._refresh_ext_tree_all()
        self._save_video_roi_store()
        self._append_log(self.txt_ext_log, f"[批量选区] {msg}")
        messagebox.showinfo("完成", msg)

    def _start_auto_detect_rois(self) -> None:
        if self._is_busy:
            messagebox.showwarning("提示", "已有任务正在运行中，请先等待完成或点击停止。")
            return
        if not self._ext_videos:
            messagebox.showerror("错误", "请先选择包含视频文件的原视频文件夹！")
            return

        self._set_busy(True, "detect")
        self._cancel_event.clear()
        self.var_ext_cur_prog.set(0.0)
        self.var_ext_tot_prog.set(0.0)
        accel = "auto" if "DirectML" in self.var_ocr_accel.get() else "cpu"
        ocr_prof = self._get_selected_ocr_profile()
        white_only = bool(self.var_main_white_only.get())
        base_cfg = self._read_ui_ext_cfg_dict(source="auto")
        videos_snapshot = list(self._ext_videos)

        def _worker():
            def _prog(cur_idx: int, total: int, msg: str):
                pct = (cur_idx / max(1, total)) * 100.0

                def _upd():
                    self.var_ext_cur_prog.set(100.0)
                    self.var_ext_cur_msg.set(f"当前视频：{msg}")
                    self.var_ext_tot_prog.set(pct)
                    self.var_ext_tot_msg.set(f"自动定位进度：第 {cur_idx} / {total} 个视频 ({pct:.1f}%)")

                self.after(0, _upd)

            try:
                res = auto_detect_batch_rois(
                    videos_snapshot,
                    accel_mode=accel,
                    main_white_only=white_only,
                    fallback_roi=tuple(base_cfg["main_roi"]),
                    progress_cb=_prog,
                    cancel_event=self._cancel_event,
                    ocr_profile=ocr_prof,
                )
            except ExtractCancelled:
                self.after(0, lambda: (self._set_busy(False, "detect"), self._append_log(self.txt_ext_log, "[已取消] 自动识别红框被用户中止")))
                return

            def _done():
                self._set_busy(False, "detect")
                hit_names: list[str] = []
                miss_names: list[str] = []
                for vname, (roi, found, hits) in res.items():
                    if found:
                        item_cfg = dict(base_cfg)
                        item_cfg["main_roi"] = [round(v, 3) for v in roi]
                        item_cfg["source"] = "auto"
                        self._ext_video_configs[vname] = item_cfg
                        hit_names.append(vname)
                        self._append_log(self.txt_ext_log, f"[自动定位命中] {vname}: Y={roi[1]*100:.1f}%~{roi[3]*100:.1f}%, X={roi[0]*100:.1f}%~{roi[2]*100:.1f}% (命中{hits}处)")
                    else:
                        miss_names.append(vname)
                        self._append_log(self.txt_ext_log, f"[自动定位未命中] {vname}: 请在列表点击该视频并手动拖拽红框")

                self._save_video_roi_store()
                self._refresh_ext_tree_all()
                if self._ext_current_video_name:
                    self._apply_video_cfg_to_ui(self._ext_current_video_name)

                summary = f"自动定位完成：成功识别 {len(hit_names)} / {len(videos_snapshot)} 个视频的红框位置！"
                if miss_names:
                    summary += "\n\n以下视频未抽中字幕帧，请在左上清单点选后手动框一下：\n" + "\n".join(miss_names)
                messagebox.showinfo("自动定位完成", summary)

            self.after(0, _done)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _start_extract(self) -> None:
        if self._is_busy:
            messagebox.showwarning("提示", "已有任务正在运行中，请先等待完成或点击停止。")
            return
        vdir = Path(self.var_video_dir.get().strip())
        zh_dir = Path(self.var_zh_dir.get().strip())
        en_dir = Path(self.var_en_dir.get().strip())
        if not vdir.exists() or not self._ext_videos:
            messagebox.showerror("错误", "请先选择包含视频文件的原视频文件夹！")
            return

        self._sync_ext_rois(persist=True)
        default_cfg = self._dict_to_extract_config(self._ext_default_cfg)
        per_video_cfgs = {p.name: self._dict_to_extract_config(self._ext_video_configs[p.name]) for p in self._ext_videos if p.name in self._ext_video_configs}

        self._set_busy(True, "extract")
        self._cancel_event.clear()
        self.txt_ext_log.delete("1.0", tk.END)
        self.var_ext_cur_prog.set(0.0)
        self.var_ext_tot_prog.set(0.0)
        prompt_path = zh_dir / "AGENT_PROMPT.txt"
        videos_snapshot = list(self._ext_videos)
        auto_unset = bool(self.var_auto_detect_unset.get())
        default_dict_snapshot = dict(self._ext_default_cfg)

        def _worker():
            def log(msg: str):
                self.after(0, lambda m=msg: self._append_log(self.txt_ext_log, m))

            def prog(cur_idx: int, total: int, cur_pct: float, msg: str):
                overall_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0

                def _upd():
                    self.var_ext_cur_prog.set(cur_pct)
                    self.var_ext_cur_msg.set(f"当前视频：{msg} ({cur_pct:.1f}%)")
                    self.var_ext_tot_prog.set(min(100.0, overall_pct))
                    self.var_ext_tot_msg.set(f"总进度：第 {cur_idx} / {total} 个视频（总计 {overall_pct:.1f}%）")

                self.after(0, _upd)

            def _on_auto_roi(vname: str, det_roi: tuple[float, float, float, float]):
                def _apply():
                    item_cfg = dict(default_dict_snapshot)
                    item_cfg["main_roi"] = [round(v, 3) for v in det_roi]
                    item_cfg["source"] = "auto"
                    self._ext_video_configs[vname] = item_cfg
                    self._refresh_ext_tree_row(vname)
                    self._save_video_roi_store()
                    if self._ext_current_video_name == vname:
                        self._apply_video_cfg_to_ui(vname)

                self.after(0, _apply)

            res = extract_batch(
                videos_snapshot,
                zh_dir,
                en_dir,
                default_cfg,
                self._glossary_text,
                prompt_path,
                progress_cb=prog,
                log_cb=log,
                cancel_event=self._cancel_event,
                per_video_configs=per_video_cfgs,
                auto_detect_unset=auto_unset,
                on_auto_roi_cb=_on_auto_roi,
            )

            def _done():
                self._set_busy(False, "extract")
                self._last_prompt_text = res["prompt_text"]
                self._burn_zh_bands.clear()
                self._refresh_translate_review_list()
                self._refresh_burn_matches()
                if not res["cancelled"]:
                    self._show_prompt_dialog(res["prompt_text"])

            self.after(0, _done)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _edit_glossary_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("编辑方言专属术语词典（自动写入内置 AI 与外部 Agent 提示词）")
        win.geometry("620x460")
        win.transient(self)

        ttk.Label(win, text="每行一条词典规则，例如：- 赶大集 = village fair（保存后立即生效）：", padding=(10, 8)).pack(anchor="w")
        txt = tk.Text(win, wrap=tk.WORD, font=("Microsoft YaHei UI", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        txt.insert("1.0", self._glossary_text)

        def _save():
            self._glossary_text = txt.get("1.0", tk.END).strip()
            save_glossary(self._glossary_text)
            win.destroy()
            messagebox.showinfo("已保存", "方言词典已更新！下次翻译或生成提示词将自动带上新词典。")

        btn_f = ttk.Frame(win, padding=(10, 0, 10, 10))
        btn_f.pack(fill=tk.X)
        ttk.Button(btn_f, text="保存词典", command=_save).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_f, text="取消", command=win.destroy).pack(side=tk.RIGHT)

    def _show_prompt_dialog(self, prompt_override: Optional[str] = None) -> None:
        zh_str = self.var_zh_dir.get().strip()
        en_str = self.var_en_dir.get().strip()
        if prompt_override:
            content = prompt_override
        elif self._last_prompt_text:
            content = self._last_prompt_text
        elif zh_str and en_str:
            zh_dir = Path(zh_str)
            en_dir = Path(en_str)
            srts = sorted([p.name for p in zh_dir.glob("*.srt")]) if zh_dir.exists() else []
            content = generate_agent_prompt(zh_dir, en_dir, srts, self._glossary_text, zh_dir / "AGENT_PROMPT.txt")
        else:
            messagebox.showinfo("提示", "请先选择文件夹并提取一次中文 SRT，再生成提示词。")
            return

        win = tk.Toplevel(self)
        win.title("给外部 Agent 的批量修字 + 翻译提示词（可复制发给 WorkBuddy）")
        win.geometry("760x620")
        win.transient(self)

        ttk.Label(
            win,
            text="你可以直接点「第 2 步」或「一键一条龙」由内置模型自动翻译，也可以复制下方提示词发给外部 Agent：",
            padding=(12, 8),
            foreground="#0b6bcb",
        ).pack(anchor="w")

        txt = tk.Text(win, wrap=tk.WORD, font=("Microsoft YaHei UI", 10), bg="#f8fafc")
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        txt.insert("1.0", content)

        status_var = tk.StringVar(value="")
        bot = ttk.Frame(win, padding=(12, 0, 12, 12))
        bot.pack(fill=tk.X)

        def _copy():
            self.clipboard_clear()
            self.clipboard_append(txt.get("1.0", tk.END).strip())
            self.update()
            status_var.set("✅ 已复制到剪贴板！")

        ttk.Button(bot, text="📋 一键复制到剪贴板", command=_copy).pack(side=tk.LEFT)
        ttk.Label(bot, textvariable=status_var, foreground="#15803d", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Button(bot, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

    # ==================== Tab 2：内置 AI 自动翻译与校对逻辑 ====================

    def _toggle_api_key_mask(self) -> None:
        self._api_key_shown = not self._api_key_shown
        self._api_key_entry.configure(show="" if self._api_key_shown else "*")
        self._btn_toggle_key.configure(text="隐藏" if self._api_key_shown else "显示")

    def _read_trans_cfg_from_ui(self) -> TranslatorConfig:
        try:
            bsize = max(4, min(25, int(self.var_trans_batch_size.get())))
        except Exception:
            bsize = 12
        return TranslatorConfig(
            provider=self.var_trans_provider.get().strip() or "ollama",
            ollama_url=self.var_ollama_url.get().strip() or "http://127.0.0.1:11434",
            ollama_model=self.var_ollama_model.get().strip() or "hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M",
            api_base_url=self.var_api_base.get().strip() or "https://api.deepseek.com/v1",
            api_key=self.var_api_key.get().strip(),
            api_model=self.var_api_model.get().strip() or "deepseek-chat",
            batch_size=bsize,
            temperature=0.1,
            skip_existing=bool(self.var_trans_skip.get()),
            unload_after_done=bool(self.var_trans_unload.get()),
            auto_detect_dialect=bool(self.var_trans_dialect.get()),
        )

    def _save_trans_cfg_from_ui(self, show_tip: bool = False) -> TranslatorConfig:
        cfg = self._read_trans_cfg_from_ui()
        self._trans_cfg_dict = asdict(cfg)
        save_translator_config(self._trans_cfg_dict)
        if show_tip:
            messagebox.showinfo("已保存", "AI 翻译引擎配置已保存到本地 config/translator.json。")
        return cfg

    def _refresh_ollama_models_ui(self) -> None:
        url = self.var_ollama_url.get().strip() or "http://127.0.0.1:11434"

        def _worker():
            try:
                models = list_ollama_models(url)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Ollama 连接失败", f"无法连接 {url}：\n{exc}\n请确认 Ollama 已运行。"))
                return

            def _done():
                self.cmb_ollama_model["values"] = models
                if models and (self.var_ollama_model.get().strip() not in models):
                    self.var_ollama_model.set(models[0])
                self._append_log(self.txt_trans_log, f"[Ollama] 已刷新 {len(models)} 个可用模型")

            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _test_translator_async(self) -> None:
        cfg = self._save_trans_cfg_from_ui(show_tip=False)
        self.btn_test_trans.configure(state=tk.DISABLED)
        self._append_log(self.txt_trans_log, f"[连通测试] 正在尝试连接 {cfg.provider} 并试翻 1 句农村方言...")

        def _worker():
            ok, msg = test_translator_connection(cfg)

            def _done():
                self.btn_test_trans.configure(state=tk.NORMAL)
                self._append_log(self.txt_trans_log, f"[测试结果] {msg}")
                if ok:
                    messagebox.showinfo("测试成功", msg)
                else:
                    messagebox.showerror("测试失败", msg)

            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_translate_review_list(self) -> None:
        if not hasattr(self, "cmb_trans_review_file"):
            return
        zh_s = self.var_zh_dir.get().strip()
        en_s = self.var_en_dir.get().strip()
        names: list[str] = []
        if zh_s and Path(zh_s).exists():
            names = [p.name for p in sorted(Path(zh_s).glob("*.srt"))]
        elif en_s and Path(en_s).exists():
            names = [p.name for p in sorted(Path(en_s).glob("*.srt"))]

        self._trans_review_files = names
        self.cmb_trans_review_file["values"] = names
        if names:
            if self.var_trans_review_file.get() not in names:
                self.var_trans_review_file.set(names[0])
            self._load_review_file_rows()
        else:
            self.var_trans_review_file.set("")
            for row in self.tree_trans_lines.get_children():
                self.tree_trans_lines.delete(row)

    def _load_review_file_rows(self) -> None:
        for row in self.tree_trans_lines.get_children():
            self.tree_trans_lines.delete(row)
        fname = self.var_trans_review_file.get().strip()
        if not fname:
            return
        zh_p = Path(self.var_zh_dir.get().strip()) / fname if self.var_zh_dir.get().strip() else None
        en_p = Path(self.var_en_dir.get().strip()) / fname if self.var_en_dir.get().strip() else None
        zh_b = parse_srt_blocks(zh_p) if (zh_p and zh_p.exists()) else []
        en_b = parse_srt_blocks(en_p) if (en_p and en_p.exists()) else []
        self._trans_cur_zh_blocks = zh_b
        self._trans_cur_en_blocks = en_b

        en_map = {b.index: b for b in en_b}
        base_list = zh_b if zh_b else en_b
        for i, blk in enumerate(base_list):
            eb = en_map.get(blk.index)
            en_txt = eb.text.replace("\n", " ") if eb else "(尚未翻译)"
            zh_txt = blk.text.replace("\n", " ") if zh_b else "-"
            self.tree_trans_lines.insert(
                "", tk.END, iid=str(i),
                values=(blk.index, f"{blk.start_str} -> {blk.end_str}", zh_txt, en_txt),
            )

    def _edit_selected_trans_line(self) -> None:
        sel = self.tree_trans_lines.selection()
        fname = self.var_trans_review_file.get().strip()
        en_dir_s = self.var_en_dir.get().strip()
        if not sel or not fname or not en_dir_s:
            return
        row_idx = int(sel[0])
        base_list = self._trans_cur_zh_blocks if self._trans_cur_zh_blocks else self._trans_cur_en_blocks
        if row_idx < 0 or row_idx >= len(base_list):
            return

        target_blk = base_list[row_idx]
        en_map = {b.index: b for b in self._trans_cur_en_blocks}
        cur_en = en_map[target_blk.index].text if target_blk.index in en_map else ""

        win = tk.Toplevel(self)
        win.title(f"修改第 {target_blk.index} 条译文 - {fname}")
        win.geometry("620x280")
        win.transient(self)

        ttk.Label(win, text=f"时间轴：{target_blk.start_str} --> {target_blk.end_str}", padding=(12, 8, 12, 2), foreground="#0b6bcb").pack(anchor="w")
        ttk.Label(win, text=f"原中文：{target_blk.text}", padding=(12, 2, 12, 6), wraplength=580).pack(anchor="w")
        ttk.Label(win, text="英文译文：", padding=(12, 2, 12, 2)).pack(anchor="w")

        txt = tk.Text(win, height=4, wrap=tk.WORD, font=("Microsoft YaHei UI", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        txt.insert("1.0", cur_en)
        txt.focus_set()

        def _save_line():
            new_en = txt.get("1.0", tk.END).strip()
            if not new_en:
                messagebox.showwarning("提示", "英文译文不能为空。", parent=win)
                return
            en_path = Path(en_dir_s) / fname
            if self._trans_cur_en_blocks:
                updated = []
                hit = False
                for eb in self._trans_cur_en_blocks:
                    if eb.index == target_blk.index:
                        updated.append(SrtBlock(eb.index, eb.start_str, eb.end_str, new_en))
                        hit = True
                    else:
                        updated.append(eb)
                if not hit:
                    updated.append(SrtBlock(target_blk.index, target_blk.start_str, target_blk.end_str, new_en))
                    updated.sort(key=lambda b: b.index)
            else:
                updated = [
                    SrtBlock(b.index, b.start_str, b.end_str, new_en if b.index == target_blk.index else b.text)
                    for b in base_list
                ]
            write_srt_blocks(updated, en_path)
            win.destroy()
            self._load_review_file_rows()
            self._refresh_burn_matches()

        bf = ttk.Frame(win, padding=(12, 0, 12, 12))
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="保存修改", command=_save_line).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bf, text="取消", command=win.destroy).pack(side=tk.RIGHT)

    def _start_translate(self) -> None:
        if self._is_busy:
            messagebox.showwarning("提示", "已有任务正在运行中，请先等待完成或点击停止。")
            return
        zh_dir = Path(self.var_zh_dir.get().strip())
        en_dir = Path(self.var_en_dir.get().strip())
        if not zh_dir.exists() or not list(zh_dir.glob("*.srt")):
            messagebox.showerror("错误", "中文 SRT 目录中还没有可翻译的 .srt 文件，请先执行第 1 步提取！")
            return

        cfg = self._save_trans_cfg_from_ui(show_tip=False)
        self._set_busy(True, "translate")
        self._cancel_event.clear()
        self.txt_trans_log.delete("1.0", tk.END)
        self.var_trans_cur_prog.set(0.0)
        self.var_trans_tot_prog.set(0.0)

        def _worker():
            def log(msg: str):
                self.after(0, lambda m=msg: self._append_log(self.txt_trans_log, m))

            def prog(cur_idx: int, total: int, cur_pct: float, msg: str):
                overall_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0

                def _upd():
                    self.var_trans_cur_prog.set(cur_pct)
                    self.var_trans_cur_msg.set(f"当前文件：{msg} ({cur_pct:.1f}%)")
                    self.var_trans_tot_prog.set(min(100.0, overall_pct))
                    self.var_trans_tot_msg.set(f"总进度：第 {cur_idx} / {total} 个字幕（总计 {overall_pct:.1f}%）")

                self.after(0, _upd)

            res = translate_batch(
                zh_dir,
                en_dir,
                cfg,
                self._glossary_text,
                progress_cb=prog,
                log_cb=log,
                cancel_event=self._cancel_event,
            )

            def _done():
                self._set_busy(False, "translate")
                self._refresh_translate_review_list()
                self._refresh_burn_matches()
                if not res["cancelled"]:
                    msg = f"AI 自动翻译结束：完成 {res['done']} 个，跳过 {res['skipped']} 个，失败 {res['failed']} 个。\n可在左侧直接双击任意句微调，或切换到「第 3 步」预览烧录。"
                    messagebox.showinfo("翻译完成", msg)

            self.after(0, _done)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    # ==================== 全自动一条龙调度 ====================

    def _start_full_pipeline(self) -> None:
        if self._is_busy:
            messagebox.showwarning("提示", "已有任务正在运行中，请先等待完成或点击停止。")
            return
        vdir_s = self.var_video_dir.get().strip()
        if not vdir_s or not Path(vdir_s).exists() or not self._ext_videos:
            messagebox.showerror("错误", "请先点击「选择原视频文件夹」加载视频！")
            return

        vdir = Path(vdir_s)
        zh_dir = Path(self.var_zh_dir.get().strip() or (vdir / "zh_srt"))
        en_dir = Path(self.var_en_dir.get().strip() or (vdir / "en_srt"))
        out_dir = Path(self.var_out_dir.get().strip() or (vdir / "output_bilingual"))
        self.var_zh_dir.set(str(zh_dir))
        self.var_en_dir.set(str(en_dir))
        self.var_out_dir.set(str(out_dir))

        self._sync_ext_rois(persist=True)
        default_ext_cfg = self._dict_to_extract_config(self._ext_default_cfg)
        per_video_ext_cfgs = {p.name: self._dict_to_extract_config(self._ext_video_configs[p.name]) for p in self._ext_videos if p.name in self._ext_video_configs}
        videos_snapshot = list(self._ext_videos)
        auto_unset = bool(self.var_auto_detect_unset.get())
        default_dict_snapshot = dict(self._ext_default_cfg)

        trans_cfg = self._save_trans_cfg_from_ui(show_tip=False)
        pause_after_trans = bool(self.var_pipeline_pause_review.get())

        enc_idx = self._enc_labels.index(self.var_enc_label.get()) if self.var_enc_label.get() in self._enc_labels else 0
        enc_code = self._enc_codes[enc_idx]
        dedup_opts = self._get_dedup_options()
        default_burn_style = self._style_dict_to_cfg(self._burn_default_style)
        per_video_burn_styles = {name: self._style_dict_to_cfg(st_dict) for name, st_dict in self._burn_video_styles.items()}
        burn_skip = bool(self.var_burn_skip.get())

        self._set_busy(True, "pipeline")
        self._cancel_event.clear()
        self.txt_ext_log.delete("1.0", tk.END)
        self.txt_trans_log.delete("1.0", tk.END)
        self.txt_burn_log.delete("1.0", tk.END)

        def _worker():
            def _set_pstat(txt: str):
                self.after(0, lambda t=txt: self.var_pipeline_status.set(t))

            # ---- 阶段 1/3：PP-OCRv5 双轨提取 ----
            _set_pstat("⏳ [1/3] 正在 PP-OCRv5 提取中文字幕与花字...")
            self.after(0, lambda: self.notebook.select(self.tab_extract))

            def _log1(m: str):
                self.after(0, lambda msg=m: self._append_log(self.txt_ext_log, msg))

            def _prog1(cur_idx: int, total: int, cur_pct: float, msg: str):
                tot_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0
                self.after(0, lambda: (
                    self.var_ext_cur_prog.set(cur_pct),
                    self.var_ext_cur_msg.set(f"当前视频：{msg} ({cur_pct:.1f}%)"),
                    self.var_ext_tot_prog.set(min(100.0, tot_pct)),
                    self.var_ext_tot_msg.set(f"总进度：第 {cur_idx}/{total} 个 ({tot_pct:.1f}%)"),
                ))

            def _on_auto_roi(vname: str, det_roi: tuple[float, float, float, float]):
                def _apply():
                    item_cfg = dict(default_dict_snapshot)
                    item_cfg["main_roi"] = [round(v, 3) for v in det_roi]
                    item_cfg["source"] = "auto"
                    self._ext_video_configs[vname] = item_cfg
                    self._refresh_ext_tree_row(vname)
                    self._save_video_roi_store()

                self.after(0, _apply)

            ext_res = extract_batch(
                videos_snapshot,
                zh_dir,
                en_dir,
                default_ext_cfg,
                self._glossary_text,
                zh_dir / "AGENT_PROMPT.txt",
                progress_cb=_prog1,
                log_cb=_log1,
                cancel_event=self._cancel_event,
                per_video_configs=per_video_ext_cfgs,
                auto_detect_unset=auto_unset,
                on_auto_roi_cb=_on_auto_roi,
            )
            gc.collect()
            self._last_prompt_text = ext_res.get("prompt_text", "")
            if ext_res["cancelled"] or self._cancel_event.is_set():
                self.after(0, lambda: (self._set_busy(False, "pipeline"), self.var_pipeline_status.set("⏹ 一条龙已在提取阶段中止")))
                return

            # ---- 阶段 2/3：内置 AI 自动翻译 ----
            _set_pstat(f"⏳ [2/3] 正在用 {trans_cfg.ollama_model if trans_cfg.provider == 'ollama' else trans_cfg.api_model} 自动翻译字幕...")
            self.after(0, lambda: (self.notebook.select(self.tab_translate), self._refresh_translate_review_list()))

            def _log2(m: str):
                self.after(0, lambda msg=m: self._append_log(self.txt_trans_log, msg))

            def _prog2(cur_idx: int, total: int, cur_pct: float, msg: str):
                tot_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0
                self.after(0, lambda: (
                    self.var_trans_cur_prog.set(cur_pct),
                    self.var_trans_cur_msg.set(f"当前文件：{msg} ({cur_pct:.1f}%)"),
                    self.var_trans_tot_prog.set(min(100.0, tot_pct)),
                    self.var_trans_tot_msg.set(f"总进度：第 {cur_idx}/{total} 个 ({tot_pct:.1f}%)"),
                ))

            tr_res = translate_batch(
                zh_dir,
                en_dir,
                trans_cfg,
                self._glossary_text,
                progress_cb=_prog2,
                log_cb=_log2,
                cancel_event=self._cancel_event,
            )
            if trans_cfg.provider == "ollama":
                unload_ollama_model(trans_cfg.ollama_url, trans_cfg.ollama_model)
            gc.collect()

            if tr_res["cancelled"] or self._cancel_event.is_set():
                self.after(0, lambda: (self._set_busy(False, "pipeline"), self.var_pipeline_status.set("⏹ 一条龙已在翻译阶段中止")))
                return

            self.after(0, lambda: (self._refresh_translate_review_list(), self._refresh_burn_matches()))

            if pause_after_trans:
                self.after(0, lambda: (
                    self._set_busy(False, "pipeline"),
                    self.var_pipeline_status.set(f"⏸ 翻译完成（成功{tr_res['done']} / 跳过{tr_res['skipped']} / 失败{tr_res['failed']}），请抽查后点烧录"),
                    messagebox.showinfo("一条龙暂停抽查", f"提取与 AI 翻译已全部完成！\n成功 {tr_res['done']} 个，跳过 {tr_res['skipped']} 个，失败 {tr_res['failed']} 个。\n\n请在「第 2 步」抽查译文或直接去「第 3 步」点烧录出片。"),
                ))
                return

            # ---- 阶段 3/3：全自动防重叠烧录出片 ----
            ready_items = [m for m in scan_burn_matches(vdir, zh_dir, en_dir) if m["ready"]]
            if not ready_items:
                self.after(0, lambda: (
                    self._set_busy(False, "pipeline"),
                    self.var_pipeline_status.set("❌ 未找到就绪的英文 SRT，请检查翻译日志"),
                    messagebox.showerror("无法进入烧录阶段", "翻译阶段未能产出就绪的英文 SRT，请查看第 2 页日志排查连通性或模型状态。"),
                ))
                return

            _set_pstat(f"⏳ [3/3] 正在全自动防重叠排版与硬件烧录 ({len(ready_items)} 个视频)...")
            self.after(0, lambda: self.notebook.select(self.tab_burn))

            def _log3(m: str):
                self.after(0, lambda msg=m: self._append_log(self.txt_burn_log, msg))

            def _prog3(cur_idx: int, total: int, cur_pct: float, msg: str):
                tot_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0
                self.after(0, lambda: (
                    self.var_burn_cur_prog.set(cur_pct),
                    self.var_burn_cur_msg.set(f"当前视频：{msg} ({cur_pct:.1f}%)"),
                    self.var_burn_tot_prog.set(min(100.0, tot_pct)),
                    self.var_burn_tot_msg.set(f"总进度：第 {cur_idx}/{total} 个 ({tot_pct:.1f}%)"),
                ))

            b_res = burn_batch(
                ready_items,
                out_dir,
                mode=enc_code,
                dedup_options=dedup_opts,
                style=default_burn_style,
                test_sec=0,
                skip_existing=burn_skip,
                progress_cb=_prog3,
                log_cb=_log3,
                cancel_event=self._cancel_event,
                per_video_styles=per_video_burn_styles,
            )

            def _all_done():
                self._set_busy(False, "pipeline")
                self._refresh_burn_matches()
                if b_res["cancelled"]:
                    self.var_pipeline_status.set("⏹ 一条龙已在烧录阶段中止")
                else:
                    summary = (
                        f"🎉 一条龙全部完成！\n"
                        f"1) 提取：新处理 {ext_res['processed']} 个，跳过 {ext_res['skipped']} 个\n"
                        f"2) 翻译：新翻译 {tr_res['done']} 份，跳过 {tr_res['skipped']} 份，失败 {tr_res['failed']} 份\n"
                        f"3) 烧录：成功出片 {b_res['done']} 个，跳过 {b_res['skipped']} 个，失败 {b_res['failed']} 个\n\n"
                        f"成品输出目录：{out_dir}"
                    )
                    self.var_pipeline_status.set(f"✅ 一条龙完成：成功出片 {b_res['done']} 个（跳过 {b_res['skipped']} 个）")
                    messagebox.showinfo("一条龙完成", summary)

            self.after(0, _all_done)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    # ==================== Tab 3 烧录样式、(X, Y) 拖拽与压制逻辑 ====================

    def _get_burn_style_store_path(self) -> Optional[Path]:
        zh_s = self.var_zh_dir.get().strip()
        return (Path(zh_s) / "_video_burn_styles.json") if zh_s else None

    def _normalize_burn_style(self, raw: Optional[dict]) -> dict:
        base = asdict(BurnStyleConfig())
        if isinstance(raw, dict):
            for k in base:
                if k in raw:
                    base[k] = raw[k]
            if "pos_y_ratio" not in raw and "margin_v_ratio" in raw:
                base["pos_y_ratio"] = round(max(0.08, min(0.985, 1.0 - float(raw["margin_v_ratio"]))), 4)
            if not isinstance(base.get("note_pos_overrides"), dict):
                base["note_pos_overrides"] = {}
        return base

    @staticmethod
    def _style_dict_to_cfg(st_dict: dict) -> BurnStyleConfig:
        valid = asdict(BurnStyleConfig())
        kwargs = {k: st_dict.get(k, valid[k]) for k in valid}
        if not isinstance(kwargs.get("note_pos_overrides"), dict):
            kwargs["note_pos_overrides"] = {}
        return BurnStyleConfig(**kwargs)

    def _load_burn_style_store(self) -> None:
        self._burn_default_style = asdict(BurnStyleConfig())
        self._burn_video_styles = {}
        p = self._get_burn_style_store_path()
        if not p or not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload.get("default"), dict):
                self._burn_default_style = self._normalize_burn_style(payload["default"])
            vmap = payload.get("videos", {})
            if isinstance(vmap, dict):
                for name, st in vmap.items():
                    self._burn_video_styles[name] = self._normalize_burn_style(st)
        except Exception:
            pass

    def _save_burn_style_store(self) -> None:
        p = self._get_burn_style_store_path()
        if not p:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"default": self._burn_default_style, "videos": self._burn_video_styles}
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _apply_burn_style_dict_to_ui(self, sty_dict: dict) -> None:
        self._burn_loading_ui = True
        try:
            mode_code = str(sty_dict.get("auto_place_mode", "auto"))
            m_idx = self._auto_place_codes.index(mode_code) if mode_code in self._auto_place_codes else 0
            self.var_auto_place_label.set(self._auto_place_labels[m_idx])
            self.var_pos_x.set(float(sty_dict.get("pos_x_ratio", 0.50)))
            pos_y = float(sty_dict.get("pos_y_ratio", 1.0 - float(sty_dict.get("margin_v_ratio", 0.052))))
            self.var_pos_y.set(max(0.08, min(0.985, pos_y)))
            self.var_auto_gap.set(float(sty_dict.get("auto_gap_ratio", 0.012)))
            self.var_font_size.set(float(sty_dict.get("font_size_ratio", 0.036)))
            self.var_color_name.set(str(sty_dict.get("font_color_name", "暖白")))
            self.var_outline_w.set(float(sty_dict.get("outline_width", 3.0)))
            self.var_semi_box.set(bool(sty_dict.get("use_semi_bg_box", False)))
            self.var_note_mult.set(float(sty_dict.get("note_scale_mult", 1.0)))
            self.var_note_gap.set(int(sty_dict.get("note_gap_px", 8)))
            self.var_top_banner.set(str(sty_dict.get("top_banner_text", "90s Rural China · Official")))
            self.var_zoom_comp.set(bool(sty_dict.get("auto_zoom_margin_comp", True)))
            self._cur_note_pos_overrides = {
                str(k): [float(v[0]), float(v[1])]
                for k, v in (sty_dict.get("note_pos_overrides") or {}).items()
                if isinstance(v, (list, tuple)) and len(v) >= 2
            }
        finally:
            self._burn_loading_ui = False

    def _refresh_burn_matches(self) -> None:
        vdir_s = self.var_video_dir.get().strip()
        zh_s = self.var_zh_dir.get().strip()
        en_s = self.var_en_dir.get().strip()
        for row in self.tree_matches.get_children():
            self.tree_matches.delete(row)
        if not vdir_s:
            return

        self._burn_matches = scan_burn_matches(Path(vdir_s), Path(zh_s), Path(en_s))
        for i, m in enumerate(self._burn_matches):
            vname = m["video_path"].name
            disp_name = f"★ {vname}" if vname in self._burn_video_styles else vname
            self.tree_matches.insert(
                "", tk.END, iid=str(i),
                values=(disp_name, m["resolution"], m["main_str"], m["notes_str"], m["status"]),
            )
        if self._burn_matches:
            self.tree_matches.selection_set("0")
            self._on_burn_row_selected()

    def _refresh_burn_row_marker(self, vid_name: str) -> None:
        for i, m in enumerate(self._burn_matches):
            if m["video_path"].name == vid_name:
                iid = str(i)
                if self.tree_matches.exists(iid):
                    disp = f"★ {vid_name}" if vid_name in self._burn_video_styles else vid_name
                    self.tree_matches.item(iid, values=(disp, m["resolution"], m["main_str"], m["notes_str"], m["status"]))
                break

    def _on_burn_row_selected(self) -> None:
        sel = self.tree_matches.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._burn_matches):
            return
        m = self._burn_matches[idx]
        self._burn_selected_match = m
        vname = m["video_path"].name

        if vname in self._burn_video_styles:
            self.var_burn_per_video.set(True)
            self._apply_burn_style_dict_to_ui(self._burn_video_styles[vname])
        else:
            self.var_burn_per_video.set(False)
            self._apply_burn_style_dict_to_ui(self._burn_default_style)

        self._burn_cur_en_srt = parse_srt(m["en_main_path"]) if m["en_main_path"] else []
        self._burn_cur_notes_srt = parse_srt(m["en_notes_path"]) if m["en_notes_path"] else []
        self._burn_cur_pos_items = []
        if m["pos_json_path"]:
            try:
                self._burn_cur_pos_items = json.loads(m["pos_json_path"].read_text(encoding="utf-8")).get("items", [])
            except Exception:
                pass

        cap = cv2.VideoCapture(str(m["video_path"]))
        self._burn_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._burn_total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

        total_sec = self._burn_total_frames / self._burn_fps
        self.scl_burn_time.configure(to=max(1.0, total_sec))
        default_sec = self._burn_cur_en_srt[0]["start_sec"] + 0.2 if self._burn_cur_en_srt else min(total_sec, max(0.5, total_sec * 0.72))
        self.var_burn_time.set(default_sec)

        layout_json = m.get("layout_json_path")
        if vname not in self._burn_zh_bands and layout_json is not None and Path(layout_json).exists():
            try:
                data = json.loads(Path(layout_json).read_text(encoding="utf-8"))
                y0, y1 = float(data["zh_y0"]), float(data["zh_y1"])
                if 0.35 <= y0 < y1 <= 1.0:
                    self._burn_zh_bands[vname] = (y0, y1)
            except Exception:
                pass

        self._load_burn_frame(default_sec)

        style_now = self._get_burn_style_config()
        if style_now.auto_place_mode != "manual" and vname not in self._burn_zh_bands and vname not in self._burn_probing_videos:
            self._burn_probing_videos.add(vname)
            self.var_auto_place_info.set(f"🔍 正在自动检测 [{vname}] 原中文字幕高度...")
            vid_p = m["video_path"]

            def _probe_band():
                band = load_or_detect_zh_band(vid_p, layout_json_path=layout_json)

                def _done():
                    self._burn_probing_videos.discard(vname)
                    self._burn_zh_bands[vname] = band
                    if self._burn_selected_match and self._burn_selected_match["video_path"].name == vname:
                        self._update_burn_preview()

                self.after(0, _done)

            threading.Thread(target=_probe_band, daemon=True).start()

    def _on_toggle_burn_per_video(self) -> None:
        if not self._burn_selected_match:
            return
        vname = self._burn_selected_match["video_path"].name
        if self.var_burn_per_video.get():
            self._burn_video_styles[vname] = asdict(self._get_burn_style_config())
        else:
            self._burn_video_styles.pop(vname, None)
            self._apply_burn_style_dict_to_ui(self._burn_default_style)
        self._refresh_burn_row_marker(vname)
        self._save_burn_style_store()
        self._update_burn_preview()

    def _apply_burn_style_to_all(self) -> None:
        cur_dict = asdict(self._get_burn_style_config())
        # 套用全局样式时保留全局样式不把单集单条花字微调强行套给其它集
        cur_dict["note_pos_overrides"] = {}
        self._burn_default_style = cur_dict
        self._burn_video_styles.clear()
        self._cur_note_pos_overrides.clear()
        self.var_burn_per_video.set(False)
        for m in self._burn_matches:
            self._refresh_burn_row_marker(m["video_path"].name)
        self._save_burn_style_store()
        self._update_burn_preview()
        messagebox.showinfo("完成", "已将当前主英文字幕样式设为全部视频的统一默认样式。")

    def _on_pos_y_slider_moved(self) -> None:
        """用户手动拖动主字幕垂直 Y 滑块时自动切入当前视频自由定位模式。"""
        if self._burn_loading_ui:
            self._update_burn_preview()
            return
        manual_lbl = self._auto_place_labels[self._auto_place_codes.index("manual")]
        self._burn_loading_ui = True
        try:
            self.var_auto_place_label.set(manual_lbl)
            if self._burn_selected_match:
                self.var_burn_per_video.set(True)
        finally:
            self._burn_loading_ui = False
        self._on_burn_style_edited()

    def _center_main_x(self) -> None:
        self.var_pos_x.set(0.50)
        self._on_burn_style_edited()

    def _restore_main_auto_place(self) -> None:
        auto_lbl = self._auto_place_labels[self._auto_place_codes.index("auto")]
        self._burn_loading_ui = True
        try:
            self.var_auto_place_label.set(auto_lbl)
            self.var_pos_x.set(0.50)
        finally:
            self._burn_loading_ui = False
        self._on_burn_style_edited()

    def _reset_cur_note_override(self) -> None:
        if self._burn_last_note_index is None:
            return
        key = str(self._burn_last_note_index)
        if key in self._cur_note_pos_overrides:
            self._cur_note_pos_overrides.pop(key, None)
            self._on_burn_style_edited()

    def _reset_all_note_overrides(self) -> None:
        if not self._cur_note_pos_overrides:
            return
        self._cur_note_pos_overrides.clear()
        self._on_burn_style_edited()

    def _jump_to_next_note_frame(self) -> None:
        if not self._burn_selected_match or not self._burn_cur_notes_srt:
            messagebox.showinfo("提示", "当前选中视频没有花字字幕 (*_notes.srt)。")
            return
        cur_t = self.var_burn_time.get()
        starts = [max(0.1, (ev["start_sec"] + ev["end_sec"]) / 2.0) for ev in self._burn_cur_notes_srt]
        target = next((s for s in starts if s > cur_t + 0.35), starts[0])
        self.var_burn_time.set(target)
        self._load_burn_frame(target)

    def _on_burn_sub_dragged(self, kind: str, note_index: Optional[int], nx: float, ny: float, is_final: bool) -> None:
        """左图鼠标直接拖拽主英文字幕或当前帧花字的实时回调。"""
        if self._burn_loading_ui:
            return
        self._burn_loading_ui = True
        try:
            if kind == "main":
                self.var_pos_x.set(round(nx, 4))
                self.var_pos_y.set(round(ny, 4))
                manual_lbl = self._auto_place_labels[self._auto_place_codes.index("manual")]
                self.var_auto_place_label.set(manual_lbl)
                if is_final and self._burn_selected_match:
                    self.var_burn_per_video.set(True)
            elif kind == "note" and note_index is not None:
                self._cur_note_pos_overrides[str(note_index)] = [round(nx, 4), round(ny, 4)]
                if is_final and self._burn_selected_match:
                    self.var_burn_per_video.set(True)
        finally:
            self._burn_loading_ui = False

        if is_final:
            self._on_burn_style_edited()
        else:
            self._update_burn_preview()

    def _on_burn_style_edited(self) -> None:
        if not self._burn_loading_ui:
            cur_dict = asdict(self._get_burn_style_config())
            if self.var_burn_per_video.get() and self._burn_selected_match:
                vname = self._burn_selected_match["video_path"].name
                self._burn_video_styles[vname] = cur_dict
                self._refresh_burn_row_marker(vname)
            else:
                self._burn_default_style = cur_dict
            self._save_burn_style_store()
        self._update_burn_preview()

    def _on_burn_seek(self, _val=None) -> None:
        if self._burn_seek_after_id:
            self.after_cancel(self._burn_seek_after_id)
        self._burn_seek_after_id = self.after(50, lambda: self._load_burn_frame(self.var_burn_time.get()))

    def _jump_burn_ratio(self, ratio: float) -> None:
        if not self._burn_selected_match:
            return
        total_sec = self._burn_total_frames / max(1.0, self._burn_fps)
        sec = max(0.1, min(total_sec - 0.1, total_sec * ratio))
        self.var_burn_time.set(sec)
        self._load_burn_frame(sec)

    def _load_burn_frame(self, sec: float) -> None:
        if not self._burn_selected_match:
            return
        total_sec = self._burn_total_frames / max(1.0, self._burn_fps)
        self.var_burn_time_lbl.set(f"{_fmt_sec(sec)} / {_fmt_sec(total_sec)}")
        f_idx = max(0, min(self._burn_total_frames - 1, int(round(sec * self._burn_fps))))

        cap = cv2.VideoCapture(str(self._burn_selected_match["video_path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._burn_frame_bgr = frame
            self._update_burn_preview()

    def _get_dedup_options(self) -> dict[str, bool]:
        return {k: bool(var.get()) for k, var in self.dedup_vars.items()}

    def _get_burn_style_config(self) -> BurnStyleConfig:
        lbl = self.var_auto_place_label.get()
        idx = self._auto_place_labels.index(lbl) if lbl in self._auto_place_labels else 0
        auto_mode = self._auto_place_codes[idx]
        pos_x = max(0.10, min(0.90, float(self.var_pos_x.get())))
        pos_y = max(0.08, min(0.985, float(self.var_pos_y.get())))
        return BurnStyleConfig(
            margin_v_ratio=round(max(0.015, 1.0 - pos_y), 4),
            font_size_ratio=float(self.var_font_size.get()),
            font_color_name=self.var_color_name.get(),
            outline_width=float(self.var_outline_w.get()),
            use_semi_bg_box=bool(self.var_semi_box.get()),
            note_scale_mult=float(self.var_note_mult.get()),
            note_gap_px=int(self.var_note_gap.get()),
            top_banner_text=self.var_top_banner.get(),
            auto_zoom_margin_comp=bool(self.var_zoom_comp.get()),
            auto_place_mode=auto_mode,
            auto_gap_ratio=float(self.var_auto_gap.get()),
            pos_x_ratio=round(pos_x, 4),
            pos_y_ratio=round(pos_y, 4),
            note_pos_overrides={k: [float(v[0]), float(v[1])] for k, v in self._cur_note_pos_overrides.items()},
        )

    def _set_dedup_preset(self, enable_default: bool) -> None:
        for k, var in self.dedup_vars.items():
            var.set(DEFAULT_DEDUP_OPTIONS[k] if enable_default else False)
        self._update_burn_preview()

    def _update_burn_preview(self) -> None:
        if self._burn_frame_bgr is None:
            return
        cur_t = self.var_burn_time.get()

        sample_main = self.var_burn_sample_text.get()
        if self._burn_cur_en_srt:
            matched_main = [ev["text"] for ev in self._burn_cur_en_srt if ev["start_sec"] <= cur_t <= ev["end_sec"]]
            sample_main = matched_main[0] if matched_main else ""

        sample_note = None
        if self._burn_cur_notes_srt and self._burn_cur_pos_items:
            pos_map = {int(it["index"]): it for it in self._burn_cur_pos_items}
            for i, ev in enumerate(self._burn_cur_notes_srt):
                if ev["start_sec"] <= cur_t <= ev["end_sec"]:
                    p_info = pos_map.get(ev["index"]) or (self._burn_cur_pos_items[i] if i < len(self._burn_cur_pos_items) else None)
                    if p_info:
                        sample_note = {"text_en": ev["text"], "pos_info": p_info}
                        break

        style_cfg = self._get_burn_style_config()
        dedup_opts = self._get_dedup_options()
        vname = self._burn_selected_match["video_path"].name if self._burn_selected_match else None
        zh_band = self._burn_zh_bands.get(vname) if vname else None

        fh, fw = self._burn_frame_bgr.shape[:2]
        eff_layout = compute_effective_main_layout(fw, fh, style_cfg, dedup_opts, zh_band=zh_band)
        if style_cfg.auto_place_mode != "manual" and not self._burn_loading_ui:
            self._burn_loading_ui = True
            try:
                self.var_pos_y.set(round(eff_layout["pos_y_ratio"], 4))
            finally:
                self._burn_loading_ui = False

        if vname and vname in self._burn_probing_videos:
            self.var_auto_place_info.set(f"🔍 正在自动检测 [{vname}] 原中文字幕高度...")
        else:
            self.var_auto_place_info.set(f"✨ {eff_layout['reason']}")

        self.var_burn_coord_info.set(
            f"主字幕中心坐标：X={eff_layout['pos_x_ratio']*100:.1f}% ({eff_layout['px']}px), Y={eff_layout['pos_y_ratio']*100:.1f}% ({eff_layout['py']}px)"
        )

        pil_img, meta = render_burn_preview_with_meta(
            self._burn_frame_bgr,
            style_cfg,
            dedup_opts,
            sample_en_text=sample_main,
            sample_note=sample_note,
            zh_band=zh_band,
        )
        self._burn_last_note_index = meta.get("note_index")
        nl = meta.get("note_layout")
        if nl:
            mode_tag = "已手动拖拽" if nl.get("is_manual") else "自动避让"
            self.var_cur_note_info.set(
                f"当前花字 #{nl['index']} [{nl['nx_ratio']*100:.1f}%, {nl['ny_ratio']*100:.1f}%] ({mode_tag})"
            )
        else:
            cnt = len(self._cur_note_pos_overrides)
            tip = f"（本集已微调 {cnt} 条花字）" if cnt > 0 else "（可点「跳到花字帧」后在左图拖动）"
            self.var_cur_note_info.set(f"当前帧无花字{tip}")

        self.burn_canvas.set_image(pil_img)
        self.burn_canvas.set_sub_targets(
            meta.get("main_box"),
            meta.get("main_anchor"),
            note_box=meta.get("note_box"),
            note_anchor=meta.get("note_anchor"),
            note_index=meta.get("note_index"),
            enabled=bool(self.var_burn_drag_enable.get()),
        )

    def _start_burn(self, test_sec: int) -> None:
        if self._is_busy:
            messagebox.showwarning("提示", "已有任务正在运行中，请先等待完成或点击停止。")
            return
        self._refresh_burn_matches()

        if test_sec > 0:
            if not self._burn_selected_match or not self._burn_selected_match["ready"]:
                messagebox.showerror("错误", "请在左侧列表中先选中一个已匹配英文 SRT 的视频来试压！")
                return
            items = [self._burn_selected_match]
        else:
            items = [m for m in self._burn_matches if m["ready"]]
            if not items:
                messagebox.showerror("错误", "当前没有匹配就绪的视频（请检查英文 SRT 文件名是否与原视频完全同名）。")
                return

        out_dir = Path(self.var_out_dir.get().strip())
        enc_idx = self._enc_labels.index(self.var_enc_label.get()) if self.var_enc_label.get() in self._enc_labels else 0
        enc_code = self._enc_codes[enc_idx]
        dedup_opts = self._get_dedup_options()
        default_style_cfg = self._style_dict_to_cfg(self._burn_default_style) if not (test_sec > 0 and self.var_burn_per_video.get()) else self._get_burn_style_config()
        per_video_styles = {name: self._style_dict_to_cfg(st_dict) for name, st_dict in self._burn_video_styles.items()}
        skip_exist = bool(self.var_burn_skip.get())

        self._set_busy(True, "burn")
        self._cancel_event.clear()
        self.txt_burn_log.delete("1.0", tk.END)
        self.var_burn_cur_prog.set(0.0)
        self.var_burn_tot_prog.set(0.0)

        def _worker():
            def log(msg: str):
                self.after(0, lambda m=msg: self._append_log(self.txt_burn_log, m))

            def prog(cur_idx: int, total: int, cur_pct: float, msg: str):
                overall_pct = (((max(1, cur_idx) - 1) + (cur_pct / 100.0)) / max(1, total)) * 100.0

                def _upd():
                    self.var_burn_cur_prog.set(cur_pct)
                    self.var_burn_cur_msg.set(f"当前视频：{msg} ({cur_pct:.1f}%)")
                    self.var_burn_tot_prog.set(min(100.0, overall_pct))
                    self.var_burn_tot_msg.set(f"总进度：第 {cur_idx} / {total} 个视频（总计 {overall_pct:.1f}%）")

                self.after(0, _upd)

            res = burn_batch(
                items,
                out_dir,
                mode=enc_code,
                dedup_options=dedup_opts,
                style=default_style_cfg,
                test_sec=test_sec,
                skip_existing=skip_exist,
                progress_cb=prog,
                log_cb=log,
                cancel_event=self._cancel_event,
                per_video_styles=per_video_styles,
            )

            def _done():
                self._set_busy(False, "burn")
                if not res["cancelled"]:
                    msg = f"压制结束：成功 {res['done']} 个，跳过 {res['skipped']} 个，失败 {res['failed']} 个。\n成品目录：{out_dir}"
                    messagebox.showinfo("完成", msg)

            self.after(0, _done)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    # ==================== 运行状态管理 ====================

    def _set_busy(self, busy: bool, which: str) -> None:
        self._is_busy = busy
        state_run = tk.DISABLED if busy else tk.NORMAL
        state_stop = tk.NORMAL if busy else tk.DISABLED

        self.btn_pipeline.configure(state=state_run)
        self.btn_auto_detect.configure(state=state_run)
        self.btn_run_extract.configure(state=state_run)
        self.btn_run_translate.configure(state=state_run)
        self.btn_test_trans.configure(state=state_run)
        self.btn_run_burn.configure(state=state_run)
        self.btn_test_burn.configure(state=state_run)

        self.btn_stop_pipeline.configure(state=state_stop if which == "pipeline" else tk.DISABLED)
        self.btn_stop_extract.configure(state=state_stop if which in ("extract", "detect", "pipeline") else tk.DISABLED)
        self.btn_stop_translate.configure(state=state_stop if which in ("translate", "pipeline") else tk.DISABLED)
        self.btn_stop_burn.configure(state=state_stop if which in ("burn", "pipeline") else tk.DISABLED)

    def _stop_task(self) -> None:
        if self._is_busy:
            self._cancel_event.set()
            self.var_pipeline_status.set("正在请求安全停止，请稍候...")
            self._append_log(self.txt_ext_log, "[正在请求停止...] 请稍候...")
            self._append_log(self.txt_trans_log, "[正在请求停止...] 请稍候...")
            self._append_log(self.txt_burn_log, "[正在请求停止...] 正在安全终止当前 FFmpeg 进程...")

    def _append_log(self, txt_widget: tk.Text, msg: str) -> None:
        txt_widget.insert(tk.END, msg + "\n")
        txt_widget.see(tk.END)

    def _on_close(self) -> None:
        if self._is_busy:
            if not messagebox.askyesno("确认退出", "当前仍有任务正在处理，确定要强制停止并退出吗？"):
                return
            self._cancel_event.set()
        if self._ext_cap:
            self._ext_cap.release()
        self.destroy()
