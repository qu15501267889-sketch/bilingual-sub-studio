import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from PIL import Image, ImageTk


class PreviewCanvas(ttk.Frame):
    """自适应缩放视频帧预览画布，支持红/蓝框 ROI 选区（提取页）与主字幕/花字 X/Y 鼠标拖拽（烧录页）。"""

    def __init__(
        self,
        parent,
        interactive_roi: bool = False,
        on_roi_changed: Optional[Callable[[str, tuple[float, float, float, float]], None]] = None,
        interactive_sub_pos: bool = False,
        on_sub_dragged: Optional[Callable[[str, Optional[int], float, float, bool], None]] = None,
    ):
        super().__init__(parent)
        self.interactive_roi = interactive_roi
        self.on_roi_changed = on_roi_changed
        self.interactive_sub_pos = interactive_sub_pos
        self.on_sub_dragged = on_sub_dragged

        self.canvas = tk.Canvas(self, bg="#181a1f", highlightthickness=1, highlightbackground="#3a3f4b")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._pil_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._disp_box: tuple[int, int, int, int] = (0, 0, 1, 1)  # (dx, dy, dw, dh)

        # 提取页 ROI 状态
        self.main_roi: tuple[float, float, float, float] = (0.08, 0.81, 0.92, 0.92)
        self.note_roi: tuple[float, float, float, float] = (0.00, 0.00, 1.00, 1.00)
        self.show_note_roi: bool = True
        self.active_box: str = "main"  # "main" or "note"

        self._drag_start_px: Optional[tuple[int, int]] = None
        self._drag_temp_rect: Optional[tuple[float, float, float, float]] = None

        # 烧录页字幕与花字自由拖拽状态
        self._sub_drag_enabled: bool = True
        self._sub_main_box: Optional[tuple[float, float, float, float]] = None
        self._sub_main_anchor: Optional[tuple[float, float]] = None
        self._sub_note_box: Optional[tuple[float, float, float, float]] = None
        self._sub_note_anchor: Optional[tuple[float, float]] = None
        self._sub_note_index: Optional[int] = None
        self._sub_dragging: Optional[tuple[str, Optional[int], float, float]] = None  # (kind, note_idx, off_rx, off_ry)
        self._sub_last_emit_ts: float = 0.0

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        if self.interactive_roi:
            self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
            self.canvas.bind("<B1-Motion>", self._on_mouse_move)
            self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
            self.canvas.config(cursor="crosshair")
        elif self.interactive_sub_pos:
            self.canvas.bind("<Motion>", self._on_sub_hover)
            self.canvas.bind("<ButtonPress-1>", self._on_sub_down)
            self.canvas.bind("<B1-Motion>", self._on_sub_move)
            self.canvas.bind("<ButtonRelease-1>", self._on_sub_up)

    def set_image(self, pil_img: Optional[Image.Image]) -> None:
        self._pil_image = pil_img
        self._redraw()

    def set_rois(
        self,
        main_roi: tuple[float, float, float, float],
        note_roi: tuple[float, float, float, float],
        show_note_roi: bool = True,
    ) -> None:
        self.main_roi = main_roi
        self.note_roi = note_roi
        self.show_note_roi = show_note_roi
        self._redraw()

    def set_active_box(self, which: str) -> None:
        self.active_box = which

    def set_sub_targets(
        self,
        main_box: Optional[tuple[float, float, float, float]],
        main_anchor: Optional[tuple[float, float]],
        note_box: Optional[tuple[float, float, float, float]] = None,
        note_anchor: Optional[tuple[float, float]] = None,
        note_index: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        """设置烧录预览页中可鼠标拖拽调整的主英文字幕与当前花字命中盒。"""
        self._sub_main_box = main_box
        self._sub_main_anchor = main_anchor
        self._sub_note_box = note_box
        self._sub_note_anchor = note_anchor
        self._sub_note_index = note_index
        self._sub_drag_enabled = enabled
        self._redraw()

    def _px_to_ratio(self, cx: int, cy: int) -> tuple[float, float]:
        dx, dy, dw, dh = self._disp_box
        rx = (cx - dx) / max(1, dw)
        ry = (cy - dy) / max(1, dh)
        return max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))

    def _ratio_to_px(self, rx: float, ry: float) -> tuple[int, int]:
        dx, dy, dw, dh = self._disp_box
        return int(round(dx + rx * dw)), int(round(dy + ry * dh))

    # ---- 提取页 ROI 框选事件 ----

    def _on_mouse_down(self, event) -> None:
        if self._pil_image is None:
            return
        self._drag_start_px = (event.x, event.y)
        self._drag_temp_rect = None

    def _on_mouse_move(self, event) -> None:
        if self._pil_image is None or self._drag_start_px is None:
            return
        x0, y0 = self._px_to_ratio(*self._drag_start_px)
        x1, y1 = self._px_to_ratio(event.x, event.y)
        self._drag_temp_rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._redraw()

    def _on_mouse_up(self, event) -> None:
        if self._pil_image is None or self._drag_start_px is None:
            return
        x0, y0 = self._px_to_ratio(*self._drag_start_px)
        x1, y1 = self._px_to_ratio(event.x, event.y)
        self._drag_start_px = None
        self._drag_temp_rect = None

        rect = (round(min(x0, x1), 3), round(min(y0, y1), 3), round(max(x0, x1), 3), round(max(y0, y1), 3))
        if (rect[2] - rect[0]) >= 0.03 and (rect[3] - rect[1]) >= 0.02:
            if self.active_box == "main":
                self.main_roi = rect
            else:
                self.note_roi = rect
            if self.on_roi_changed:
                self.on_roi_changed(self.active_box, rect)
        self._redraw()

    # ---- 烧录页字幕/花字 X,Y 拖拽事件 ----

    def _hit_sub_target(self, cx: int, cy: int) -> Optional[str]:
        if (not self._sub_drag_enabled) or (self._pil_image is None):
            return None
        rx, ry = self._px_to_ratio(cx, cy)
        dx, dy, dw, dh = self._disp_box
        pad_x = 10.0 / max(1, dw)
        pad_y = 10.0 / max(1, dh)

        # 花字框较小，优先命中当前可见花字
        if self._sub_note_box and self._sub_note_anchor and self._sub_note_index is not None:
            nx0, ny0, nx1, ny1 = self._sub_note_box
            if (nx0 - pad_x) <= rx <= (nx1 + pad_x) and (ny0 - pad_y) <= ry <= (ny1 + pad_y):
                return "note"
        if self._sub_main_box and self._sub_main_anchor:
            mx0, my0, mx1, my1 = self._sub_main_box
            if (mx0 - pad_x) <= rx <= (mx1 + pad_x) and (my0 - pad_y) <= ry <= (my1 + pad_y):
                return "main"
        return None

    def _on_sub_hover(self, event) -> None:
        if self._sub_dragging is not None:
            return
        hit = self._hit_sub_target(event.x, event.y)
        self.canvas.config(cursor="fleur" if hit else "")

    def _on_sub_down(self, event) -> None:
        hit = self._hit_sub_target(event.x, event.y)
        if not hit:
            return
        rx, ry = self._px_to_ratio(event.x, event.y)
        if hit == "note" and self._sub_note_anchor is not None:
            ax, ay = self._sub_note_anchor
            self._sub_dragging = ("note", self._sub_note_index, ax - rx, ay - ry)
        elif hit == "main" and self._sub_main_anchor is not None:
            ax, ay = self._sub_main_anchor
            self._sub_dragging = ("main", None, ax - rx, ay - ry)
        self._sub_last_emit_ts = 0.0
        self.canvas.config(cursor="fleur")

    def _clamp_sub_anchor(self, kind: str, rx: float, ry: float, off_rx: float, off_ry: float) -> tuple[float, float]:
        nx = rx + off_rx
        ny = ry + off_ry
        if kind == "note":
            return round(max(0.05, min(0.95, nx)), 4), round(max(0.04, min(0.96, ny)), 4)
        return round(max(0.10, min(0.90, nx)), 4), round(max(0.08, min(0.985, ny)), 4)

    def _on_sub_move(self, event) -> None:
        if self._sub_dragging is None or self._pil_image is None:
            return
        kind, n_idx, off_x, off_y = self._sub_dragging
        now = time.monotonic()
        if (now - self._sub_last_emit_ts) < 0.035:
            return
        self._sub_last_emit_ts = now
        rx, ry = self._px_to_ratio(event.x, event.y)
        new_x, new_y = self._clamp_sub_anchor(kind, rx, ry, off_x, off_y)
        if self.on_sub_dragged:
            self.on_sub_dragged(kind, n_idx, new_x, new_y, False)

    def _on_sub_up(self, event) -> None:
        if self._sub_dragging is None:
            return
        kind, n_idx, off_x, off_y = self._sub_dragging
        self._sub_dragging = None
        self.canvas.config(cursor="")
        if self._pil_image is not None and self.on_sub_dragged:
            rx, ry = self._px_to_ratio(event.x, event.y)
            new_x, new_y = self._clamp_sub_anchor(kind, rx, ry, off_x, off_y)
            self.on_sub_dragged(kind, n_idx, new_x, new_y, True)

    # ---- 绘制 ----

    def _draw_labeled_box(self, rect: tuple[float, float, float, float], color: str, label: str, dashed: bool = False, width: int = 2) -> None:
        x1, y1 = self._ratio_to_px(rect[0], rect[1])
        x2, y2 = self._ratio_to_px(rect[2], rect[3])
        if dashed:
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=(6, 4))
        else:
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)

        ch = max(10, self.canvas.winfo_height())
        dx, _, dw, _ = self._disp_box
        lx = max(dx + 2, min(x1, max(dx + 2, dx + dw - 120)))

        if y1 >= 28:
            ty, anchor = y1 - 5, "sw"
        elif (not dashed) and y2 <= ch - 28:
            ty, anchor = y2 + 5, "nw"
        else:
            ty, anchor = y1 + 5, "nw"

        txt_id = self.canvas.create_text(lx + 4, ty, text=label, fill="#ffffff", font=("Microsoft YaHei UI", 9, "bold"), anchor=anchor)
        bbox = self.canvas.bbox(txt_id)
        if bbox:
            bg_id = self.canvas.create_rectangle(bbox[0] - 4, bbox[1] - 1, bbox[2] + 4, bbox[3] + 1, fill=color, outline="")
            self.canvas.tag_lower(bg_id, txt_id)

    def _redraw(self) -> None:
        self.canvas.delete("all")
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())

        if self._pil_image is None:
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text="请先选择原视频文件夹加载画面预览",
                fill="#8e95a5",
                font=("Microsoft YaHei UI", 11),
            )
            return

        iw, ih = self._pil_image.size
        scale = min((cw - 12) / max(1, iw), (ch - 12) / max(1, ih))
        dw = max(1, int(iw * scale))
        dh = max(1, int(ih * scale))
        dx = (cw - dw) // 2
        dy = (ch - dh) // 2
        self._disp_box = (dx, dy, dw, dh)

        resized = self._pil_image.resize((dw, dh), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(dx, dy, image=self._tk_image, anchor="nw")

        if self.interactive_roi:
            if self.show_note_roi:
                n_rect = self._drag_temp_rect if (self._drag_temp_rect and self.active_box == "note") else self.note_roi
                self._draw_labeled_box(n_rect, "#1e90ff", "花字/背景扫描区(蓝)", dashed=True, width=2)
            m_rect = self._drag_temp_rect if (self._drag_temp_rect and self.active_box == "main") else self.main_roi
            self._draw_labeled_box(m_rect, "#ff3b30", "主台词提取区(红)", dashed=False, width=3)
        elif self.interactive_sub_pos and self._sub_drag_enabled:
            if self._sub_main_box:
                x1, y1 = self._ratio_to_px(self._sub_main_box[0], self._sub_main_box[1])
                x2, y2 = self._ratio_to_px(self._sub_main_box[2], self._sub_main_box[3])
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#34d399", width=1, dash=(4, 3))
            if self._sub_note_box and self._sub_note_index is not None:
                x1, y1 = self._ratio_to_px(self._sub_note_box[0], self._sub_note_box[1])
                x2, y2 = self._ratio_to_px(self._sub_note_box[2], self._sub_note_box[3])
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#facc15", width=1, dash=(4, 3))
