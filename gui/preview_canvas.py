import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from PIL import Image, ImageTk


class PreviewCanvas(ttk.Frame):
    """自适应缩放视频帧预览画布，支持鼠标左键拖拽绘制红框（主台词）与蓝框（花字区）。"""

    def __init__(
        self,
        parent,
        interactive_roi: bool = False,
        on_roi_changed: Optional[Callable[[str, tuple[float, float, float, float]], None]] = None,
    ):
        super().__init__(parent)
        self.interactive_roi = interactive_roi
        self.on_roi_changed = on_roi_changed

        self.canvas = tk.Canvas(self, bg="#181a1f", highlightthickness=1, highlightbackground="#3a3f4b")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._pil_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._disp_box: tuple[int, int, int, int] = (0, 0, 1, 1)  # (dx, dy, dw, dh)

        self.main_roi: tuple[float, float, float, float] = (0.08, 0.81, 0.92, 0.92)
        self.note_roi: tuple[float, float, float, float] = (0.00, 0.00, 1.00, 1.00)
        self.show_note_roi: bool = True
        self.active_box: str = "main"  # "main" or "note"

        self._drag_start_px: Optional[tuple[int, int]] = None
        self._drag_temp_rect: Optional[tuple[float, float, float, float]] = None

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        if self.interactive_roi:
            self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
            self.canvas.bind("<B1-Motion>", self._on_mouse_move)
            self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
            self.canvas.config(cursor="crosshair")

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

    def _px_to_ratio(self, cx: int, cy: int) -> tuple[float, float]:
        dx, dy, dw, dh = self._disp_box
        rx = (cx - dx) / max(1, dw)
        ry = (cy - dy) / max(1, dh)
        return max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))

    def _ratio_to_px(self, rx: float, ry: float) -> tuple[int, int]:
        dx, dy, dw, dh = self._disp_box
        return int(round(dx + rx * dw)), int(round(dy + ry * dh))

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

        # 优先放在选区框顶线外侧上方（使用 sw 锚点彻底避免 High-DPI 下压进框内遮挡台词）
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
