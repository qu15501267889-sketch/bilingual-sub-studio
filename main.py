import argparse
import ctypes
import os
import sys
import tempfile
from pathlib import Path


def _fix_windowed_stdio() -> None:
    """在 PyInstaller --windowed 模式下 stdout/stderr 可能为 None，防止 print 或库输出报错。"""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _enable_high_dpi() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def run_self_test(video_path: Path) -> int:
    """无窗口端到端自测模式：验证 DirectML 显卡加速 + 双进度回调 + 新提示词 + 内置 bin/ffmpeg 全链路。"""
    import cv2
    from core.burner import (
        DEFAULT_DEDUP_OPTIONS,
        BurnStyleConfig,
        burn_video,
        compute_effective_main_layout,
        detect_best_encoder,
        get_video_info,
        render_burn_preview_pil,
    )
    from core.extractor import (
        ExtractConfig,
        auto_detect_main_roi,
        create_ocr_engine,
        detect_ocr_accelerator,
        extract_main_track,
        generate_agent_prompt,
    )
    from core.runtime_paths import (
        DEFAULT_GLOSSARY,
        get_app_dir,
        get_ffmpeg_exe,
        get_ffprobe_exe,
        get_models_dir,
        run_silent,
    )

    print("=== 开始全链路自测 ===")
    print("AppDir :", get_app_dir())
    print("Models :", get_models_dir())
    print("FFmpeg :", get_ffmpeg_exe())
    print("FFprobe:", get_ffprobe_exe())
    print("OCR加速:", detect_ocr_accelerator())
    print("Encoder:", detect_best_encoder())

    if not video_path.exists():
        print(f"[FAIL] 测试视频不存在: {video_path}")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg = ExtractConfig(accel_mode="auto", ocr_profile="v5_server")
        ocr_adapter, ocr_backend = create_ocr_engine("auto", "v5_server")
        print(f"[自测-PP-OCRv5] backend={ocr_backend}, rec={Path(ocr_adapter.rec_path).name}")
        if ocr_backend != "directml" or not Path(ocr_adapter.rec_path).exists():
            print("[FAIL] 离线 PP-OCRv5 模型加载失败或未命中 DirectML")
            return 19

        zh_srt = tmp / "test_zh.srt"
        if cfg.filter_watermark is not True:
            print("[FAIL] ExtractConfig.filter_watermark 默认值应为 True")
            return 20

        det_roi, det_found, det_hits = auto_detect_main_roi(video_path, accel_mode="auto")
        print(f"[自测-红框探测] found={det_found}, hits={det_hits}, roi={det_roi}")
        if (not det_found) or not (0.42 <= det_roi[1] < det_roi[3] <= 0.995) or det_roi[0] > 0.10 or det_roi[2] < 0.90:
            print("[FAIL] auto_detect_main_roi 未命中或返回坐标范围异常")
            return 21

        ocr_pcts: list[float] = []
        def _ocr_prog(pct: float, stage: str):
            ocr_pcts.append(pct)

        segs, info = extract_main_track(video_path, zh_srt, cfg, item_progress_cb=_ocr_prog)
        print(f"[自测-OCR提取] 提取 {len(segs)} 条底部台词 | 后端={info.get('backend')} | 耗时={info.get('elapsed', 0):.2f}s | 进度点={len(ocr_pcts)}次 ({ocr_pcts[0]:.0f}%->{ocr_pcts[-1]:.0f}%)")
        if len(segs) < 5 or info.get("backend") != "directml" or len(ocr_pcts) < 5:
            print("[FAIL] 提取字幕异常、未命中 DirectML 或单视频进度条未实时更新")
            return 2

        layout_json_p = tmp / "test_zh_main.layout.json"
        if not layout_json_p.exists() or not info.get("zh_band"):
            print("[FAIL] 提取主台词后未生成 _main.layout.json 或 zh_band")
            return 22
        l_below = compute_effective_main_layout(1920, 1080, BurnStyleConfig(), DEFAULT_DEDUP_OPTIONS, zh_band=(0.83, 0.90))
        l_above = compute_effective_main_layout(1920, 1080, BurnStyleConfig(), DEFAULT_DEDUP_OPTIONS, zh_band=(0.89, 0.955))
        print(f"[自测-自动防撞] 常规中文->{l_below['placement']} | 极低贴底->{l_above['placement']}")
        if l_below["placement"] != "below" or l_above["placement"] != "above":
            print("[FAIL] compute_effective_main_layout 自动上下防撞决策异常")
            return 23

        # 测试新提示词（多地方言 + 整句通顺度校对）
        prompt_p = tmp / "AGENT_PROMPT.txt"
        ptxt = generate_agent_prompt(tmp, tmp / "en", ["test_zh.srt"], DEFAULT_GLOSSARY, prompt_p)
        if "【东北方言高频】" not in ptxt or "【川渝方言高频】" not in ptxt or "是不是一句完整、顺畅" not in ptxt:
            print("[FAIL] 新版方言与整句通顺纠错提示词内容缺失")
            return 25

        # 测试内存实时预览图合成
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(72.5 * fps)))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[FAIL] 抽帧失败")
            return 3

        preview_pil = render_burn_preview_pil(
            frame, BurnStyleConfig(), DEFAULT_DEDUP_OPTIONS,
            sample_en_text="the \"Dave Jianguo Award\"!",
        )
        prev_jpg = tmp / "preview_check.jpg"
        preview_pil.save(prev_jpg)
        print(f"[自测-预览图] 生成成功 {preview_pil.size}, size={prev_jpg.stat().st_size} bytes")

        # 测试调用内置 FFmpeg 压制 5 秒双语去重短视频并校验实时进度推送
        out_mp4 = tmp / "selftest_5s.mp4"
        burn_pcts: list[float] = []
        def _burn_prog(pct: float, stage: str):
            burn_pcts.append(pct)

        ok = burn_video(
            video_path,
            zh_srt,
            out_mp4,
            mode="auto",
            dedup_options=DEFAULT_DEDUP_OPTIONS,
            style=BurnStyleConfig(),
            test_sec=5,
            item_progress_cb=_burn_prog,
            layout_json_path=layout_json_p,
        )
        print(f"[自测-烧录进度] 推送 {len(burn_pcts)} 次进度更新 ({burn_pcts[0]:.1f}%->{burn_pcts[-1]:.1f}%)")
        if not ok or not out_mp4.exists() or out_mp4.stat().st_size < 100_000 or len(burn_pcts) < 3:
            print("[FAIL] FFmpeg 压制失败、输出异常或烧录进度未实时回传")
            return 4

        out_info = get_video_info(out_mp4)
        leftovers = list(tmp.glob("*.building.mp4"))
        sar_res = run_silent(
            [get_ffprobe_exe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=sample_aspect_ratio", "-of", "default=nw=1:nk=1", str(out_mp4)],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        sar_val = sar_res.stdout.strip()
        print(f"[自测-FFprobe校验] {out_info['w']}x{out_info['h']} duration={out_info['duration']:.2f}s SAR={sar_val} size={out_mp4.stat().st_size} 残留临时文件={len(leftovers)}")
        if leftovers or sar_val not in ("1:1", "N/A", ""):
            print(f"[FAIL] 原子输出异常或 SAR 不规范: leftovers={leftovers}, SAR={sar_val}")
            return 5

    print("[SELF-TEST PASS] 全链路验证通过！")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="双语字幕工坊")
    parser.add_argument("--self-test", type=Path, default=None, help="运行无窗口端到端自测模式")
    parser.add_argument("--log-file", type=Path, default=None, help="将自测日志写入指定文件")
    args, _unknown = parser.parse_known_args()

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "w", encoding="utf-8", buffering=1)
        sys.stdout = log_fp
        sys.stderr = log_fp
    else:
        _fix_windowed_stdio()

    if args.self_test is not None:
        code = run_self_test(args.self_test)
        sys.stdout.flush()
        sys.exit(code)

    _enable_high_dpi()
    from gui.app import BilingualSubStudioApp

    app = BilingualSubStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
