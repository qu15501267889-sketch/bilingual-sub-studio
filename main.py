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


def run_translate_smoke_test() -> int:
    """单独验证内置 B 站 Index-Translate-9B 本地 Ollama 自动翻译与显存卸载。"""
    from core.runtime_paths import DEFAULT_GLOSSARY, load_translator_config
    from core.translator import (
        _ZH_CHAR_RE,
        TranslatorConfig,
        parse_srt_blocks,
        translate_srt_file,
        unload_ollama_model,
    )

    print("=== 开始内置 AI 翻译冒烟自测 ===")
    raw_cfg = load_translator_config()
    cfg = TranslatorConfig(**{k: raw_cfg[k] for k in TranslatorConfig.__dataclass_fields__ if k in raw_cfg})
    print(f"[翻译配置] provider={cfg.provider}, model={cfg.ollama_model if cfg.provider == 'ollama' else cfg.api_model}")

    sample_zh = (
        "1\n00:00:00,100 --> 00:00:01,500\n戴建国同志，村口赶大集都散场了\n\n"
        "2\n00:00:01,600 --> 00:00:03,100\n睫这豆子还甜嘞，自已人别客气\n\n"
        "3\n00:00:03,200 --> 00:00:04,900\n少在这扯犊子，赶紧回家干活\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zh_p = tmp / "sample_zh.srt"
        en_p = tmp / "sample_en.srt"
        zh_p.write_text(sample_zh, encoding="utf-8")

        ok, info = translate_srt_file(zh_p, en_p, cfg, DEFAULT_GLOSSARY, log_cb=print)
        if not ok or not en_p.exists():
            print("[FAIL] translate_srt_file 未产出英文 SRT")
            return 31

        zh_b = parse_srt_blocks(zh_p)
        en_b = parse_srt_blocks(en_p)
        print(f"[翻译校验] 原={len(zh_b)}条, 译={len(en_b)}条, 耗时={info.get('elapsed', 0):.2f}s")
        for b in en_b:
            print(f"  #{b.index} ({b.start_str}->{b.end_str}): {b.text}")
        if len(en_b) != len(zh_b):
            print("[FAIL] 翻译前后字幕条数不一致")
            return 32
        if any(a.start_str != b.start_str or a.end_str != b.end_str for a, b in zip(zh_b, en_b)):
            print("[FAIL] 时间轴被改动")
            return 33
        if any(_ZH_CHAR_RE.search(b.text) or not b.text.strip() for b in en_b):
            print("[FAIL] 译文为空或仍残留汉字")
            return 34

        if cfg.provider == "ollama" and cfg.unload_after_done:
            unloaded = unload_ollama_model(cfg.ollama_url, cfg.ollama_model)
            print(f"[显存卸载] unload_ollama_model={unloaded}")
            if not unloaded:
                print("[FAIL] 显存卸载调用返回 False")
                return 35

    print("[TRANSLATE-TEST PASS] 内置 AI 翻译与显存卸载验证通过！")
    return 0


def run_self_test(video_path: Path) -> int:
    """无窗口端到端自测模式：验证 PP-OCRv5 DirectML + (X,Y)自由定位 + 花字覆写 + 内置 FFmpeg 全链路。"""
    import cv2
    from core.burner import (
        DEFAULT_DEDUP_OPTIONS,
        BurnStyleConfig,
        burn_video,
        compute_effective_main_layout,
        compute_note_layout,
        detect_best_encoder,
        get_video_info,
        render_burn_preview_with_meta,
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

    print("=== 开始 V1.5 全链路自测 ===")
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
        det_roi, det_found, det_hits = auto_detect_main_roi(video_path, accel_mode="auto")
        print(f"[自测-红框探测] found={det_found}, hits={det_hits}, roi={det_roi}")
        if (not det_found) or not (0.42 <= det_roi[1] < det_roi[3] <= 0.995) or det_roi[0] > 0.10 or det_roi[2] < 0.90:
            print("[FAIL] auto_detect_main_roi 未命中或返回坐标范围异常")
            return 21

        ocr_pcts: list[float] = []
        segs, info = extract_main_track(video_path, zh_srt, cfg, item_progress_cb=lambda p, _s: ocr_pcts.append(p))
        print(f"[自测-OCR提取] 提取 {len(segs)} 条底部台词 | 后端={info.get('backend')} | 耗时={info.get('elapsed', 0):.2f}s | 进度点={len(ocr_pcts)}次")
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

        # 验证自由 (X, Y) 定位与单条花字坐标覆写
        free_style = BurnStyleConfig(
            auto_place_mode="manual",
            pos_x_ratio=0.35,
            pos_y_ratio=0.72,
            note_pos_overrides={"1": [0.75, 0.28]},
        )
        l_free = compute_effective_main_layout(1920, 1080, free_style, DEFAULT_DEDUP_OPTIONS, zh_band=(0.83, 0.90))
        n_free = compute_note_layout(
            {"index": 1, "cx": 500, "y0": 400, "y1": 460, "bh": 60},
            "Hand-Dragged Note",
            1920, 1080,
            free_style,
            DEFAULT_DEDUP_OPTIONS,
            has_main_overlap=True,
            main_zone_top_px=l_free["main_zone_top_px"],
        )
        print(f"[自测-自由(X,Y)] main=(px={l_free['px']}, py={l_free['py']}, maxW={l_free['max_line_w']:.0f}) | note=(align={n_free['align']}, px={n_free['px']}, py={n_free['py']})")
        if abs(l_free["px"] - int(round(1920 * 0.35))) > 2 or abs(l_free["py"] - int(round(1080 * 0.72))) > 2 or n_free["align"] != 5:
            print("[FAIL] 主字幕或单条花字自由 (X, Y) 定位公式异常")
            return 24

        prompt_p = tmp / "AGENT_PROMPT.txt"
        ptxt = generate_agent_prompt(tmp, tmp / "en", ["test_zh.srt"], DEFAULT_GLOSSARY, prompt_p)
        if "【东北方言高频】" not in ptxt or "是不是一句完整、顺畅" not in ptxt:
            print("[FAIL] 新版方言与整句通顺纠错提示词内容缺失")
            return 25

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(72.5 * fps)))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[FAIL] 抽帧失败")
            return 3

        preview_pil, prev_meta = render_burn_preview_with_meta(
            frame,
            free_style,
            DEFAULT_DEDUP_OPTIONS,
            sample_en_text="the \"Dave Jianguo Award\"!",
            sample_note={"text_en": "Hand-Dragged Note", "pos_info": {"index": 1, "cx": 500, "y0": 400, "y1": 460, "bh": 60}},
        )
        if not prev_meta.get("main_box") or not prev_meta.get("note_box") or prev_meta.get("note_index") != 1:
            print("[FAIL] 预览画布拖拽元数据返回缺失")
            return 26
        print(f"[自测-预览图+拖拽框] size={preview_pil.size}, main_box={prev_meta['main_box']}, note_box={prev_meta['note_box']}")

        out_mp4 = tmp / "selftest_5s.mp4"
        burn_pcts: list[float] = []
        ok = burn_video(
            video_path,
            zh_srt,
            out_mp4,
            mode="auto",
            dedup_options=DEFAULT_DEDUP_OPTIONS,
            style=BurnStyleConfig(),
            test_sec=5,
            item_progress_cb=lambda p, _s: burn_pcts.append(p),
            layout_json_path=layout_json_p,
        )
        print(f"[自测-烧录进度] 推送 {len(burn_pcts)} 次进度更新")
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
    parser = argparse.ArgumentParser(description="双语字幕工坊 V1.5")
    parser.add_argument("--self-test", type=Path, default=None, help="运行无窗口端到端自测模式")
    parser.add_argument("--test-translate", action="store_true", help="运行内置 AI 翻译冒烟自测（本机 Ollama Index-Translate-9B）")
    parser.add_argument("--log-file", type=Path, default=None, help="将自测日志写入指定文件")
    args, _unknown = parser.parse_known_args()

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "w", encoding="utf-8", buffering=1)
        sys.stdout = log_fp
        sys.stderr = log_fp
    else:
        _fix_windowed_stdio()

    exit_code = 0
    ran_cli_test = False
    if args.self_test is not None:
        ran_cli_test = True
        exit_code = run_self_test(args.self_test)
        if exit_code != 0:
            sys.stdout.flush()
            sys.exit(exit_code)
    if args.test_translate:
        ran_cli_test = True
        exit_code = run_translate_smoke_test()
        sys.stdout.flush()
        sys.exit(exit_code)
    if ran_cli_test:
        sys.stdout.flush()
        sys.exit(exit_code)

    _enable_high_dpi()
    from gui.app import BilingualSubStudioApp

    app = BilingualSubStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
