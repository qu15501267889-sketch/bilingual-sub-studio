import hashlib
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models" / "ppocr_v5"

# 官方 RapidOCR v3.10.0 对应的 PP-OCRv5 ONNX 模型清单与 SHA-256 校验值
MODEL_SPECS: list[tuple[str, str, str]] = [
    (
        "ch_PP-OCRv5_det_mobile.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "ch_PP-OCRv5_rec_server.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    ),
    (
        "ch_PP-OCRv5_rec_mobile.onnx",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    ),
]


def calc_sha256(file_path: Path) -> str:
    """计算指定文件的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with file_path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_progress(url: str, dest: Path) -> None:
    """使用标准库下载文件到临时路径并打印下载体积。"""
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "BilingualSubStudio/1.3"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp_path.open("wb") as out_fp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            out_fp.write(chunk)
            downloaded += len(chunk)
        size_mb = downloaded / 1024 / 1024
        total_mb = (total / 1024 / 1024) if total > 0 else size_mb
        print(f"    下载完成: {size_mb:.2f} / {total_mb:.2f} MB")

    os.replace(tmp_path, dest)


def ensure_models() -> None:
    """检查并下载全套离线 PP-OCRv5 模型文件，逐个进行 SHA-256 完整性校验。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"模型保存目录: {MODELS_DIR}")

    for filename, url, expected_sha in MODEL_SPECS:
        target = MODELS_DIR / filename
        if target.is_file():
            actual_sha = calc_sha256(target)
            if actual_sha == expected_sha:
                print(f"[校验通过] {filename} ({target.stat().st_size / 1024 / 1024:.2f} MB)")
                continue
            print(f"[重新下载] {filename} 本地 SHA-256 不匹配，正在重新拉取...")
        else:
            print(f"[下载中] {filename} <- {url}")

        download_with_progress(url, target)
        actual_sha = calc_sha256(target)
        if actual_sha != expected_sha:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"模型 {filename} SHA-256 校验失败: 期望 {expected_sha}，实际 {actual_sha}"
            )
        print(f"[校验通过] {filename} ({target.stat().st_size / 1024 / 1024:.2f} MB)")

    print("全部离线 PP-OCRv5 模型准备就绪！")


if __name__ == "__main__":
    ensure_models()
