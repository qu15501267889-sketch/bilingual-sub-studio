import atexit
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from core.runtime_paths import (
    get_creation_flags,
    get_embedded_ollama_dir,
    get_embedded_ollama_exe,
    get_embedded_ollama_models_dir,
    get_llm_drop_dir,
    run_silent,
)

_MANAGED_PROC: Optional[subprocess.Popen] = None
_MANAGED_URL: str = ""

_CHATML_THINK_TEMPLATE = '''"""{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "system" }}{{ else if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{- end }}<|im_start|>assistant
{{- if and .IsThinkSet (not .Think) }}<think>

</think>

{{ end }}"""'''


def _build_direct_opener() -> urllib.request.OpenerDirector:
    """回环地址强制绕过系统代理直连，防止 TUN/Clash 代理拦截导致本地请求超时或 502。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_ollama_alive(url: str = "http://127.0.0.1:11434", timeout: float = 1.2) -> bool:
    """检测指定地址的 Ollama 服务是否真实在线并响应 /api/version。"""
    base = (url or "http://127.0.0.1:11434").rstrip("/")
    req = urllib.request.Request(f"{base}/api/version", method="GET")
    try:
        with _build_direct_opener().open(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _build_ollama_child_env(
    host_bind: str,
    models_dir: Path,
    ollama_dir: Path,
) -> dict[str, str]:
    """构造隔离且跨硬件自适配的 Ollama 子进程环境变量（清 PATH 下也能正常找到自身 DLL）。"""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host_bind
    env["OLLAMA_MODELS"] = str(models_dir.resolve())

    # 将内置 Ollama 根目录及 lib/ollama 置于子进程 PATH 最前
    lib_sub = ollama_dir / "lib" / "ollama"
    extra_paths = [str(ollama_dir.resolve())]
    if lib_sub.is_dir():
        extra_paths.append(str(lib_sub.resolve()))
    cur_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(extra_paths + ([cur_path] if cur_path else []))

    # 若未打包 CUDA 驱动仅保留了 Vulkan/CPU 时，显式允许 Vulkan 加速支持 AMD/Intel 显卡
    if not (lib_sub / "cuda_v12").is_dir() and (lib_sub / "vulkan").is_dir():
        env.setdefault("OLLAMA_VULKAN", "1")
    return env


def ensure_ollama_running(
    preferred_url: str = "http://127.0.0.1:11434",
    force_embedded_port: Optional[int] = None,
    models_dir_override: Optional[Path] = None,
) -> dict:
    """
    确保 Ollama 推理服务可用：
    - 常规模式：若目标端口已在线则直接复用外置或当前已拉起服务；
    - 否则若存在绿色包内置 `runtime/ollama/ollama.exe`，自动静默启动内置引擎。
    """
    global _MANAGED_PROC, _MANAGED_URL

    parsed = urllib.parse.urlparse((preferred_url or "http://127.0.0.1:11434").strip())
    host = parsed.hostname or "127.0.0.1"
    port = int(force_embedded_port) if force_embedded_port is not None else (parsed.port or 11434)
    target_url = f"http://{host}:{port}"

    if force_embedded_port is None and is_ollama_alive(target_url, timeout=1.2):
        mode = "embedded_active" if (_MANAGED_PROC is not None and _MANAGED_PROC.poll() is None and _MANAGED_URL == target_url) else "external_active"
        return {
            "running": True,
            "mode": mode,
            "url": target_url,
            "has_embedded": get_embedded_ollama_exe() is not None,
        }

    emb_exe = get_embedded_ollama_exe()
    if emb_exe is None:
        return {
            "running": False,
            "mode": "no_embedded_and_offline",
            "url": target_url,
            "has_embedded": False,
            "error": "本机未检测到运行中的 Ollama，且当前版本（标准轻量版）未内置 Ollama。",
        }

    # 如仅指定端口测试且先清理已终止句柄
    if _MANAGED_PROC is not None and _MANAGED_PROC.poll() is None and _MANAGED_URL == target_url:
        return {"running": True, "mode": "embedded_active", "url": target_url, "has_embedded": True}

    models_dir = models_dir_override or get_embedded_ollama_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    ollama_dir = get_embedded_ollama_dir()
    env = _build_ollama_child_env(f"{host}:{port}", models_dir, ollama_dir)

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "cwd": str(ollama_dir),
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = get_creation_flags()

    proc = subprocess.Popen([str(emb_exe), "serve"], **kwargs)
    _MANAGED_PROC = proc
    _MANAGED_URL = target_url

    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _MANAGED_PROC = None
            return {
                "running": False,
                "mode": "embedded_exited",
                "url": target_url,
                "has_embedded": True,
                "error": f"内置 Ollama 启动后异常退出 (exit={proc.returncode})",
            }
        if is_ollama_alive(target_url, timeout=0.8):
            return {
                "running": True,
                "mode": "embedded_spawned",
                "url": target_url,
                "pid": proc.pid,
                "has_embedded": True,
            }
        time.sleep(0.35)

    stop_embedded_ollama()
    return {
        "running": False,
        "mode": "embedded_timeout",
        "url": target_url,
        "has_embedded": True,
        "error": f"等待内置 Ollama 就绪超时 ({target_url})",
    }


def stop_embedded_ollama() -> None:
    """退出时仅优雅关闭由本软件启动的内置 Ollama 托管子进程，绝不误杀外部用户进程。"""
    global _MANAGED_PROC, _MANAGED_URL
    proc = _MANAGED_PROC
    _MANAGED_PROC = None
    _MANAGED_URL = ""
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


atexit.register(stop_embedded_ollama)


def infer_model_tag_and_modelfile(gguf_path: Path) -> tuple[str, str]:
    """
    根据放到 models/llm/ 中的 GGUF 文件名，自动推断规范的 Ollama 模型名与最佳配套 Modelfile。
    跳过 *mmproj* 多模态视觉投影附件。
    """
    fname = gguf_path.name
    low = fname.lower()
    if "mmproj" in low:
        raise ValueError(f"跳过视觉投影附件文件（非独立文本模型）: {fname}")

    stem = gguf_path.stem
    abs_gguf = str(gguf_path.resolve())

    # 匹配常见量化后缀
    quant_match = re.search(r"[-.](IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+|Q8_0|F16|BF16)$", stem, re.IGNORECASE)
    quant_tag = quant_match.group(1).lower() if quant_match else "latest"
    base_clean = stem[: quant_match.start()] if quant_match else stem
    norm_base = re.sub(r"[^a-z0-9]+", "-", base_clean.lower()).strip("-") or "local-llm"

    if "index-translate" in low:
        if "q4_k_m" in low:
            tag = "hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M"
        else:
            tag = f"{norm_base}:{quant_tag}"
        modelfile = (
            f"FROM {abs_gguf}\n"
            f"TEMPLATE {_CHATML_THINK_TEMPLATE}\n"
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER stop "<|endoftext|>"\n'
            "PARAMETER temperature 0.1\n"
            "PARAMETER top_p 0.9\n"
            "PARAMETER repeat_penalty 1.05\n"
            "PARAMETER num_ctx 6144\n"
        )
        return tag, modelfile

    if "hy-mt" in low or "hunyuan-mt" in low:
        tag = f"{norm_base}:{quant_tag}"
        modelfile = (
            f"FROM {abs_gguf}\n"
            "PARAMETER temperature 0.1\n"
            "PARAMETER top_k 20\n"
            "PARAMETER top_p 0.6\n"
            "PARAMETER repeat_penalty 1.05\n"
            "PARAMETER num_ctx 6144\n"
        )
        return tag, modelfile

    tag = f"{norm_base}:{quant_tag}"
    if "qwen3" in low:
        modelfile = (
            f"FROM {abs_gguf}\n"
            f"TEMPLATE {_CHATML_THINK_TEMPLATE}\n"
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER stop "<|endoftext|>"\n'
            "PARAMETER temperature 0.1\n"
            "PARAMETER top_p 0.9\n"
            "PARAMETER repeat_penalty 1.05\n"
            "PARAMETER num_ctx 6144\n"
        )
    else:
        modelfile = (
            f"FROM {abs_gguf}\n"
            "PARAMETER temperature 0.1\n"
            "PARAMETER repeat_penalty 1.05\n"
            "PARAMETER num_ctx 6144\n"
        )
    return tag, modelfile


def list_drop_gguf_files() -> list[Path]:
    """列出 models/llm 目录下所有可独立导入的 *.gguf 文件（自动过滤 *mmproj*）。"""
    drop_dir = get_llm_drop_dir()
    if not drop_dir.is_dir():
        return []
    return [p for p in sorted(drop_dir.glob("*.gguf")) if p.is_file() and "mmproj" not in p.name.lower()]


def _model_exists_in_list(tag: str, existing: list[str]) -> bool:
    ex_low = {m.lower() for m in existing}
    t_low = tag.lower()
    if t_low in ex_low:
        return True
    if ":" not in t_low and f"{t_low}:latest" in ex_low:
        return True
    return False


def has_unimported_local_ggufs(existing_models: list[str]) -> list[Path]:
    """检查 models/llm/ 中是否存在尚未注册到当前 Ollama 服务的本地 .gguf 文件。"""
    pending: list[Path] = []
    for p in list_drop_gguf_files():
        try:
            tag, _ = infer_model_tag_and_modelfile(p)
            if not _model_exists_in_list(tag, existing_models):
                pending.append(p)
        except Exception:
            pass
    return pending


def scan_and_import_local_ggufs(
    ollama_url: str = "http://127.0.0.1:11434",
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    扫描绿色包 models/llm/*.gguf 并一键调用 `ollama create` 自动注册导入（跳过已存在模型）。
    返回 {"imported": [tag...], "skipped": [tag...], "failed": [(name, err)...], "ollama_state": ...}
    """
    from core.translator import list_ollama_models

    log = log_cb or (lambda m: None)
    state = ensure_ollama_running(ollama_url)
    if not state.get("running"):
        err = state.get("error") or "Ollama 服务未运行且无内置运行时"
        return {"imported": [], "skipped": [], "failed": [("Ollama服务", err)], "ollama_state": state}

    emb_exe = get_embedded_ollama_exe()
    sys_exe = shutil.which("ollama")
    ollama_bin = str(emb_exe) if emb_exe else (sys_exe or "")
    if not ollama_bin:
        return {
            "imported": [],
            "skipped": [],
            "failed": [("ollama.exe", "找不到内置或系统 ollama.exe 可执行文件")],
            "ollama_state": state,
        }

    try:
        existing = list_ollama_models(ollama_url)
    except Exception:
        existing = []

    ggufs = list_drop_gguf_files()
    imported: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    parsed = urllib.parse.urlparse((ollama_url or "http://127.0.0.1:11434").strip())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    child_env = os.environ.copy()
    child_env["OLLAMA_HOST"] = f"{host}:{port}"

    for p in ggufs:
        try:
            tag, mf_text = infer_model_tag_and_modelfile(p)
        except Exception as exc:
            failed.append((p.name, str(exc)))
            continue

        if _model_exists_in_list(tag, existing):
            skipped.append(tag)
            continue

        log(f"[导入模型] 正在导入 {p.name} -> {tag} ...")
        tmp_mf: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, encoding="utf-8") as tf:
                tf.write(mf_text)
                tmp_mf = Path(tf.name)
            res = run_silent(
                [ollama_bin, "create", tag, "-f", str(tmp_mf)],
                env=child_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=360,
            )
            if res.returncode == 0:
                imported.append(tag)
                existing.append(tag)
                log(f"[导入成功] {tag}")
            else:
                err_msg = (res.stderr or res.stdout or f"exit={res.returncode}").strip()[-300:]
                failed.append((p.name, err_msg))
                log(f"[导入失败] {p.name}: {err_msg}")
        except Exception as exc:
            failed.append((p.name, str(exc)))
            log(f"[导入异常] {p.name}: {exc}")
        finally:
            if tmp_mf is not None:
                try:
                    tmp_mf.unlink()
                except Exception:
                    pass

    return {"imported": imported, "skipped": skipped, "failed": failed, "ollama_state": state}
