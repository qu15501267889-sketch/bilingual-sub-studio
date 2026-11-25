import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_HYMT_PREFIX_RE = re.compile(r"^(?:translation|translated text|译文|翻译结果)\s*[:：]\s*", re.IGNORECASE)


class TranslateCancelled(Exception):
    """用户主动中止 AI 自动翻译时抛出的异常。"""


@dataclass
class TranslatorConfig:
    """内置 AI 字幕自动翻译配置（支持本地 Ollama 与云端 OpenAI 兼容 API）。"""

    provider: str = "ollama"  # "ollama" | "openai"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "hf.co/IndexTeam/Index-Translate-9B-GGUF:Q4_K_M"
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    api_model: str = "deepseek-chat"
    batch_size: int = 12
    temperature: float = 0.1
    skip_existing: bool = True
    unload_after_done: bool = True
    auto_detect_dialect: bool = True
    timeout_sec: int = 180
    max_retries: int = 2


@dataclass
class SrtBlock:
    """保持原序号与时间戳完整不变的 SRT 字幕条目。"""

    index: int
    start_str: str
    end_str: str
    text: str


# ==================== SRT 解析与原子回写 ====================


def parse_srt_blocks(srt_path: Path) -> list[SrtBlock]:
    """解析 SRT 文件并严格保留每条的原始序号与时间轴字符串。"""
    if not srt_path or not srt_path.exists():
        return []
    raw = srt_path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []

    blocks: list[SrtBlock] = []
    for chunk in re.split(r"\r?\n\s*\r?\n", raw):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        t_idx = 1 if "-->" in lines[1] else (0 if "-->" in lines[0] else -1)
        if t_idx < 0:
            continue
        parts = lines[t_idx].split("-->", 1)
        start_s = parts[0].strip()
        end_s = parts[1].strip()
        idx_val = int(lines[0]) if (t_idx == 1 and lines[0].isdigit()) else (len(blocks) + 1)
        text = "\n".join(lines[t_idx + 1 :]).strip()
        blocks.append(SrtBlock(index=idx_val, start_str=start_s, end_str=end_s, text=text))
    return blocks


def write_srt_blocks(blocks: list[SrtBlock], out_path: Path) -> None:
    """使用临时文件原子写入 UTF-8 SRT，确保不产生半截损坏文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.translating.srt")
    parts: list[str] = []
    for b in blocks:
        clean_txt = b.text.strip() or "..."
        parts.append(f"{b.index}\n{b.start_str} --> {b.end_str}\n{clean_txt}\n")
    tmp_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)


# ==================== HTTP 与本地/云端接口封装 ====================


def _build_opener(url: str) -> urllib.request.OpenerDirector:
    """访问本地回环地址时强制绕过系统代理，避免被 Clash/TUN 代理劫持报错。"""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _http_json(
    url: str,
    payload: Optional[dict] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 120,
    method: Optional[str] = None,
) -> dict:
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req_method = method or ("POST" if payload is not None else "GET")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=req_method)
    opener = _build_opener(url)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {err_body}") from exc


def _is_hymt_model(model_name: str) -> bool:
    low = (model_name or "").lower()
    return ("hy-mt" in low) or ("hunyuan-mt" in low)


def list_ollama_models(ollama_url: str = "http://127.0.0.1:11434") -> list[str]:
    """查询本机 Ollama 已安装模型列表，并将推荐的字幕翻译模型尽量排在前面。"""
    base = ollama_url.rstrip("/")
    data = _http_json(f"{base}/api/tags", payload=None, timeout=5, method="GET")
    names = [str(m.get("name", "")).strip() for m in data.get("models", []) if m.get("name")]

    def _rank(name: str) -> tuple[int, str]:
        low = name.lower()
        if "index-translate" in low:
            return (0, name)
        if "hy-mt" in low or "hunyuan-mt" in low:
            return (1, name)
        if low.startswith("qwen3:") or low.startswith("qwen2.5:"):
            return (2, name)
        return (3, name)

    return sorted(names, key=_rank)


def unload_ollama_model(ollama_url: str, model: str) -> bool:
    """向 Ollama 发送 keep_alive=0 立即释放显存，给后续 FFmpeg 硬编腾出充足显存。"""
    if not model:
        return False
    base = ollama_url.rstrip("/")
    try:
        _http_json(
            f"{base}/api/generate",
            payload={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=20,
        )
        return True
    except Exception:
        try:
            _http_json(
                f"{base}/api/chat",
                payload={"model": model, "messages": [], "keep_alive": 0, "stream": False},
                timeout=20,
            )
            return True
        except Exception:
            return False


def _call_chat_llm(
    cfg: TranslatorConfig,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
) -> str:
    """统一调用本地 Ollama (/api/chat) 或云端 OpenAI 兼容接口 (/chat/completions)。"""
    if cfg.provider == "openai":
        base = cfg.api_base_url.rstrip("/")
        if not base.endswith("/chat/completions"):
            endpoint = f"{base}/chat/completions"
        else:
            endpoint = base
        headers = {}
        if cfg.api_key.strip():
            headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"

        payload: dict = {
            "model": cfg.api_model.strip(),
            "temperature": float(cfg.temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = _http_json(endpoint, payload=payload, headers=headers, timeout=cfg.timeout_sec)
        except RuntimeError as exc:
            # 部分中转站/轻量模型不支持 response_format 参数时自动降级重试一次
            if expect_json and "400" in str(exc):
                payload.pop("response_format", None)
                resp = _http_json(endpoint, payload=payload, headers=headers, timeout=cfg.timeout_sec)
            else:
                raise
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 返回空 choices: {resp}")
        msg = choices[0].get("message") or {}
        return str(msg.get("content", ""))

    # 本地 Ollama
    base = cfg.ollama_url.rstrip("/")
    endpoint = f"{base}/api/chat"
    payload = {
        "model": cfg.ollama_model.strip(),
        "stream": False,
        "think": False,
        "options": {
            "temperature": float(cfg.temperature),
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 6144,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if expect_json:
        payload["format"] = "json"
    try:
        resp = _http_json(endpoint, payload=payload, timeout=cfg.timeout_sec)
    except RuntimeError as exc:
        if "think" in str(exc).lower() or "400" in str(exc):
            payload.pop("think", None)
            resp = _http_json(endpoint, payload=payload, timeout=cfg.timeout_sec)
        else:
            raise
    return str((resp.get("message") or {}).get("content", ""))


def _call_hymt_generate(
    cfg: TranslatorConfig,
    prompt: str,
) -> str:
    """腾讯 Hy-MT / Hunyuan-MT 系列专用 /api/generate 协议（无 System Prompt + 推荐采样参数）。"""
    base = cfg.ollama_url.rstrip("/")
    payload = {
        "model": cfg.ollama_model.strip(),
        "prompt": prompt,
        "stream": False,
        "options": {
            "top_k": 20,
            "top_p": 0.6,
            "repeat_penalty": 1.05,
            "temperature": float(cfg.temperature),
            "num_ctx": 6144,
        },
    }
    resp = _http_json(f"{base}/api/generate", payload=payload, timeout=cfg.timeout_sec)
    return str(resp.get("response", ""))


def test_translator_connection(cfg: TranslatorConfig) -> tuple[bool, str]:
    """检测本机 Ollama 或云端 API 是否连通，并执行 1 句轻量化字幕翻译握手。"""
    t0 = time.time()
    try:
        if cfg.provider == "ollama":
            models = list_ollama_models(cfg.ollama_url)
            if not models:
                return False, "Ollama 服务已连接，但本机还没有任何可用模型。"
            if cfg.ollama_model not in models:
                return False, f"未在本机 Ollama 中找到模型 `{cfg.ollama_model}`，请点击「刷新模型」选择已有模型。"

        if cfg.provider == "ollama" and _is_hymt_model(cfg.ollama_model):
            raw = _call_hymt_generate(cfg, "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n别扯犊子了，赶大集都散场了。")
            cleaned = _clean_single_translation(raw)
        else:
            raw = _call_chat_llm(
                cfg,
                "Translate the Chinese subtitle into natural short English. Return JSON: {\"en\": \"...\"}",
                "别扯犊子了，赶大集都散场了。",
                expect_json=True,
            )
            cleaned = _extract_single_from_raw(raw)

        if not cleaned or _ZH_CHAR_RE.search(cleaned):
            return False, f"模型已响应，但输出异常: {raw[:120]}"

        dt = time.time() - t0
        if cfg.provider == "ollama" and cfg.unload_after_done:
            unload_ollama_model(cfg.ollama_url, cfg.ollama_model)
        return True, f"连接正常 ({dt:.1f}s) | 示例译文: {cleaned}"
    except Exception as exc:
        return False, f"连接测试失败: {exc}"


# ==================== 输出清洗与严格结构校验 ====================


def _strip_thinking_and_fences(raw: str) -> str:
    txt = (raw or "").strip()
    txt = _THINK_BLOCK_RE.sub("", txt).strip()
    if "</think>" in txt:
        txt = txt.split("</think>")[-1].strip()
    txt = _CODE_FENCE_RE.sub("", txt).strip()
    return txt


def _clean_single_translation(text: str) -> str:
    t = _strip_thinking_and_fences(text)
    t = _HYMT_PREFIX_RE.sub("", t).strip()
    t = re.sub(r"\s*\n+\s*", " ", t).strip()
    if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        inner = t[1:-1].strip()
        if '"' not in inner:
            t = inner
    return t


def _extract_single_from_raw(raw: str) -> str:
    txt = _strip_thinking_and_fences(raw)
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            for k in ("en", "translation", "text"):
                if k in obj and isinstance(obj[k], str):
                    return _clean_single_translation(obj[k])
            items = obj.get("items") or obj.get("translations")
            if isinstance(items, list) and items and isinstance(items[0], dict):
                return _clean_single_translation(str(items[0].get("en", "")))
    except Exception:
        pass
    return _clean_single_translation(txt)


def _parse_json_items(raw: str, expected_ids: list[int]) -> tuple[dict[int, str], list[str]]:
    """
    从模型输出中提取 id -> en 映射，严格检查：
    1) JSON 可解析；2) 全部 expected_ids 齐备；3) 非空且不含残留中文字符。
    """
    cleaned = _strip_thinking_and_fences(raw)
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
            except Exception:
                obj = None

    if obj is None:
        return {}, [f"返回内容不是合法 JSON: {cleaned[:140]}"]

    mapping: dict[int, str] = {}
    if isinstance(obj, dict):
        items = obj.get("items") or obj.get("translations") or obj.get("subtitles") or obj.get("data")
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict) and "id" in entry:
                    try:
                        sid = int(entry["id"])
                    except Exception:
                        continue
                    val = entry.get("en") or entry.get("translation") or entry.get("text") or ""
                    mapping[sid] = _clean_single_translation(str(val))
        else:
            for k, v in obj.items():
                if str(k).isdigit():
                    mapping[int(k)] = _clean_single_translation(str(v))
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict) and "id" in entry:
                try:
                    sid = int(entry["id"])
                except Exception:
                    continue
                val = entry.get("en") or entry.get("translation") or entry.get("text") or ""
                mapping[sid] = _clean_single_translation(str(val))

    issues: list[str] = []
    valid_map: dict[int, str] = {}
    for sid in expected_ids:
        val = mapping.get(sid, "").strip()
        if not val:
            issues.append(f"id={sid} 缺失或为空")
            continue
        if _ZH_CHAR_RE.search(val):
            issues.append(f"id={sid} 仍包含中文字符({val[:24]})")
            continue
        valid_map[sid] = val

    return valid_map, issues


# ==================== 两阶段方言定调与滑动窗口精翻 ====================


def _extract_glossary_snippet(glossary_text: str, max_chars: int = 1600) -> str:
    txt = (glossary_text or "").strip()
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n..."


def _analyze_episode_style(
    cfg: TranslatorConfig,
    blocks: list[SrtBlock],
    glossary_text: str,
) -> dict:
    """阶段 A：先通读全片代表台词，判别年代、方言地域与本集核心专有名词统一译法。"""
    if not blocks:
        return {}
    if not cfg.auto_detect_dialect or (cfg.provider == "ollama" and _is_hymt_model(cfg.ollama_model)):
        return {}

    n = len(blocks)
    if n <= 36:
        sample = blocks
    else:
        step = max(1, n // 36)
        sample = blocks[::step][:36]

    sample_lines = "\n".join(f"[{b.index}] {b.text}" for b in sample)
    sys_p = (
        "You are a senior localization director for Chinese rural/retro/comedy short dramas.\n"
        "Analyze the dialogue lines and identify:\n"
        "1) dialect_region (e.g., 90s Northern Rural China, Northeast, Sichuan, Cantonese, Yunnan, Hunan/Hubei, or Modern Meme),\n"
        "2) era_tone (short English style note for dialogue),\n"
        "3) character_and_term_notes (consistent English translations for recurring names/terms in this episode, matching the glossary below).\n"
        "Return strictly JSON: {\"dialect_region\": str, \"era_tone\": str, \"term_notes\": str}."
    )
    usr_p = (
        f"Reference Glossary:\n{_extract_glossary_snippet(glossary_text, 1100)}\n\n"
        f"Sample Chinese Lines:\n{sample_lines}"
    )
    try:
        raw = _call_chat_llm(cfg, sys_p, usr_p, expect_json=True)
        obj = json.loads(_strip_thinking_and_fences(raw))
        if isinstance(obj, dict):
            return {
                "dialect_region": str(obj.get("dialect_region", "")),
                "era_tone": str(obj.get("era_tone", "")),
                "term_notes": str(obj.get("term_notes", "")),
            }
    except Exception:
        pass
    return {}


def _build_batch_prompts(
    batch: list[SrtBlock],
    prev_ctx: list[SrtBlock],
    next_ctx: list[SrtBlock],
    glossary_text: str,
    episode_style: dict,
    is_notes: bool,
    retry_hint: str = "",
) -> tuple[str, str]:
    style_hint = ""
    if episode_style:
        style_hint = (
            f"\nEpisode Context:\n"
            f"- Region/Era: {episode_style.get('dialect_region', '')}\n"
            f"- Dialogue Tone: {episode_style.get('era_tone', '')}\n"
            f"- Episode Terms: {episode_style.get('term_notes', '')}\n"
        )

    if is_notes:
        task_rule = (
            "You are translating ON-SCREEN POPUP CAPTIONS / BACKGROUND TEXT (*_notes.srt) from a Chinese short video into English.\n"
            "- Object labels/jokes (like “向日葵照明”): use ultra-short 2-5 English word screen tags.\n"
            "- Scene transition notes (like （不久之前）): use standard screen caption style like (A Little While Ago).\n"
            "- Keep each line super concise so viewers can read it in a glance.\n"
        )
    else:
        task_rule = (
            "You are translating SPOKEN SUBTITLES from a Chinese short drama into idiomatic, punchy American English.\n"
            "- CRITICAL OCR TYPO REPAIR STEP: Before translating each line, mentally fix obvious OCR homophone/visual lookalike typos using the surrounding lines and regional Chinese dialect pronunciation (e.g., 睫这豆子->咦这豆子, 自已人->自己人), making sure the restored Chinese is a natural, coherent spoken sentence.\n"
            "- Never translate dialect slang word-for-word; capture the humor, emotion, and colloquial punch in short single-line-friendly English.\n"
        )

    system_prompt = (
        f"{task_rule}"
        f"{style_hint}\n"
        f"Mandatory Glossary Rules:\n{_extract_glossary_snippet(glossary_text, 1400)}\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Translate ONLY the entries inside `translate_items`. Do NOT translate `context_before` or `context_after`.\n"
        "2. Return strictly a JSON object: {\"items\": [{\"id\": <int>, \"en\": \"<English>\"}]}.\n"
        "3. Every single `id` in `translate_items` MUST appear once with non-empty English.\n"
        "4. ZERO Chinese characters allowed in any `en` value. No notes or explanations."
    )
    if retry_hint:
        system_prompt += f"\n\n[PREVIOUS ATTEMPT HAD ISSUES - FIX THEM NOW]: {retry_hint}"

    user_payload = {
        "context_before": [b.text for b in prev_ctx],
        "translate_items": [{"id": b.index, "zh": b.text} for b in batch],
        "context_after": [b.text for b in next_ctx],
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False)


def _translate_single_line(
    cfg: TranslatorConfig,
    block: SrtBlock,
    prev_ctx: list[SrtBlock],
    next_ctx: list[SrtBlock],
    glossary_text: str,
    is_notes: bool,
) -> str:
    """单句精准兜底补译（用于批次重试仍遗漏或腾讯 Hy-MT 单句直译模式）。"""
    if cfg.provider == "ollama" and _is_hymt_model(cfg.ollama_model):
        gloss_hint = _extract_glossary_snippet(glossary_text, 600)
        ctx_lines = [b.text for b in prev_ctx] + [b.text for b in next_ctx]
        ctx_str = "\n".join(ctx_lines)
        kind_tip = "（画面极简短标签，2-5个英文单词）" if is_notes else "（影视口语对白，自然地道简洁）"
        prompt = ""
        if gloss_hint:
            prompt += f"参考下面的翻译术语表：\n{gloss_hint}\n"
        if ctx_str:
            prompt += f"{ctx_str}\n参考上面的语境信息，把下面的文本翻译为自然英文{kind_tip}，注意不需要翻译上文，只需要输出翻译后的结果，不要额外解释：\n{block.text}"
        else:
            prompt += f"将以下文本翻译为自然英文{kind_tip}，注意只需要输出翻译后的结果，不要额外解释：\n{block.text}"
        raw = _call_hymt_generate(cfg, prompt)
        return _clean_single_translation(raw)

    sys_p = (
        "Fix any obvious OCR typo by context and translate this single Chinese subtitle line into short, natural English. "
        f"Glossary:\n{_extract_glossary_snippet(glossary_text, 700)}\n"
        "Return JSON strictly: {\"en\": \"...\"}. Zero Chinese characters allowed."
    )
    usr_p = json.dumps(
        {
            "context_before": [b.text for b in prev_ctx],
            "target_zh": block.text,
            "context_after": [b.text for b in next_ctx],
        },
        ensure_ascii=False,
    )
    raw = _call_chat_llm(cfg, sys_p, usr_p, expect_json=True)
    return _extract_single_from_raw(raw)


def translate_srt_file(
    zh_srt_path: Path,
    en_srt_path: Path,
    cfg: TranslatorConfig,
    glossary_text: str,
    episode_style_cache: Optional[dict] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    item_progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[bool, dict]:
    """翻译单个 SRT 文件（保持原有序号与时间轴 100% 不变），返回 (是否成功, 统计信息)。"""
    log = log_cb or (lambda m: None)
    iprog = item_progress_cb or (lambda p, s: None)
    is_notes = zh_srt_path.stem.endswith("_notes")

    blocks = parse_srt_blocks(zh_srt_path)
    if not blocks:
        write_srt_blocks([], en_srt_path)
        return True, {"count": 0, "style": {}}

    use_hymt = (cfg.provider == "ollama") and _is_hymt_model(cfg.ollama_model)
    ep_style = episode_style_cache if episode_style_cache is not None else {}

    if (not is_notes) and (not ep_style) and (not use_hymt):
        iprog(3.0, "正在判别方言年代与剧情用词...")
        ep_style = _analyze_episode_style(cfg, blocks, glossary_text)
        if ep_style and ep_style.get("dialect_region"):
            log(f"[剧情定调] {zh_srt_path.name}: {ep_style.get('dialect_region')} | {ep_style.get('era_tone', '')[:60]}")

    total = len(blocks)
    translated_texts: dict[int, str] = {}
    t0 = time.time()

    if use_hymt:
        # Hy-MT 专用模型逐句带上下文与术语精准直译
        for idx, blk in enumerate(blocks):
            if cancel_event is not None and cancel_event.is_set():
                raise TranslateCancelled(f"用户中止翻译: {zh_srt_path.name}")
            prev_ctx = blocks[max(0, idx - 2) : idx]
            next_ctx = blocks[idx + 1 : min(total, idx + 3)]
            en_val = _translate_single_line(cfg, blk, prev_ctx, next_ctx, glossary_text, is_notes)
            if not en_val or _ZH_CHAR_RE.search(en_val):
                en_val = _translate_single_line(cfg, blk, [], [], glossary_text, is_notes)
            if not en_val or _ZH_CHAR_RE.search(en_val):
                raise RuntimeError(f"{zh_srt_path.name} 第 {blk.index} 条翻译失败或含残留中文: {en_val}")
            translated_texts[blk.index] = en_val
            pct = ((idx + 1) / total) * 100.0
            iprog(pct, f"正在翻译 {idx + 1}/{total} 条")
    else:
        bsize = max(4, min(25, int(cfg.batch_size)))
        for start_i in range(0, total, bsize):
            if cancel_event is not None and cancel_event.is_set():
                raise TranslateCancelled(f"用户中止翻译: {zh_srt_path.name}")

            end_i = min(total, start_i + bsize)
            batch = blocks[start_i:end_i]
            prev_ctx = blocks[max(0, start_i - 2) : start_i]
            next_ctx = blocks[end_i : min(total, end_i + 2)]
            expected_ids = [b.index for b in batch]

            batch_map: dict[int, str] = {}
            retry_hint = ""
            for attempt in range(cfg.max_retries + 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise TranslateCancelled(f"用户中止翻译: {zh_srt_path.name}")
                sys_p, usr_p = _build_batch_prompts(
                    batch, prev_ctx, next_ctx, glossary_text, ep_style, is_notes, retry_hint=retry_hint
                )
                raw = _call_chat_llm(cfg, sys_p, usr_p, expect_json=True)
                parsed, issues = _parse_json_items(raw, expected_ids)
                batch_map.update(parsed)
                if not issues:
                    break
                retry_hint = "; ".join(issues[:4])
                if attempt < cfg.max_retries:
                    log(f"[重试校验] {zh_srt_path.name} 第 {batch[0].index}~{batch[-1].index} 条 ({retry_hint})")

            # 对极少数批内仍缺失或残中文的条目执行单句补译兜底
            for b_pos, blk in enumerate(batch):
                if blk.index not in batch_map:
                    g_pos = start_i + b_pos
                    s_prev = blocks[max(0, g_pos - 2) : g_pos]
                    s_next = blocks[g_pos + 1 : min(total, g_pos + 3)]
                    for _ in range(2):
                        single_en = _translate_single_line(cfg, blk, s_prev, s_next, glossary_text, is_notes)
                        if single_en and not _ZH_CHAR_RE.search(single_en):
                            batch_map[blk.index] = single_en
                            break
                    if blk.index not in batch_map:
                        raise RuntimeError(
                            f"{zh_srt_path.name} 第 {blk.index} 条「{blk.text}」单句补译仍失败或含汉字"
                        )
                translated_texts[blk.index] = batch_map[blk.index]

            pct = (end_i / total) * 100.0
            iprog(pct, f"已校对翻译 {end_i}/{total} 条")

    out_blocks = [
        SrtBlock(
            index=b.index,
            start_str=b.start_str,
            end_str=b.end_str,
            text=translated_texts[b.index],
        )
        for b in blocks
    ]
    write_srt_blocks(out_blocks, en_srt_path)
    elapsed = time.time() - t0
    iprog(100.0, f"完成 ({total}条, {elapsed:.1f}s)")
    log(f"[翻译完成] {zh_srt_path.name} -> {en_srt_path.name} (共 {total} 条, 耗时 {elapsed:.1f}s)")
    return True, {"count": total, "elapsed": elapsed, "style": ep_style}


def _is_valid_existing_en_srt(zh_srt: Path, en_srt: Path) -> bool:
    if not en_srt.exists() or en_srt.stat().st_size == 0:
        return False
    zh_b = parse_srt_blocks(zh_srt)
    en_b = parse_srt_blocks(en_srt)
    if len(zh_b) == 0 or len(en_b) != len(zh_b):
        return False
    return not any(_ZH_CHAR_RE.search(b.text) for b in en_b)


def translate_batch(
    zh_dir: Path,
    en_dir: Path,
    cfg: TranslatorConfig,
    glossary_text: str,
    progress_cb: Optional[Callable[[int, int, float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    file_filter: Optional[list[str]] = None,
) -> dict:
    """批量自动翻译 zh_dir 中的中文 SRT 到 en_dir，同集先翻主台词再翻花字以共享方言定调。"""
    en_dir.mkdir(parents=True, exist_ok=True)
    log_file = en_dir / "_translate_log.txt"

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

    all_srts = sorted([p for p in zh_dir.glob("*.srt") if p.is_file()]) if zh_dir.exists() else []
    if file_filter:
        allow = set(file_filter)
        all_srts = [p for p in all_srts if p.name in allow]

    # 按同集排序：先主台词 (<stem>.srt)，紧接着花字 (<stem>_notes.srt)
    def _sort_key(p: Path) -> tuple[str, int]:
        if p.stem.endswith("_notes"):
            return (p.stem[: -len("_notes")], 1)
        return (p.stem, 0)

    srts = sorted(all_srts, key=_sort_key)
    total = len(srts)
    done = 0
    skipped = 0
    failed = 0
    cancelled = False

    model_label = cfg.ollama_model if cfg.provider == "ollama" else f"{cfg.api_model} ({cfg.api_base_url})"
    log(f"=== 开始批量 AI 自动翻译 (共 {total} 个字幕文件 | 引擎={cfg.provider} | 模型={model_label}) ===")

    episode_styles: dict[str, dict] = {}
    used_ollama = False

    try:
        for idx, zh_srt in enumerate(srts, 1):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            en_srt = en_dir / zh_srt.name
            ep_key = zh_srt.stem[: -len("_notes")] if zh_srt.stem.endswith("_notes") else zh_srt.stem

            if cfg.skip_existing and _is_valid_existing_en_srt(zh_srt, en_srt):
                log(f"[跳过] 已存在完整英文 SRT: {en_srt.name}")
                skipped += 1
                if progress_cb:
                    progress_cb(idx, total, 100.0, f"[{zh_srt.name}] 已跳过")
                continue

            if cfg.provider == "ollama":
                used_ollama = True

            if progress_cb:
                progress_cb(idx, total, 0.0, f"[{zh_srt.name}] 准备翻译...")

            try:
                def _item_prog(pct: float, stage: str):
                    if progress_cb:
                        progress_cb(idx, total, pct, f"[{zh_srt.name}] {stage}")

                ok, info = translate_srt_file(
                    zh_srt,
                    en_srt,
                    cfg,
                    glossary_text,
                    episode_style_cache=episode_styles.get(ep_key),
                    log_cb=log,
                    item_progress_cb=_item_prog,
                    cancel_event=cancel_event,
                )
                if info.get("style"):
                    episode_styles[ep_key] = info["style"]
                if ok:
                    done += 1
                else:
                    failed += 1
            except TranslateCancelled:
                cancelled = True
                log(f"[已取消] 翻译已在 {zh_srt.name} 处中止")
                break
            except Exception as exc:
                failed += 1
                log(f"[失败] 翻译 {zh_srt.name} 出错: {exc}")
    finally:
        if used_ollama and cfg.provider == "ollama" and cfg.unload_after_done:
            if unload_ollama_model(cfg.ollama_url, cfg.ollama_model):
                log(f"[显存已释放] 已卸载本地翻译模型 {cfg.ollama_model}")

    if not cancelled and progress_cb and total > 0:
        progress_cb(total, total, 100.0, "批量 AI 翻译全部完成")

    return {"total": total, "done": done, "skipped": skipped, "failed": failed, "cancelled": cancelled}
