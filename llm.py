import hashlib
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import requests

import config as _cfg

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 300
TIMEOUT_S = 30
MIN_CALL_GAP_S = 4.5
DAILY_LIMIT = 450
RETRY_WAIT_S = 2.0
USAGE_PATH = _cfg.RESULTS_DIR / "llm_usage.json"
NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

SYSTEM_PROMPT = (
    "You explain firewall-rule metrics to a network or security administrator. "
    "Use ONLY the numbers present in the JSON context (the same number as a "
    "percentage of 100 is allowed). Maximum 120 words. No new facts, no "
    "speculation. If the context is not enough to answer, reply exactly: "
    "insufficient evidence"
)


class LLMError(Exception):
    pass


def cache_key(context):
    payload = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def template_explanation(context):
    rule = context.get("rule_text", "unknown rule")
    parts = ["Deterministic template, no AI model was consulted."]
    parts.append("Rule: %s." % rule)
    for key in ("validation", "kddtest"):
        d = context["datasets"][key]
        parts.append(
            "%s: %d of %d attacks blocked (%.1f%% recall), %d of %d normal rows "
            "blocked (%.2f%% collateral), precision %.1f%%, verdict %s."
            % (
                d["name"],
                d["attacks_blocked"],
                d["attacks_total"],
                d["recall_pct"],
                d["normal_blocked"],
                d["normal_total"],
                d["collateral_pct"],
                d["precision_pct"],
                d["verdict"],
            )
        )
    return " ".join(parts)


def _flatten_numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))


def grounding_ok(text, context):
    nums = []
    _flatten_numbers(context, nums)
    forms = set()
    for c in nums:
        for f in (c, c * 100.0, c / 100.0):
            for d in (0, 1, 2):
                forms.add(round(f, d))
    for raw in NUMBER_RE.findall(text):
        val = float(raw.replace(",", ""))
        for d in (0, 1, 2):
            if round(val, d) in forms:
                break
        else:
            return False
    return True


def _usage_get(usage_path):
    try:
        with open(usage_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    if data.get("date") != date.today().isoformat():
        return 0
    return int(data.get("count", 0))


def _usage_add(usage_path):
    n = _usage_get(usage_path) + 1
    path = Path(usage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date.today().isoformat(), "count": n}, f)
    return n


def _scrub(text, api_key):
    s = str(text)
    if api_key:
        s = s.replace(api_key, "[redacted]")
    return s[:300]


def _post(model, api_key, system_prompt, user_payload, timeout, usage_path):
    url = API_URL.format(model=model)
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_payload}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    headers = {"x-goog-api-key": api_key}
    if usage_path is not None:
        _usage_add(usage_path)
    return requests.post(url, json=body, headers=headers, timeout=timeout)


def _parse_text(resp):
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise LLMError("empty text")
    return text


def _chain(api_key, system_prompt, user_payload, timeout, usage_path):
    last_reason = "unknown error"
    for model, label, retries in (
        (_cfg.GEMINI_MODEL, "primary", 1),
        (_cfg.GEMINI_FALLBACK_MODEL, "backup", 0),
    ):
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = _post(model, api_key, system_prompt, user_payload, timeout, usage_path)
            except requests.Timeout:
                last_reason = "timeout"
                if attempt <= retries:
                    time.sleep(RETRY_WAIT_S)
                    continue
                break
            except requests.RequestException as exc:
                last_reason = "connection error: %s" % _scrub(exc, api_key)
                break
            if resp.status_code == 200:
                try:
                    return _parse_text(resp), label, "ok"
                except Exception as exc:
                    last_reason = "bad response shape: %s" % _scrub(exc, api_key)
                    break
            if resp.status_code in (429, 503):
                last_reason = "http %d" % resp.status_code
                if attempt <= retries:
                    time.sleep(RETRY_WAIT_S)
                    continue
                break
            last_reason = "http %d" % resp.status_code
            break
    return None, None, last_reason


def explain(
    context,
    cache=None,
    api_key=None,
    timeout=TIMEOUT_S,
    usage_path=USAGE_PATH,
    min_gap=MIN_CALL_GAP_S,
):
    key = cache_key(context)
    if cache is not None and key in cache:
        return cache[key]
    if api_key is None:
        api_key = _cfg.GEMINI_API_KEY
    if not api_key:
        out = (template_explanation(context), "template", "missing API key")
    elif _usage_get(usage_path) >= DAILY_LIMIT:
        out = (template_explanation(context), "template", "daily limit reached")
    else:
        user_payload = json.dumps(context, sort_keys=True)
        now = time.monotonic()
        gap = getattr(explain, "_last_call_ts", None)
        if gap is not None and now - gap < min_gap:
            time.sleep(min_gap - (now - gap))
        explain._last_call_ts = time.monotonic()
        text, source, reason = _chain(
            api_key, SYSTEM_PROMPT, user_payload, timeout, usage_path
        )
        if source is None:
            out = (template_explanation(context), "template", reason)
        elif not grounding_ok(text, context):
            out = (template_explanation(context), "template", "grounding failed")
        else:
            out = (text, source, reason)
    if cache is not None:
        cache[key] = out
    return out
