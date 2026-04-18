import json
import os
import re
from typing import Any, List, Dict

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. "
    "For valid tool requests, output exactly one tool call as: "
    "<tool_call>{\"tool\":...,\"args\":...}</tool_call>. "
    "For refusals, output plain text with no tool call."
)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/quantized/model.gguf")
_LLM = None
VALID_TOOLS = {"weather", "calendar", "convert", "currency", "sql"}
REFUSAL_TEXT = "I can only help with weather, calendar, conversion, currency, or SQL tools."
AMBIGUOUS_HINTS = {"that", "it", "same", "there", "previous", "above", "wahi", "us"}


def format_chatml(messages: List[Dict], add_generation_prompt: bool) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def get_llm() -> Any:
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _LLM = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=max(1, os.cpu_count() or 4),
            n_batch=256,
            seed=42,
            verbose=False,
        )
    return _LLM


def last_tool_call(history: List[Dict]) -> str:
    for item in reversed(history or []):
        content = item.get("content", "") if isinstance(item, dict) else ""
        if "<tool_call>" in content:
            return content
    return ""


def should_refuse_without_model(prompt: str, history: List[Dict]) -> bool:
    p = prompt.lower().strip()
    if any(token in p.split() for token in AMBIGUOUS_HINTS):
        if not last_tool_call(history):
            return True

    out_of_scope_phrases = [
        "tell me a joke",
        "write a poem",
        "book a flight",
        "set alarm",
        "capital of",
    ]
    if any(phrase in p for phrase in out_of_scope_phrases):
        return True
    return False


def extract_json_tool_call(text: str):
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def to_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return None
    return None


def canonical_tool_call(payload):
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    args = payload.get("args")
    if tool not in VALID_TOOLS or not isinstance(args, dict):
        return None

    clean = {"tool": tool, "args": {}}
    if tool == "weather":
        location = args.get("location")
        unit = str(args.get("unit", "")).upper()
        if not isinstance(location, str) or not location.strip() or unit not in {"C", "F"}:
            return None
        clean["args"] = {"location": location.strip(), "unit": unit}
    elif tool == "calendar":
        action = str(args.get("action", "")).lower()
        date = args.get("date")
        if action not in {"list", "create"} or not isinstance(date, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            return None
        if action == "create":
            title = args.get("title")
            if not isinstance(title, str) or not title.strip():
                return None
            clean["args"] = {"action": action, "date": date, "title": title.strip()}
        else:
            clean["args"] = {"action": action, "date": date}
    elif tool == "convert":
        value = to_number(args.get("value"))
        from_unit = args.get("from_unit")
        to_unit = args.get("to_unit")
        if value is None or not isinstance(from_unit, str) or not isinstance(to_unit, str):
            return None
        clean["args"] = {"value": value, "from_unit": from_unit.strip(), "to_unit": to_unit.strip()}
    elif tool == "currency":
        amount = to_number(args.get("amount"))
        from_code = str(args.get("from", "")).upper()
        to_code = str(args.get("to", "")).upper()
        if amount is None or not re.match(r"^[A-Z]{3}$", from_code) or not re.match(r"^[A-Z]{3}$", to_code):
            return None
        clean["args"] = {"amount": amount, "from": from_code, "to": to_code}
    elif tool == "sql":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return None
        clean["args"] = {"query": query.strip()}
    else:
        return None

    return "<tool_call>" + json.dumps(clean, ensure_ascii=True, separators=(",", ":")) + "</tool_call>"


def run(prompt: str, history: List[Dict]) -> str:
    if should_refuse_without_model(prompt, history or []):
        return REFUSAL_TEXT

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    text = format_chatml(messages, add_generation_prompt=True)
    llm = get_llm()

    out = llm(
        text,
        max_tokens=96,
        temperature=0.0,
        top_p=0.9,
        stop=["<|im_end|>"],
    )
    result = out["choices"][0]["text"]

    payload = extract_json_tool_call(result)
    if payload is not None:
        clean = canonical_tool_call(payload)
        if clean is not None:
            return clean
        return REFUSAL_TEXT

    text_out = result.strip()
    return text_out if text_out else REFUSAL_TEXT
