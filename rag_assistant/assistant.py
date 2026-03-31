"""RAG query pipeline.

Two-model approach (mirrors whisper-tray):
  - gpt-4o-mini       → fast answer with RAG context (cheap, <1s)
  - claude-sonnet-4-6 → deep hint / structured explanation (smarter, used sparingly)

Prompts are loaded from prompts/<mode>.yaml — edit those files to customize behavior.
"""

import os
import logging
import yaml
from openai import OpenAI
import anthropic
from . import knowledge

log = logging.getLogger("rag_assistant")

_openai: OpenAI | None = None
_claude: anthropic.Anthropic | None = None

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
_prompts_cache: dict[str, dict] = {}

# ── prompt loader ─────────────────────────────────────────────────────────────

def _load_prompts(mode: str) -> dict:
    if mode in _prompts_cache:
        return _prompts_cache[mode]

    path = os.path.join(_PROMPTS_DIR, f"{mode}.yaml")
    if not os.path.isfile(path):
        log.warning("Prompt file not found: %s — using empty prompts", path)
        return {"quick_system": "", "hint_system": ""}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _prompts_cache[mode] = data
    return data


def reload_prompts():
    """Clear cache so prompts are reloaded from disk on next call."""
    _prompts_cache.clear()

# ── clients ───────────────────────────────────────────────────────────────────

def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai


def _get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _claude

# ── context builder ───────────────────────────────────────────────────────────

def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(
        f"[{c['meta'].get('topic', 'info')}]\n{c['text']}"
        for c in chunks
    )

# ── public API ────────────────────────────────────────────────────────────────

def query(text: str, mode: str = "interview", n_chunks: int = 4) -> dict:
    """Fast RAG answer via gpt-4o-mini.

    Returns: {"answer": str, "context": list[dict], "mode": str}
    """
    prompts = _load_prompts(mode)
    chunks = knowledge.search(mode, text, n=n_chunks)
    context = _build_context(chunks)

    user_msg = f"Knowledge base context:\n{context}\n\nQuestion: {text}" if context else text

    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompts["quick_system"]},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("OpenAI query error: %s", e)
        answer = f"[Error: {e}]"

    return {"answer": answer, "context": chunks, "mode": mode}


def hint(text: str, mode: str = "interview", n_chunks: int = 4) -> str:
    """Deep structured hint via claude-sonnet-4-6.

    Returns hint string, or "" if nothing useful.
    """
    if not text or len(text) < 15:
        return ""

    prompts = _load_prompts(mode)
    chunks = knowledge.search(mode, text, n=n_chunks)
    context = _build_context(chunks)

    user_msg = f"Knowledge base context:\n{context}\n\nSpeech/question: {text}" if context else text

    try:
        msg = _get_claude().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=prompts["hint_system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        result = msg.content[0].text.strip()
        return "" if result == "-" else result
    except Exception as e:
        log.error("Claude hint error: %s", e)
        return ""
