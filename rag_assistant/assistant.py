"""RAG query pipeline.

Two-model approach (mirrors whisper-tray):
  - gpt-4o-mini       → fast answer with RAG context (cheap, <1s)
  - claude-sonnet-4-6 → deep hint / structured explanation (smarter, used sparingly)
"""

import os
import logging
from openai import OpenAI
import anthropic
from . import knowledge

log = logging.getLogger("rag_assistant")

_openai: OpenAI | None = None
_claude: anthropic.Anthropic | None = None

# ── prompts ───────────────────────────────────────────────────────────────────

_QUICK_SYSTEM = {
    "interview": (
        "You are a real-time assistant helping during a job interview. "
        "Use the provided context from the candidate's CV and knowledge base. "
        "Give a concise, confident answer in 2-3 sentences. "
        "If the context is relevant, use it. Otherwise answer from general knowledge. "
        "Answer in Russian."
    ),
    "german": (
        "You are a German language assistant. "
        "Use the provided grammar rules and vocabulary from the knowledge base. "
        "Give a short, clear answer with one example. Answer in Russian."
    ),
}

_HINT_SYSTEM = {
    "interview": (
        "You are a smart real-time assistant for a job interview. "
        "You receive a question or topic from the interviewer.\n\n"
        "Use the provided knowledge base context if relevant.\n\n"
        "Format your response exactly like this:\n"
        "▶ <Topic in Russian>\n"
        "<brief explanation, 3-5 bullet points>\n"
        "<short code example if applicable>\n\n"
        "If the input has no useful content, reply with just: -\n"
        "Always answer in Russian. "
        "Stack: React, Next.js, TypeScript, Redux, RTK Query, Node.js, .NET, Python, FastAPI, PostgreSQL, Docker, Azure."
    ),
    "german": (
        "You are a German language tutor assistant.\n\n"
        "Use the provided grammar/vocabulary context if relevant.\n\n"
        "Format your response:\n"
        "▶ <тема>\n"
        "<правило или объяснение>\n"
        "<2-3 примера с переводом>\n\n"
        "If nothing useful to add, reply with just: -\n"
        "Always answer in Russian."
    ),
}

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
    chunks = knowledge.search(mode, text, n=n_chunks)
    context = _build_context(chunks)

    if context:
        user_msg = f"Knowledge base context:\n{context}\n\nQuestion: {text}"
    else:
        user_msg = text

    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _QUICK_SYSTEM.get(mode, _QUICK_SYSTEM["interview"])},
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

    chunks = knowledge.search(mode, text, n=n_chunks)
    context = _build_context(chunks)

    if context:
        user_msg = f"Knowledge base context:\n{context}\n\nSpeech/question: {text}"
    else:
        user_msg = text

    try:
        msg = _get_claude().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=_HINT_SYSTEM.get(mode, _HINT_SYSTEM["interview"]),
            messages=[{"role": "user", "content": user_msg}],
        )
        result = msg.content[0].text.strip()
        return "" if result == "-" else result
    except Exception as e:
        log.error("Claude hint error: %s", e)
        return ""
