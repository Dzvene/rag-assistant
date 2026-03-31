"""RAG query pipeline — retrieve context, call LLM."""

import os
from openai import OpenAI
from . import knowledge

_client: OpenAI | None = None

SYSTEM_PROMPTS = {
    "interview": (
        "You are a real-time assistant helping during a job interview. "
        "Use the provided context from the candidate's CV and knowledge base. "
        "Give concise, confident answers in 2-3 sentences max. "
        "If the context is relevant, use it. Otherwise answer from general knowledge."
    ),
    "german": (
        "You are a German language assistant. "
        "Use the provided grammar rules and vocabulary from the knowledge base. "
        "Explain clearly and give examples. Answer in Russian unless asked otherwise."
    ),
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def query(text: str, mode: str = "interview", n_chunks: int = 4) -> dict:
    """
    Search knowledge base and ask LLM with retrieved context.

    Returns: {"answer": str, "context": list[dict], "mode": str}
    """
    chunks = knowledge.search(mode, text, n=n_chunks)

    context_block = ""
    if chunks:
        context_block = "\n\n".join(
            f"[{c['meta'].get('topic', 'info')}]\n{c['text']}"
            for c in chunks
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["interview"])}]

    if context_block:
        messages.append({
            "role": "user",
            "content": f"Relevant context from knowledge base:\n{context_block}\n\nQuestion: {text}",
        })
    else:
        messages.append({"role": "user", "content": text})

    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "context": chunks,
        "mode": mode,
    }
