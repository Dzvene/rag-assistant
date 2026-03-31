#!/usr/bin/env python3
"""Load knowledge base files into ChromaDB.

File lists are defined in prompts/<mode>.yaml under the `files:` key.
Paths are relative to knowledge_base/personal/.

Usage:
    python ingest.py                        # ingest all modes
    python ingest.py --mode german          # ingest one mode
    python ingest.py --mode german --reset  # clear and re-ingest
"""

import os
import re
import argparse
import hashlib
import yaml
from rag_assistant import knowledge
from rag_assistant.knowledge import available_modes

KB_ROOT    = os.path.join(os.path.dirname(__file__), "knowledge_base", "personal")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_mode_config(mode: str) -> dict:
    with open(os.path.join(PROMPTS_DIR, f"{mode}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chunk_markdown(text: str, source: str) -> list[dict]:
    chunks = []
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        topic = lines[0].strip() if len(lines) > 1 else f"section_{i}"
        body  = lines[1].strip() if len(lines) > 1 else lines[0].strip()
        if len(body) < 10:
            continue
        chunk_id = hashlib.md5(f"{source}:{topic}".encode()).hexdigest()[:16]
        chunks.append({
            "id":   chunk_id,
            "text": f"{topic}\n{body}",
            "meta": {"source": source, "topic": topic},
        })
    return chunks


def ingest_mode(mode: str, reset: bool = False):
    cfg = _load_mode_config(mode)
    files = cfg.get("files") or []

    if reset:
        knowledge.delete_collection(mode)
        print(f"  Cleared '{mode}'")

    total = 0
    for rel_path in files:
        full_path = os.path.join(KB_ROOT, rel_path)
        if not os.path.isfile(full_path):
            print(f"  [skip] {rel_path} — not found")
            continue
        with open(full_path, encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_markdown(text, rel_path)
        if not chunks:
            print(f"  [skip] {rel_path} — no chunks")
            continue
        knowledge.add_chunks(mode, chunks)
        print(f"  [ok]   {rel_path} → {len(chunks)} chunks")
        total += len(chunks)

    print(f"  Total: {total} chunks in '{mode}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=available_modes(), default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    targets = [args.mode] if args.mode else available_modes()
    for mode in targets:
        print(f"\n[{mode}]")
        ingest_mode(mode, reset=args.reset)

    print("\nStats:", knowledge.stats())


if __name__ == "__main__":
    main()
