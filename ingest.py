#!/usr/bin/env python3
"""Load markdown files from knowledge_base/ into ChromaDB.

Usage:
    python ingest.py                    # load all
    python ingest.py --collection german
    python ingest.py --reset german     # clear and reload
"""

import os
import re
import sys
import argparse
import hashlib
from rag_assistant import knowledge

KB_ROOT = os.path.join(os.path.dirname(__file__), "knowledge_base", "personal")

COLLECTION_DIRS = {
    "interview": ["cv.md", "tech_stack.md"],
    "german":    ["german/grammar.md", "german/vocabulary.md"],
}


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split markdown by ## headers into chunks."""
    chunks = []
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        topic = lines[0].strip() if len(lines) > 1 else f"section_{i}"
        body = lines[1].strip() if len(lines) > 1 else lines[0].strip()

        if len(body) < 10:
            continue

        chunk_id = hashlib.md5(f"{source}:{topic}".encode()).hexdigest()[:16]
        chunks.append({
            "id": chunk_id,
            "text": f"{topic}\n{body}",
            "meta": {"source": source, "topic": topic},
        })

    return chunks


def ingest_collection(name: str, reset: bool = False):
    if reset:
        try:
            knowledge.delete_collection(name)
            print(f"  Cleared '{name}'")
        except Exception:
            pass

    files = COLLECTION_DIRS.get(name, [])
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

        knowledge.add_chunks(name, chunks)
        print(f"  [ok]   {rel_path} → {len(chunks)} chunks")
        total += len(chunks)

    print(f"  Total: {total} chunks in '{name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", choices=list(COLLECTION_DIRS), default=None)
    parser.add_argument("--reset", metavar="COLLECTION", default=None)
    args = parser.parse_args()

    if args.reset:
        print(f"Resetting and reingesting '{args.reset}'...")
        ingest_collection(args.reset, reset=True)
        return

    targets = [args.collection] if args.collection else list(COLLECTION_DIRS)
    for name in targets:
        print(f"\n[{name}]")
        ingest_collection(name)

    print("\nDone. Stats:", knowledge.stats())


if __name__ == "__main__":
    main()
