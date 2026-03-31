#!/usr/bin/env python3
"""RAG Assistant — interactive CLI.

Usage:
    python main.py                      # interview mode
    python main.py --mode german
    python main.py --stats
"""

import os
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rag_assistant import knowledge, assistant

load_dotenv()
console = Console()


def show_stats():
    stats = knowledge.stats()
    for col, count in stats.items():
        console.print(f"  [cyan]{col}[/cyan]: {count} chunks")


def interactive(mode: str):
    console.print(Panel(
        f"[bold]RAG Assistant[/bold] — mode: [cyan]{mode}[/cyan]\n"
        "Type your question. [dim]Ctrl+C to exit.[/dim]",
        border_style="blue",
    ))

    stats = knowledge.stats()
    count = stats.get(mode, 0)
    if count == 0:
        console.print(f"[yellow]Warning: '{mode}' knowledge base is empty. Run: python ingest.py[/yellow]\n")

    while True:
        try:
            question = console.input("[bold green]>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not question:
            continue

        with console.status("[dim]Searching...[/dim]", spinner="dots"):
            result = assistant.query(question, mode=mode)

        # Show answer
        console.print(Panel(
            result["answer"],
            title="[bold]Answer[/bold]",
            border_style="green",
        ))

        # Show retrieved context (collapsed)
        if result["context"]:
            console.print("[dim]Context used:[/dim]")
            for c in result["context"]:
                score = c["score"]
                topic = c["meta"].get("topic", "")
                preview = c["text"][:80].replace("\n", " ")
                console.print(f"  [dim]{score:.2f}  [{topic}] {preview}...[/dim]")
        console.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interview", "german"], default="interview")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    interactive(args.mode)


if __name__ == "__main__":
    main()
