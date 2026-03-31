#!/usr/bin/env python3
"""RAG Assistant — interactive CLI.

Usage:
    python main.py                      # interview mode
    python main.py --mode german
    python main.py --stats

Commands during session:
    <question>          → fast answer (gpt-4o-mini + RAG)
    !<question>         → deep hint   (claude-sonnet-4-6 + RAG)
    /mode interview     → switch mode
    /mode german
    /stats              → chunk counts
    /quit
"""

import os
import argparse
import threading
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rag_assistant import knowledge, assistant

load_dotenv()
logging.basicConfig(level=logging.WARNING)
console = Console()


def show_stats():
    stats = knowledge.stats()
    for col, count in stats.items():
        status = "[green]ok[/green]" if count > 0 else "[yellow]empty[/yellow]"
        console.print(f"  [cyan]{col}[/cyan]: {count} chunks {status}")


def _print_answer(result: dict):
    console.print(Panel(
        result["answer"],
        title="[bold green]Answer[/bold green]  [dim]gpt-4o-mini[/dim]",
        border_style="green",
    ))
    if result["context"]:
        console.print("[dim]Context:[/dim]")
        for c in result["context"]:
            topic = c["meta"].get("topic", "")
            preview = c["text"][:80].replace("\n", " ")
            console.print(f"  [dim]{c['score']:.2f}  [{topic}] {preview}[/dim]")
    console.print()


def _print_hint(text: str):
    if text:
        console.print(Panel(
            text,
            title="[bold yellow]Hint[/bold yellow]  [dim]claude-sonnet-4-6[/dim]",
            border_style="yellow",
        ))
    else:
        console.print("[dim]No hint generated.[/dim]")
    console.print()


def interactive(mode: str):
    console.print(Panel(
        f"[bold]RAG Assistant[/bold] — mode: [cyan]{mode}[/cyan]\n"
        "[dim]<question>[/dim]   → fast answer (gpt-4o-mini)\n"
        "[dim]!<question>[/dim]  → deep hint   (claude-sonnet-4-6)\n"
        "[dim]/mode interview|german  /stats  /quit[/dim]",
        border_style="blue",
    ))

    stats = knowledge.stats()
    if stats.get(mode, 0) == 0:
        console.print(f"[yellow]Warning: '{mode}' knowledge base is empty. Run: python ingest.py[/yellow]\n")

    while True:
        try:
            line = console.input(f"[bold blue]{mode}[/bold blue] [bold green]>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not line:
            continue

        # built-in commands
        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0]
            if cmd == "/quit":
                console.print("[dim]Bye.[/dim]")
                break
            elif cmd == "/stats":
                show_stats()
            elif cmd == "/mode" and len(parts) > 1 and parts[1] in ("interview", "german"):
                mode = parts[1]
                console.print(f"[dim]Switched to [cyan]{mode}[/cyan][/dim]\n")
                if knowledge.stats().get(mode, 0) == 0:
                    console.print(f"[yellow]Warning: '{mode}' knowledge base is empty.[/yellow]\n")
            else:
                console.print("[dim]Unknown command.[/dim]\n")
            continue

        # deep hint via claude-sonnet-4-6
        if line.startswith("!"):
            question = line[1:].strip()
            if not question:
                continue
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                h = assistant.hint(question, mode=mode)
            _print_hint(h)
            continue

        # fast answer via gpt-4o-mini
        with console.status("[dim]Searching...[/dim]", spinner="dots"):
            result = assistant.query(line, mode=mode)
        _print_answer(result)


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
