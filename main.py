"""
main.py
-------
Entry point for the PDFRag CLI.

Usage:
    python main.py              # interactive chat loop
    python main.py --rebuild    # force re-index all documents, then chat
"""

import argparse

import config
from src.memory import ConversationMemory
from src.rag import RAGPipeline

COMMANDS = {
    "clear": "clear conversation history and start fresh",
    "history": "show the current conversation history",
    "exit": "quit",
    "quit": "quit",
}


def parse_args():
    parser = argparse.ArgumentParser(description="PDFRag — Ask questions about your documents")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-index all documents even if the vector store already has data",
    )
    return parser.parse_args()


def chat_loop(pipeline: RAGPipeline):
    memory = ConversationMemory()

    print("\n" + "=" * 58)
    print("  PDFRag — Ask anything about your documents")
    print("  Commands: 'clear' (reset history) | 'history' | 'exit'")
    print("=" * 58 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        # ── Commands ───────────────────────────────────────────────────────────
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if question.lower() == "clear":
            memory.clear()
            print("  (Conversation history cleared — starting fresh)\n")
            continue

        if question.lower() == "history":
            if memory.is_empty:
                print("  (No conversation history yet)\n")
            else:
                print("\n── Conversation history ──")
                print(memory.format_for_prompt())
                print("─" * 26 + "\n")
            continue

        # ── Chat with memory ───────────────────────────────────────────────────
        result = pipeline.chat(question, memory)

        # Show reformulated question if it differed
        standalone = result.get("standalone_question", question)
        if standalone != question:
            print(f"  (Searching for: \"{standalone}\")")

        print(f"\nAssistant: {result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            seen = set()
            for s in result["sources"]:
                key = (s["source"], s["page"])
                if key not in seen:
                    seen.add(key)
                    print(
                        f"  • {s['source']}  "
                        f"page {s['page']}  "
                        f"(rerank: {s['rerank_score']})"
                    )
        print(f"  [Turn {memory.turn_count()} | {memory.max_turns} max in context]\n")


def main():
    config.validate_config()

    args = parse_args()

    pipeline = RAGPipeline()
    pipeline.build(force=args.rebuild)

    chat_loop(pipeline)


if __name__ == "__main__":
    main()
