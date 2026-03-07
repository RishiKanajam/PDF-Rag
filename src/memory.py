"""
memory.py
---------
Manages conversation history for multi-turn chat.

What you learn here:

STATELESS vs STATEFUL queries:
  The basic RAG pipeline (query()) is stateless — each question is answered
  independently, with no knowledge of what was asked before.
  For real conversations, you need state: the system must "remember" what
  was just discussed so follow-up questions like "tell me more about that"
  or "how does it compare to the other method?" actually make sense.

HOW MEMORY WORKS HERE:
  ConversationMemory stores a list of {"role": ..., "content": ...} dicts —
  the same format used by LLM chat APIs. Each completed Q&A pair is one
  "turn". We keep only the last MEMORY_MAX_TURNS turns to avoid the context
  window growing forever.

  Example after 2 turns:
    [
      {"role": "user",      "content": "What is NMF?"},
      {"role": "assistant", "content": "NMF stands for Neural Mesh Flow..."},
      {"role": "user",      "content": "How does it compare to AtlasNet?"},
      {"role": "assistant", "content": "Compared to AtlasNet, NMF..."},
    ]

WHY WE FORMAT HISTORY AS A STRING FOR THE PROMPT:
  LLMs read history as plain text in the prompt. We format it as:
      User: What is NMF?
      Assistant: NMF stands for...
  This is clear, readable, and works with any LLM regardless of whether
  it natively supports "messages" or just a single text prompt.

MAX TURNS WINDOW:
  Keeping unlimited history eventually overflows the LLM's context window
  (~8k tokens for llama-3.1-8b-instant). MEMORY_MAX_TURNS=5 means we keep
  the last 5 Q&A pairs — enough context for a coherent conversation without
  hitting token limits.
"""

from typing import List, Dict

import config


class ConversationMemory:
    def __init__(self, max_turns: int = config.MEMORY_MAX_TURNS):
        """
        Args:
            max_turns: number of past Q&A pairs to keep in context.
                       Each turn = 1 user message + 1 assistant message.
        """
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_user(self, message: str):
        self._history.append({"role": "user", "content": message})

    def add_assistant(self, message: str):
        self._history.append({"role": "assistant", "content": message})

    def clear(self):
        """Reset the conversation — useful for starting a fresh topic."""
        self._history.clear()
        print("[memory] Conversation cleared.")

    # ── Read ───────────────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    def get_recent(self) -> List[Dict[str, str]]:
        """
        Return the last max_turns Q&A pairs as a flat list of message dicts.
        Each turn = 2 messages (user + assistant), so we slice -(max_turns * 2).
        """
        return self._history[-(self.max_turns * 2):]

    def format_for_prompt(self) -> str:
        """
        Format recent history as a plain-text block for injection into a prompt.

        Example output:
            User: What is NMF?
            Assistant: NMF is a method that generates manifold meshes...
            User: How does it compare to AtlasNet?
            Assistant: Unlike AtlasNet, NMF guarantees manifoldness...
        """
        recent = self.get_recent()
        if not recent:
            return ""
        lines = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    def turn_count(self) -> int:
        """Number of complete Q&A turns stored."""
        return len(self._history) // 2
