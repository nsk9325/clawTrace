from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """Owns the conversational history for one episode."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""

    @classmethod
    def with_initial_user(cls, content: str, system_prompt: str = "") -> "Memory":
        return cls(
            messages=[{"role": "user", "content": content}],
            system_prompt=system_prompt,
        )

    def append_assistant(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        self.messages.append({
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        })

    def append_tool(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })

    def final_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                return str(message.get("content", ""))
        return ""

    def __len__(self) -> int:
        return len(self.messages)
