"""Tests for citadel_agents.memory — ConversationMemory and VectorMemory."""

from __future__ import annotations

from citadel_agents.memory import ConversationMemory, VectorMemory


class TestConversationMemory:
    """Tests for short-term conversation history."""

    def test_add_and_retrieve(self) -> None:
        """Messages can be added and retrieved in order."""
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        mem.add("user", "How are you?")

        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_max_turns_truncation(self) -> None:
        """Older messages are dropped when max_turns is exceeded."""
        mem = ConversationMemory(max_turns=3)

        mem.add("user", "msg1")
        mem.add("assistant", "msg2")
        mem.add("user", "msg3")
        mem.add("assistant", "msg4")
        mem.add("user", "msg5")

        messages = mem.get_messages()
        assert len(messages) == 3
        # Should keep the last 3
        assert messages[0]["content"] == "msg3"
        assert messages[1]["content"] == "msg4"
        assert messages[2]["content"] == "msg5"

    def test_clear(self) -> None:
        """Clear removes all messages."""
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")

        mem.clear()
        assert mem.get_messages() == []


class TestVectorMemory:
    """Tests for long-term memory with keyword search fallback."""

    def test_store_and_recall(self) -> None:
        """Stored texts can be recalled by keyword search."""
        mem = VectorMemory()
        assert mem._backend == "keyword"  # citadel_vector not installed

        mem.store("Python is a programming language used for AI and web development")
        mem.store("JavaScript runs in the browser and on Node.js servers")
        mem.store("Rust is a systems programming language focused on safety")
        mem.store("The weather today is sunny and warm")

        results = mem.recall("programming language", k=3)
        assert len(results) > 0
        # Should match the programming-related entries
        assert any("Python" in r for r in results)
