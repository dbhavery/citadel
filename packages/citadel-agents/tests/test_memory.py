"""Tests for citadel_agents.memory — ConversationMemory and VectorMemory."""

from __future__ import annotations

import importlib.util

import pytest

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
    """Tests for long-term memory: keyword fallback and the vector backend."""

    def test_store_and_recall_returns_relevant_text(self) -> None:
        """Stored texts are recalled by relevance on whichever backend is active."""
        mem = VectorMemory()

        mem.store("Python is a programming language used for AI and web development")
        mem.store("JavaScript runs in the browser and on Node.js servers")
        mem.store("Rust is a systems programming language focused on safety")
        mem.store("The weather today is sunny and warm")

        results = mem.recall("programming language", k=3)
        assert len(results) > 0
        # Should surface the programming-related entries, not the weather one.
        assert any("Python" in r or "Rust" in r for r in results)
        assert all("weather" not in r for r in results)

    def test_recall_on_empty_memory_returns_empty(self) -> None:
        """Recall against an empty store returns [] rather than raising."""
        mem = VectorMemory()
        assert mem.recall("anything at all", k=5) == []

    def test_backend_selection_matches_availability(self) -> None:
        """Backend is 'vector' iff citadel_vector is importable, else 'keyword'."""
        mem = VectorMemory()
        if importlib.util.find_spec("citadel_vector") is None:
            assert mem._backend == "keyword"
        else:
            assert mem._backend == "vector"

    def test_vector_backend_embeds_stores_and_recalls(self, tmp_path) -> None:
        """With citadel_vector installed, the vector path embeds, stores, and recalls.

        This is the integration test that the isolated CI matrix used to skip —
        it fails loudly if the vector-memory wiring (dim, embedding, add/search
        signatures) is broken.
        """
        pytest.importorskip("citadel_vector")

        mem = VectorMemory(path=str(tmp_path / "vec_mem"))
        assert mem._backend == "vector"

        mem.store("neural networks and deep learning power modern AI")
        mem.store("baking sourdough bread needs a good starter")
        mem.store("distributed systems and consensus protocols")

        results = mem.recall("deep learning", k=1)
        assert results
        assert "neural networks" in results[0]
