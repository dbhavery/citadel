"""Tests for document parser."""

import os
import tempfile

import pytest

from citadel_ingest.parser import DocumentParser


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


class TestDocumentParser:
    """Tests for DocumentParser."""

    def test_parse_markdown_file(self, parser: DocumentParser) -> None:
        """Parser correctly reads a Markdown file and sets metadata."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Hello\n\nThis is **markdown** content.")
            f.flush()
            path = f.name

        try:
            doc = parser.parse(path)
            assert "# Hello" in doc.text
            assert "**markdown**" in doc.text
            assert doc.metadata["extension"] == ".md"
            assert doc.metadata["filename"].endswith(".md")
            assert doc.source_path != ""
        finally:
            os.unlink(path)

    def test_parse_python_file_with_language_metadata(
        self, parser: DocumentParser
    ) -> None:
        """Parser reads a .py file and includes language metadata."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("def greet():\n    return 'hello'\n")
            f.flush()
            path = f.name

        try:
            doc = parser.parse(path)
            assert "def greet():" in doc.text
            assert doc.metadata["language"] == "python"
            assert doc.metadata["extension"] == ".py"
        finally:
            os.unlink(path)

    def test_parse_plain_text_file(self, parser: DocumentParser) -> None:
        """Parser reads a plain .txt file correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Just some plain text content.\nLine two.")
            f.flush()
            path = f.name

        try:
            doc = parser.parse(path)
            assert doc.text == "Just some plain text content.\nLine two."
            assert doc.metadata["extension"] == ".txt"
            assert doc.metadata["size"] > 0
        finally:
            os.unlink(path)
