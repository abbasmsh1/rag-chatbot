import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag import chunk_text, build_prompt


def test_chunks_cover_whole_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=600, overlap=120)
    assert len(chunks) > 1
    joined = " ".join(chunks)
    assert "word0" in joined and "word999" in joined
    # consecutive chunks share content
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_empty_and_bad_input():
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_prompt_numbers_contexts_and_guards():
    p = build_prompt("what is x?", ["ctx one", "ctx two"])
    assert "[1] ctx one" in p and "[2] ctx two" in p
    assert "don't know" in p
