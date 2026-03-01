"""Tests for span chunking functionality."""

import pytest

from milvus_segment_generator.segmentation.base import (
    chunk_spans,
    MAX_SEGMENT_CHAR_SPAN,
    MAX_SEGMENT_UTF8_BYTES,
)
from milvus_segment_generator.segmentation.rules import tibetan, english, chinese


class TestTibetanChunkSpans:
    """Test span chunking for Tibetan text."""
    
    
    def test_tibetan_realistic_sentence(self):
        """Test chunking with realistic character-level Tibetan sentence."""
        # Character and subword level tokenization like Gemma tokenizer produces
        tokens = [
            "ཤ", "ཱ", "་", "རི", "འི", "་", "བུ", "།",
            "དེ", "་", "ལྟ", "་", "བ", "ས", "།", "ན", "་",
            "སྟ", "ོང", "་", "པ", "༎", "ཉིད", "་", "ལ", "་",
            "གཟུགས", "་", "མེད", "།",
            "ཚོར", "་", "མེད", "།",
            "འདུ", "་", "ཤེས", "༔", "མེད", "།"
        ]
        expected_segmented_text = """​ཤཱ་རིའི་བུ།
དེ་ལྟ་བས།
ན་སྟོང་པ༎
ཉིད་ལ་གཟུགས་མེད།
ཚོར་མེད།​འདུ་ཤེས༔
མེད།"""
        expected_segmented_text = expected_segmented_text.replace("\u200b", "")
        expected_spans = [
            {"span": {"start": 0, "end": 11}},
            {"span": {"start": 11, "end": 20}},
            {"span": {"start": 20, "end": 29}},
            {"span": {"start": 29, "end": 45}},
            {"span": {"start": 45, "end": 61}},
            {"span": {"start": 61, "end": 65}},
        ]
        # Should chunk at each ། delimiter
        span_result, segmented_text = chunk_spans(tokens, tibetan.rules, segment_size=9, has_delimiter=True)
        assert len(span_result) == 6  # Three segments ending with །
        assert span_result == expected_spans
        assert segmented_text == expected_segmented_text

        

class TestEnglishChunkSpans:
    """Test span chunking for English text."""
    
    def test_english_realistic_subword_tokens(self):
        """Test English chunking with realistic subword tokenization."""
        # Subword tokenization like Gemma tokenizer produces for English
        tokens = [
            "The", " ", "quick", " ", "brown", " ", "fox", ".",
            "It", " ", "jump", "s", " ", "over", ".",
            "The", " ", "la", "zy", " ", "dog", "!",
            "What", " ", "happen", "ed", "?",
            "Every", "thing", " ", "is", " ", "good", "."
        ]
        expected_segmented_text = """The quick brown fox.
It jumps over.
The lazy dog!
What happened?
Everything is good."""
        expected_spans = [
            {"span": {"start": 0, "end": 20}},   # "The quick brown fox."
            {"span": {"start": 20, "end": 34}},  # "It jumps over."
            {"span": {"start": 34, "end": 47}},  # "The lazy dog!"
            {"span": {"start": 47, "end": 61}},  # "What happened?"
            {"span": {"start": 61, "end": 80}},  # "Everything is good."
        ]
        
        result, segmented_text = chunk_spans(tokens, english.rules, segment_size=10, has_delimiter=True)
        assert len(result) == 5
        assert result == expected_spans
        assert segmented_text == expected_segmented_text


def test_chunk_spans_empty_tokens_returns_empty():
    """When tokens list is empty, chunk_spans should return no spans and empty text."""
    tokens = []
    result, segmented_text = chunk_spans(tokens, tibetan.rules, segment_size=5, has_delimiter=True)
    assert result == []
    assert segmented_text == ""

def test_chunk_spans_empty_tokens_english_returns_empty():
    """Empty tokens for English should yield no spans and empty text."""
    result, segmented_text = chunk_spans([], english.rules, segment_size=5, has_delimiter=True)
    assert result == []
    assert segmented_text == ""

def test_chunk_spans_empty_tokens_chinese_returns_empty():
    """Empty tokens for Chinese should yield no spans and empty text."""
    result, segmented_text = chunk_spans([], chinese.rules, segment_size=5, has_delimiter=True)
    assert result == []
    assert segmented_text == ""

def test_smaller_than_segment_size_tibetan_single_segment():
    """Tokens fewer than segment_size should still produce a single segment if ending with delimiter."""
    tokens = ["བདེ", "་", "ལེགས", "།"]
    # segment_size much larger than tokens length
    result, segmented_text = chunk_spans(tokens, tibetan.rules, segment_size=50, has_delimiter=True)
    # Entire text is one segment
    assert result == [{"span": {"start": 0, "end": 9}}]
    assert segmented_text == "བདེ་ལེགས།"

def test_smaller_than_segment_size_english_single_segment():
    """English tokens fewer than segment_size should form one segment if last token is a delimiter."""
    tokens = ["Hello", ".", " ", "World", "!"]
    result, segmented_text = chunk_spans(tokens, english.rules, segment_size=50, has_delimiter=True)
    # One segment covering the entire string ending with '!'
    assert result == [{"span": {"start": 0, "end": 13}}]
    # Do not assert exact segmented_text formatting to avoid newline sensitivity
    assert segmented_text == "Hello. World!"

def test_smaller_than_segment_size_chinese_single_segment():
    """Chinese tokens fewer than segment_size should form one segment if last token is a delimiter."""
    tokens = ["你", "好", "。"]
    result, segmented_text = chunk_spans(tokens, chinese.rules, segment_size=50, has_delimiter=True)
    assert result == [{"span": {"start": 0, "end": 3}}]
    assert segmented_text == "你好。"


class TestChineseChunkSpans:
    """Test span chunking for Chinese text."""
    
    def test_chinese_realistic_subword_tokens(self):
        """Test Chinese chunking with realistic subword tokenization."""
        # Subword/character tokenization like Gemma tokenizer produces for Chinese
        tokens = [
            "我", "爱", "中", "国", "。",
            "这", "是", "一", "个", "测", "试", "。",
            "你", "好", "吗", "？",
            "很", "高", "兴", "见", "到", "你", "！",
            "今", "天", "天", "气", "很", "好", "。"
        ]
        expected_segmented_text = """我爱中国。
这是一个测试。
你好吗？
很高兴见到你！
今天天气很好。"""
        expected_spans = [
            {"span": {"start": 0, "end": 5}},   # "我爱中国。"
            {"span": {"start": 5, "end": 12}},  # "这是一个测试。"
            {"span": {"start": 12, "end": 16}},  # "你好吗？"
            {"span": {"start": 16, "end": 23}},  # "很高兴见到你！"
            {"span": {"start": 23, "end": 30}},  # "今天天气很好。"
        ]
        
        result,segmented_text = chunk_spans(tokens, chinese.rules, segment_size=8, has_delimiter=True)
        assert len(result) == 5
        assert result == expected_spans
        assert segmented_text == expected_segmented_text


def test_chunk_spans_splits_oversized_segment_by_max_char_span():
    """Oversized segments should be split into contiguous <= max-span chunks."""
    long_token = "a" * (MAX_SEGMENT_CHAR_SPAN + 100)
    tokens = [long_token, "."]

    result, segmented_text = chunk_spans(tokens, english.rules, segment_size=10, has_delimiter=True)

    assert len(result) == 2
    assert result[0]["span"]["start"] == 0
    assert result[0]["span"]["end"] == MAX_SEGMENT_CHAR_SPAN
    assert result[1]["span"]["start"] == MAX_SEGMENT_CHAR_SPAN
    assert result[1]["span"]["end"] == len(long_token) + 1
    assert result[0]["span"]["end"] - result[0]["span"]["start"] <= MAX_SEGMENT_CHAR_SPAN
    assert result[1]["span"]["end"] - result[1]["span"]["start"] <= MAX_SEGMENT_CHAR_SPAN
    assert segmented_text == f"{long_token[:MAX_SEGMENT_CHAR_SPAN]}\n{long_token[MAX_SEGMENT_CHAR_SPAN:] + '.'}"


def test_chunk_spans_prefers_whitespace_when_splitting_oversized_segment():
    """Splitting should prefer a nearby natural boundary when available."""
    prefix = "a" * (MAX_SEGMENT_CHAR_SPAN - 5)
    tokens = [prefix, " ", "word", "."]

    result, segmented_text = chunk_spans(tokens, english.rules, segment_size=10, has_delimiter=True)

    assert len(result) == 2
    assert result[0] == {"span": {"start": 0, "end": len(prefix) + 1}}
    assert result[1] == {"span": {"start": len(prefix) + 1, "end": len(prefix) + 6}}
    assert segmented_text == f"{prefix} \nword."


def test_chunk_spans_splits_by_utf8_bytes_for_multibyte_characters():
    """Segments must stay within Milvus VARCHAR byte limit, not only char count."""
    long_token = ("a" * (MAX_SEGMENT_UTF8_BYTES - 1)) + "ཀ"
    tokens = [long_token, "."]

    result, segmented_text = chunk_spans(tokens, english.rules, segment_size=10, has_delimiter=True)

    assert len(result) == 2
    first_piece = segmented_text.split("\n")[0]
    second_piece = segmented_text.split("\n")[1]
    assert len(first_piece.encode("utf-8")) <= MAX_SEGMENT_UTF8_BYTES
    assert len(second_piece.encode("utf-8")) <= MAX_SEGMENT_UTF8_BYTES
    assert result[0]["span"] == {"start": 0, "end": MAX_SEGMENT_UTF8_BYTES - 1}
    assert result[1]["span"] == {"start": MAX_SEGMENT_UTF8_BYTES - 1, "end": len(long_token) + 1}

