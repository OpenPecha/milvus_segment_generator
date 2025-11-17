"""Tests for span chunking functionality."""

import pytest

from milvus_segment_generator.segmentation.base import chunk_spans
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
མེད།
"""
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
        span_result, segmented_text = chunk_spans(tokens, tibetan.rules, segment_size=9)
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
Everything is good.
"""
        expected_spans = [
            {"span": {"start": 0, "end": 20}},   # "The quick brown fox."
            {"span": {"start": 20, "end": 34}},  # "It jumps over."
            {"span": {"start": 34, "end": 47}},  # "The lazy dog!"
            {"span": {"start": 47, "end": 61}},  # "What happened?"
            {"span": {"start": 61, "end": 80}},  # "Everything is good."
        ]
        
        result, segmented_text = chunk_spans(tokens, english.rules, segment_size=10)
        assert len(result) == 5
        assert result == expected_spans
        assert segmented_text == expected_segmented_text


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
今天天气很好。
"""
        expected_spans = [
            {"span": {"start": 0, "end": 5}},   # "我爱中国。"
            {"span": {"start": 5, "end": 12}},  # "这是一个测试。"
            {"span": {"start": 12, "end": 16}},  # "你好吗？"
            {"span": {"start": 16, "end": 23}},  # "很高兴见到你！"
            {"span": {"start": 23, "end": 30}},  # "今天天气很好。"
        ]
        
        result,segmented_text = chunk_spans(tokens, chinese.rules, segment_size=8)
        assert len(result) == 5
        assert result == expected_spans
        assert segmented_text == expected_segmented_text

