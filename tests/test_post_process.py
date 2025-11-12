"""Tests for token post-processing across languages."""

import pytest

from milvus_segment_generator.segmentation.base import post_process_tokens
from milvus_segment_generator.segmentation.rules import tibetan, english, chinese


class TestTibetanPostProcessing:
    """Test Tibetan-specific token merging patterns."""
    
    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            # No changes when no patterns match (character-level tokens)
            (["ཤ", "ཱ", "་", "རི", "འི", "་", "བུ"], ["ཤ", "ཱ", "་", "རི", "འི", "་", "བུ"]),
            # Single delimiter with space: ། ␣ །
            (["མེད", "།", " ", "།", "ཚོར", "་", "བ"], ["མེད", "། །", "ཚོར", "་", "བ"]),
            
            # Single delimiter with space: ༎ ␣ ༎
            (["མིག", "༎", " ", "༎", "རྣ"], ["མིག", "༎ ༎", "རྣ"]),
        ],
    )
    def test_tibetan_merging_patterns(self, tokens, expected):
        """Test various Tibetan token merging patterns with character-level tokens."""
        assert post_process_tokens(tokens, tibetan.rules) == expected
    
    
    
    def test_tibetan_realistic_sentence(self):
        """Test with realistic character-level Tibetan sentence tokens."""
        # Character-level tokenization like Gemma would produce
        tokens = [
            "ཤ", "ཱ", "་", "རི", "འི", "་", "བུ", "་",
            "དེ", "་", "ལྟ", "་", "བ", "ས", "་", "ན", "་",
            "སྟ", "ོང", "་", "པ", "་", "ཉིད", "་", "ལ", "་",
            "གཟུགས", "་", "མེད", "།", " ", "།",
            "ཚོར", "་", "བ", "་", "མེད", "།"
        ]
        result = post_process_tokens(tokens, tibetan.rules)
        # The ། ␣ ། pattern should be merged
        assert "། །" in result
        assert result.count("།") < tokens.count("།")  # Some delimiters merged
    
    def test_delimiter_merging_shad_space_shad(self):
        """Test that ། ␣ ། pattern is correctly merged."""
        tokens = ["མེད", "།", " ", "།", "ཚོར"]
        result = post_process_tokens(tokens, tibetan.rules)
        
        assert result == ["མེད", "། །", "ཚོར"]
        assert "། །" in result
        assert result.count("།") == 0  # Both individual ། merged into one token
    

