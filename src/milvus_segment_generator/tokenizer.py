"""Gemma tokenizer service for all languages using transformers."""

from functools import lru_cache
from typing import List
import os
try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "The `transformers` package is required for tokenization. "
        "Install it with `pip install transformers`."
    ) from exc


@lru_cache(maxsize=1)
def _get_gemma_tokenizer():
    """Load and cache the Gemma tokenizer from transformers.
    
    Uses google/gemma-2-2b-it model tokenizer.
    You can change this to any Gemma model variant:
    - google/gemma-2-2b
    - google/gemma-2-9b
    - google/gemma-2-27b
    - google/gemma-7b
    """
    tokenizer = AutoTokenizer.from_pretrained("openpecha/segment_tokenizer",token=os.getenv("HF_TOKEN"))
    return tokenizer


def tokenize(text: str, rules) -> List[str]:
    """Tokenize text with the Gemma tokenizer, returning decoded token strings.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of decoded token strings (each token decoded individually).
    """
    tokenizer = _get_gemma_tokenizer()
    # Decode each token ID individually to get the token string

    tokens = tokenizer.batch_decode(tokenizer.encode(text,add_special_tokens=False),skip_special_tokens=True)

    tokens, has_delimiter = delimiter_check(tokens, rules)

    return tokens, has_delimiter

def tokenize_with_char(text: str, rules) -> List[str]:
    """Tokenize text with the Gemma tokenizer, returning decoded token strings.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of decoded token strings (each token decoded individually).
    """
    tokens = [char for char in text]
    tokens, has_delimiter = delimiter_check(tokens, rules)  
    return tokens, has_delimiter


def delimiter_check(tokens: List[str], rules) -> List[str]:
    has_delimiter = True
    if tokens[-1] not in rules.delimiters:
        tokens.append(rules.delimiters[0])
        has_delimiter = False
    return tokens, has_delimiter


__all__ = ["tokenize"]

