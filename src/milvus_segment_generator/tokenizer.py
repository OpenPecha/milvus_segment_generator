"""Gemma3 tokenizer service for all languages."""

from functools import lru_cache
from typing import List

try:
    from gemma import gm
except ImportError as exc:
    raise ImportError(
        "The `gemma` package is required for tokenization. "
        "Install it with `pip install gemma`."
    ) from exc


@lru_cache(maxsize=1)
def _get_gemma3_tokenizer():
    """Load and cache the Gemma3 tokenizer."""
    return gm.text.Gemma3Tokenizer()


def tokenize(text: str) -> List[str]:
    """Tokenize text with the Gemma3 tokenizer, returning decoded token strings.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of decoded token strings.
    """
    tokenizer = _get_gemma3_tokenizer()
    token_ids = tokenizer.encode(text)
    return [tokenizer.decode([token_id]) for token_id in token_ids]


__all__ = ["tokenize"]

