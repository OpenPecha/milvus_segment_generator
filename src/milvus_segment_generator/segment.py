"""Public API for text segmentation across multiple languages."""

import json
from pathlib import Path
from typing import List

from milvus_segment_generator.tokenizer import tokenize
from milvus_segment_generator.segmentation.base import post_process_tokens, chunk_spans
from milvus_segment_generator.segmentation.factory import get_rules


def segment_text(text: str, lang: str, segment_size: int = 2200) -> List[dict]:
    """Segment text into chunks and return character spans.
    
    Args:
        text: Input text to segment.
        lang: Language code or name (e.g., 'tibetan', 'bo', 'english', 'en', 'chinese', 'zh').
        segment_size: Maximum number of tokens per segment (default: 1990).
        
    Returns:
        List of span dictionaries with 'start' and 'end' character offsets.
        
    Example:
        >>> spans = segment_text("བཅོམ་ལྡན་འདས།", lang="tibetan", segment_size=100)
        >>> print(spans)
        [{"span": {"start": 0, "end": 15}}]
    """
    rules = get_rules(lang)
    tokens = tokenize(text)
    tokens = post_process_tokens(tokens, rules)
    spans, segments = chunk_spans(tokens, rules, segment_size)
    print(len(segments))
    return spans, segments


def segment_text_to_json(
    text: str,
    lang: str,
    output_path: str | Path,
    segment_size: int = 2000,
) -> Path:
    """Segment text and write spans to a JSON file.
    
    Args:
        text: Input text to segment.
        lang: Language code or name (e.g., 'tibetan', 'bo', 'english', 'en', 'chinese', 'zh').
        output_path: Path where the JSON file will be written.
        segment_size: Maximum number of tokens per segment (default: 2000).
        
    Returns:
        Path object pointing to the created JSON file.
        
    Example:
        >>> path = segment_text_to_json(
        ...     "བཅོམ་ལྡན་འདས།",
        ...     lang="tibetan",
        ...     output_path="output.json",
        ...     segment_size=100
        ... )
        >>> print(f"Spans saved to {path}")
    """
    spans = segment_text(text, lang=lang, segment_size=segment_size)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(spans, handle, ensure_ascii=False, indent=4)
    
    return output_file


__all__ = ["segment_text", "segment_text_to_json"]

