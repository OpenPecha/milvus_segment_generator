"""Shared types and logic for text segmentation across languages."""

from dataclasses import dataclass
from typing import List, Tuple

# Placeholder for merge templates to indicate "any delimiter from the rule set"
ANY_DELIM = "DELIM"


@dataclass(frozen=True)
class LanguageRules:
    """Configuration for language-specific segmentation rules.
    
    Attributes:
        name: Language identifier.
        delimiters: Tuple of delimiter characters that mark segment boundaries.
        merge_templates: Tuple of token patterns to merge. Each pattern is a tuple
            where ANY_DELIM represents "the same delimiter". Empty tuple means no merging.
    """
    name: str
    delimiters: Tuple[str, ...]
    merge_templates: Tuple[Tuple[str, ...], ...] = tuple()


def _expand_merge_patterns(rules: LanguageRules) -> List[List[str]]:
    """Expand merge templates into concrete patterns for each delimiter.
    
    Args:
        rules: Language rules containing delimiters and merge templates.
        
    Returns:
        List of concrete token patterns to match and merge.
    """
    if not rules.merge_templates:
        return []
    
    patterns: List[List[str]] = []
    for delimiter in rules.delimiters:
        for template in rules.merge_templates:
            pattern = [delimiter if token == ANY_DELIM else token for token in template]
            patterns.append(pattern)
    return patterns


def post_process_tokens(tokens: List[str], rules: LanguageRules) -> List[str]:
    """Merge token sequences according to language-specific rules.
    
    Args:
        tokens: List of decoded token strings.
        rules: Language rules specifying merge patterns.
        
    Returns:
        List of tokens with specified patterns merged into single tokens.
    """
    patterns = _expand_merge_patterns(rules)
    if not patterns:
        return tokens  # No merging needed
    
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for pattern in patterns:
            n = len(pattern)
            if i + n <= len(tokens) and tokens[i:i+n] == pattern:
                merged.append("".join(pattern))
                i += n
                matched = True
                break
        
        if not matched:
            merged.append(tokens[i])
            i += 1
    
    return merged


def chunk_spans(tokens: List[str], rules: LanguageRules, segment_size: int) -> List[dict]:
    """Chunk tokens into segments ending at delimiters and return character spans.
    
    Args:
        tokens: List of decoded token strings.
        rules: Language rules specifying valid delimiters.
        segment_size: Maximum number of tokens per segment.
        
    Returns:
        List of span dictionaries with 'start' and 'end' character offsets.
        
    Raises:
        ValueError: If segment_size is invalid or no delimiter found within window.
    """
    if segment_size <= 0:
        raise ValueError("segment_size must be a positive integer")
    
    spans: List[dict] = []
    start_index = 0
    char_offset = 0
    total_tokens = len(tokens)
    
    while start_index < total_tokens:
        upper_bound = min(start_index + segment_size, total_tokens)
        cut_index = None
        
        # Search backward from upper_bound for a token ending with a delimiter
        for idx in range(upper_bound - 1, start_index - 1, -1):
            if any(tokens[idx].endswith(delimiter) for delimiter in rules.delimiters):
                cut_index = idx + 1
                break
        
        if cut_index is None:
            raise ValueError(
                f"Unable to find a delimiter {rules.delimiters} within "
                f"{segment_size} tokens starting at index {start_index}."
            )
        
        segment_tokens = tokens[start_index:cut_index]
        segment_text = "".join(segment_tokens)
        segment_length = len(segment_text)
        
        spans.append({
            "span": {
                "start": char_offset,
                "end": char_offset + segment_length
            }
        })
        
        char_offset += segment_length
        start_index = cut_index
    
    return spans


__all__ = ["LanguageRules", "ANY_DELIM", "post_process_tokens", "chunk_spans"]

