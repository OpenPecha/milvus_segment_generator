"""Shared types and logic for text segmentation across languages."""

from dataclasses import dataclass
from typing import List, Tuple

# Placeholder for merge templates to indicate "any delimiter from the rule set"
ANY_DELIM = "DELIM"
MAX_SEGMENT_CHAR_SPAN = 65_535
MAX_SEGMENT_UTF8_BYTES = 65_535


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


def _find_split_end(segment_text: str, start: int, candidate_end: int, delimiters: Tuple[str, ...]) -> int:
    """Choose a split end <= candidate_end, preferring delimiters and whitespace."""
    delimiter_set = set(delimiters)

    for pos in range(candidate_end, start, -1):
        if segment_text[pos - 1] in delimiter_set:
            return pos

    for pos in range(candidate_end, start, -1):
        if segment_text[pos - 1].isspace():
            return pos

    return candidate_end


def _find_max_utf8_safe_end(
    segment_text: str,
    start: int,
    end_bound: int,
    max_segment_utf8_bytes: int,
) -> int:
    """Find the furthest character end whose UTF-8 byte length stays within the limit."""
    low = start
    high = end_bound
    best = start

    while low <= high:
        mid = (low + high) // 2
        byte_length = len(segment_text[start:mid].encode("utf-8"))
        if byte_length <= max_segment_utf8_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def _split_segment_text(
    segment_text: str,
    delimiters: Tuple[str, ...],
    max_segment_char_span: int,
    max_segment_utf8_bytes: int = MAX_SEGMENT_UTF8_BYTES,
) -> List[Tuple[int, int]]:
    """Split a segment into contiguous bounds that satisfy char and UTF-8 byte limits."""
    if (
        len(segment_text) <= max_segment_char_span
        and len(segment_text.encode("utf-8")) <= max_segment_utf8_bytes
    ):
        return [(0, len(segment_text))]

    bounds: List[Tuple[int, int]] = []
    cursor = 0
    text_end = len(segment_text)

    while cursor < text_end:
        char_limited_end = min(cursor + max_segment_char_span, text_end)
        byte_limited_end = _find_max_utf8_safe_end(
            segment_text,
            cursor,
            char_limited_end,
            max_segment_utf8_bytes,
        )
        candidate_end = min(char_limited_end, byte_limited_end)
        if candidate_end <= cursor:
            candidate_end = min(cursor + 1, text_end)
        if candidate_end < text_end:
            split_end = _find_split_end(segment_text, cursor, candidate_end, delimiters)
            if split_end <= cursor:
                split_end = candidate_end
        else:
            split_end = candidate_end

        bounds.append((cursor, split_end))
        cursor = split_end

    return bounds


def chunk_spans(tokens: List[str], rules: LanguageRules, segment_size: int, has_delimiter: bool) -> List[dict]:
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
    segmented_parts: List[str] = []
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
        segment_bounds = _split_segment_text(
            segment_text,
            rules.delimiters,
            MAX_SEGMENT_CHAR_SPAN,
        )

        for rel_start, rel_end in segment_bounds:
            piece = segment_text[rel_start:rel_end]
            piece_length = rel_end - rel_start
            segmented_parts.append(piece)

            spans.append({
                "span": {
                    "start": char_offset,
                    "end": char_offset + piece_length
                }
            })
            char_offset += piece_length

        start_index = cut_index
    
    segmented_text = "\n".join(segmented_parts)

    if not has_delimiter and spans:
        spans[-1]["span"]["end"] = spans[-1]['span']['end'] - 1
        if segmented_parts:
            segmented_parts[-1] = segmented_parts[-1][:-1]
            segmented_text = "\n".join(segmented_parts)
    
    return spans, segmented_text


__all__ = [
    "LanguageRules",
    "ANY_DELIM",
    "MAX_SEGMENT_CHAR_SPAN",
    "MAX_SEGMENT_UTF8_BYTES",
    "post_process_tokens",
    "chunk_spans",
]

