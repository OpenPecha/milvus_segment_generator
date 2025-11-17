"""Milvus Segment Generator - Multi-language text segmentation using Gemma tokenizer."""

from milvus_segment_generator.segment import segment_text, segment_text_to_json
from milvus_segment_generator.segmentation.factory import list_supported_languages

__version__ = "0.0.1"

__all__ = [
    "segment_text",
    "segment_text_to_json",
    "list_supported_languages",
]

