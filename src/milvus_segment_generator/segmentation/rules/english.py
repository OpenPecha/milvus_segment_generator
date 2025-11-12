"""English language segmentation rules."""

from milvus_segment_generator.segmentation.base import LanguageRules

rules = LanguageRules(
    name="english",
    delimiters=(".", "!", "?", ";", ":"),
    merge_templates=(),  # No token merging needed for English
)


__all__ = ["rules"]

