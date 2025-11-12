"""Chinese language segmentation rules."""

from milvus_segment_generator.segmentation.base import LanguageRules

rules = LanguageRules(
    name="chinese",
    delimiters=("。", "！", "？", "；", "、"),
    merge_templates=(),  # No token merging needed for Chinese
)


__all__ = ["rules"]

