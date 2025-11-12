"""Tibetan language segmentation rules."""

from milvus_segment_generator.segmentation.base import LanguageRules, ANY_DELIM

rules = LanguageRules(
    name="tibetan",
    delimiters=("།", "༔", "༎"),
    merge_templates=(
        # Merge patterns like: ། ␣ །  or  ༔ ␣ ༔  or  ༎ ␣ ༎
        (ANY_DELIM, " ", ANY_DELIM),
        # Merge patterns like: །། ␣ །།  or  ༔༔ ␣ ༔༔
        (ANY_DELIM, ANY_DELIM, " ", ANY_DELIM, ANY_DELIM),
    ),
)


__all__ = ["rules"]

