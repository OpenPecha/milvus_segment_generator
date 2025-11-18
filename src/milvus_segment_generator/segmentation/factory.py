"""Factory for retrieving language-specific segmentation rules."""

from milvus_segment_generator.segmentation.base import LanguageRules
from milvus_segment_generator.segmentation.rules import tibetan, english, chinese


# Map language codes and names to their rules
_LANGUAGE_REGISTRY: dict[str, LanguageRules] = {
    # Tibetan
    "tibetan": tibetan.rules,
    "bo": tibetan.rules,
    "bod": tibetan.rules,
    
    # English
    "english": english.rules,
    "en": english.rules,
    "eng": english.rules,
    
    # Chinese
    "chinese": chinese.rules,
    "zh": chinese.rules,
    "zho": chinese.rules,
    "cmn": chinese.rules,
    "lzh": chinese.rules,
}


def get_rules(lang: str) -> LanguageRules:
    """Get segmentation rules for a given language.
    
    Args:
        lang: Language code or name (case-insensitive). 
              Examples: 'tibetan', 'bo', 'english', 'en', 'chinese', 'zh'
              
    Returns:
        LanguageRules for the specified language. Defaults to English if not found.
    """
    normalized = (lang or "").lower().strip()
    return _LANGUAGE_REGISTRY.get(normalized, english.rules)


def list_supported_languages() -> list[str]:
    """Return list of supported language identifiers.
    
    Returns:
        Sorted list of all registered language codes and names.
    """
    return sorted(_LANGUAGE_REGISTRY.keys())


__all__ = ["get_rules", "list_supported_languages"]

