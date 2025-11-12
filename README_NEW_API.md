# Milvus Segment Generator - New Multi-Language API

## Overview

The library has been restructured to support multiple languages with a clean separation of concerns:

- **Single tokenizer** (Gemma3) for all languages
- **Language-specific rules** for delimiters and token merging
- **Shared segmentation logic** for consistent behavior

## Supported Languages

- **Tibetan** (`tibetan`, `bo`, `bod`)
- **English** (`english`, `en`, `eng`)
- **Chinese** (`chinese`, `zh`, `zho`, `cmn`)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from milvus_segment_generator import segment_text, segment_text_to_json

# Segment Tibetan text
spans = segment_text(
    "བཅོམ་ལྡན་འདས།",
    lang="tibetan",
    segment_size=2000
)

# Save to JSON
output_path = segment_text_to_json(
    "བཅོམ་ལྡན་འདས།",
    lang="bo",
    output_path="output.json",
    segment_size=2000
)
```

## API Reference

### `segment_text(text, lang, segment_size=2000)`

Segment text into chunks and return character spans.

**Parameters:**
- `text` (str): Input text to segment
- `lang` (str): Language code (e.g., 'tibetan', 'bo', 'english', 'en', 'chinese', 'zh')
- `segment_size` (int): Maximum tokens per segment (default: 2000)

**Returns:**
- List of span dictionaries with 'start' and 'end' character offsets

**Example:**
```python
spans = segment_text("Hello world.", lang="english", segment_size=100)
# [{"span": {"start": 0, "end": 12}}]
```

### `segment_text_to_json(text, lang, output_path, segment_size=2000)`

Segment text and write spans to a JSON file.

**Parameters:**
- `text` (str): Input text to segment
- `lang` (str): Language code
- `output_path` (str | Path): Output file path
- `segment_size` (int): Maximum tokens per segment (default: 2000)

**Returns:**
- Path object pointing to the created JSON file

**Example:**
```python
path = segment_text_to_json(
    "你好世界。",
    lang="chinese",
    output_path="output.json",
    segment_size=100
)
```

### `list_supported_languages()`

Get list of all supported language codes.

**Returns:**
- Sorted list of language identifiers

**Example:**
```python
from milvus_segment_generator import list_supported_languages

langs = list_supported_languages()
# ['bo', 'bod', 'chinese', 'cmn', 'en', 'eng', 'english', 'tibetan', 'zh', 'zho']
```

## Architecture

```
src/milvus_segment_generator/
├── __init__.py                    # Public API exports
├── tokenizer.py                   # Gemma3 tokenizer (shared)
├── segment.py                     # Main API functions
└── segmentation/
    ├── __init__.py
    ├── base.py                    # Shared types and logic
    ├── factory.py                 # Language rules lookup
    └── rules/
        ├── __init__.py
        ├── tibetan.py             # Tibetan-specific rules
        ├── english.py             # English-specific rules
        └── chinese.py             # Chinese-specific rules
```

## Language-Specific Behavior

### Tibetan
- **Delimiters:** `།`, `༔`, `༎`
- **Token Merging:** 
  - Merges `། ␣ །` → `། །`
  - Merges `།། ␣ །།` → `།། །།`
  - Same patterns for `༔` and `༎`

### English
- **Delimiters:** `.`, `!`, `?`, `;`, `:`
- **Token Merging:** None

### Chinese
- **Delimiters:** `。`, `！`, `？`, `；`, `、`
- **Token Merging:** None

## Adding New Languages

1. Create a new rules file in `segmentation/rules/`:

```python
# segmentation/rules/french.py
from milvus_segment_generator.segmentation.base import LanguageRules

rules = LanguageRules(
    name="french",
    delimiters=(".", "!", "?", ";", ":", "..."),
    merge_templates=(),  # Add if needed
)
```

2. Register in `segmentation/factory.py`:

```python
from milvus_segment_generator.segmentation.rules import french

_LANGUAGE_REGISTRY = {
    # ... existing entries ...
    "french": french.rules,
    "fr": french.rules,
    "fra": french.rules,
}
```

## Migration from Old API

### Old Code
```python
from milvus_segment_generator.segmentor import segment_text_to_json

output = segment_text_to_json(text, output_path, segment_size)
```

### New Code
```python
from milvus_segment_generator import segment_text_to_json

output = segment_text_to_json(text, lang="tibetan", output_path=output_path, segment_size=segment_size)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run the example:

```bash
python examples/segment_example.py
```

## Notes

- The old `segmentor.py` module is deprecated but still present for reference
- All new code should use the new API from `milvus_segment_generator.segment`
- The tokenizer is cached for performance across multiple calls

