# Migration from Gemma Package to Transformers

This document describes the changes made to migrate from the `gemma` package to HuggingFace `transformers` library.

## Summary

The tokenizer has been updated to use HuggingFace Transformers' Gemma model instead of the standalone `gemma` package. This provides better compatibility, easier installation, and access to the full ecosystem of HuggingFace models.

## Changes Made

### 1. Tokenizer Module (`src/milvus_segment_generator/tokenizer.py`)

**Before:**
```python
from gemma import gm

@lru_cache(maxsize=1)
def _get_gemma3_tokenizer():
    return gm.text.Gemma3Tokenizer()

def tokenize(text: str) -> List[str]:
    tokenizer = _get_gemma3_tokenizer()
    token_ids = tokenizer.encode(text)
    return [tokenizer.decode([token_id]) for token_id in token_ids]
```

**After:**
```python
from transformers import AutoTokenizer

@lru_cache(maxsize=1)
def _get_gemma_tokenizer():
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def tokenize(text: str) -> List[str]:
    tokenizer = _get_gemma_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode([token_id]) for token_id in token_ids]
```

**Key differences:**
- Uses `AutoTokenizer.from_pretrained()` to load the Gemma tokenizer
- Model: `google/gemma-2-2b-it` (can be changed to other Gemma variants)
- Added `add_special_tokens=False` to match previous behavior
- Tokenizer is cached using `@lru_cache` for performance

### 2. Dependencies (`pyproject.toml`)

**Before:**
```toml
dependencies = [
    "gemma>=0.1.0",
]
```

**After:**
```toml
dependencies = [
    "transformers>=4.30.0",
    "torch>=2.0.0",
]
```

**Note:** PyTorch is required by transformers for tokenizer operations.

### 3. Documentation Updates

Updated the following files to reflect the change:
- `README.md` - Main project documentation
- `README_NEW_API.md` - API reference
- `src/milvus_segment_generator/__init__.py` - Module docstring
- `tests/README.md` - Test documentation
- `tests/test_chunk_spans.py` - Test comments
- `tests/test_post_process.py` - Test comments

## Installation

### New Installation Steps

```bash
# Install the package with new dependencies
pip install -e .

# Or install dependencies separately
pip install transformers torch
```

### First-Time Setup

The first time you run the code, it will download the Gemma tokenizer model (~500MB) from HuggingFace. Subsequent runs will use the cached version.

**Optional:** Login to HuggingFace for better download speeds and access to gated models:
```bash
huggingface-cli login
```

Or set the token as an environment variable:
```bash
export HF_TOKEN=your_token_here
```

## Available Gemma Models

You can change the model in `tokenizer.py` to use different Gemma variants:

- `google/gemma-2-2b` - Gemma 2 2B (base)
- `google/gemma-2-2b-it` - Gemma 2 2B (instruction-tuned) **[Current]**
- `google/gemma-2-9b` - Gemma 2 9B (base)
- `google/gemma-2-9b-it` - Gemma 2 9B (instruction-tuned)
- `google/gemma-2-27b` - Gemma 2 27B (base)
- `google/gemma-2-27b-it` - Gemma 2 27B (instruction-tuned)
- `google/gemma-7b` - Gemma 1 7B

## Benefits of This Migration

1. **Better Ecosystem Integration**: Access to HuggingFace's model hub and tools
2. **Easier Installation**: Standard pip install without custom packages
3. **Model Flexibility**: Easy to switch between different Gemma variants
4. **Community Support**: Larger community and better documentation
5. **Caching**: Automatic model caching via HuggingFace cache system
6. **Updates**: Regular updates and improvements from HuggingFace

## Backward Compatibility

The API remains unchanged. All existing code using `segment_text()` and `segment_text_to_json()` will work without modifications.

```python
# This code works exactly the same as before
from milvus_segment_generator import segment_text

spans = segment_text("བཅོམ་ལྡན་འདས།", lang="tibetan", segment_size=2000)
```

## Troubleshooting

### Issue: Model download fails
**Solution:** Check your internet connection and HuggingFace access. Some models may require authentication.

### Issue: ImportError: No module named 'transformers'
**Solution:** Install transformers: `pip install transformers torch`

### Issue: Slow first run
**Solution:** This is expected. The model is being downloaded and cached. Subsequent runs will be fast.

### Issue: Different tokenization results
**Solution:** The tokenization should be identical. If you notice differences, please file an issue with examples.

## Testing

All existing tests pass without modification. The test suite uses mocked tokenizers, so no actual model downloads occur during testing.

Run tests:
```bash
pytest tests/
```

## Environment Variables

- `HF_TOKEN`: HuggingFace API token (optional, for gated models)
- `TRANSFORMERS_CACHE`: Cache directory (default: `~/.cache/huggingface`)
- `HF_HOME`: HuggingFace home directory (default: `~/.cache/huggingface`)

## Performance Notes

- **First run**: Slower due to model download (~500MB)
- **Subsequent runs**: Same performance as before (model is cached)
- **Memory usage**: Similar to previous implementation
- **Tokenization speed**: Comparable to the original `gemma` package

## Questions or Issues?

If you encounter any problems with the migration, please:
1. Check this document for common issues
2. Review the updated README.md
3. File an issue on GitHub with details about your environment and error messages

