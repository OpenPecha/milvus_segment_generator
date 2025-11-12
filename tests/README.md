# Test Suite Organization

This directory contains the test suite for the milvus_segment_generator library, organized by functionality.

## Test Files

### `test_post_process.py`
Tests for token post-processing (merging) functionality:
- **TestTibetanPostProcessing**: Tibetan-specific token merging patterns
  - Single delimiter patterns (། ␣ །, ༔ ␣ ༔, ༎ ␣ ༎)
  - Double delimiter patterns (།། ␣ །།, ༔༔ ␣ ༔༔)
  - Delimiter merging verification tests
  - Multiple consecutive merge patterns
  - Edge cases (no space, different delimiters, space preservation)
  - Realistic character-level Tibetan sentence tokens
- **TestEnglishPostProcessing**: Verifies no merging for English
- **TestChinesePostProcessing**: Verifies no merging for Chinese

### `test_chunk_spans.py`
Tests for text chunking and span generation with realistic subword tokenization:
- **TestTibetanChunkSpans**: 
  - Realistic character-level Tibetan sentence with multiple delimiters (།, ༔, ༎)
  - Small segment sizes (2-9 tokens)
  - Mixed delimiter types in single test
  - Character and subword level tokens
- **TestEnglishChunkSpans**: 
  - Realistic subword tokenization (e.g., "jump" + "s", "la" + "zy")
  - Space tokens as separate elements
  - Multiple delimiters (., !, ?, ;, :)
  - Small segment sizes
- **TestChineseChunkSpans**: 
  - Character-level tokenization (typical for Chinese)
  - Multiple delimiters (。, ！, ？, ；, 、)
  - Real Chinese sentences with accurate character offsets
  - Small segment sizes (2-8 tokens)

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_post_process.py
pytest tests/test_chunk_spans.py
```

### Run specific test class
```bash
pytest tests/test_post_process.py::TestTibetanPostProcessing
pytest tests/test_chunk_spans.py::TestEnglishChunkSpans
```

### Run specific test
```bash
pytest tests/test_post_process.py::TestTibetanPostProcessing::test_delimiter_merging_shad_space_shad
pytest tests/test_chunk_spans.py::TestTibetanChunkSpans::test_tibetan_realistic_sentence
```

### Run with coverage
```bash
pytest --cov=milvus_segment_generator tests/
```

### Run with verbose output
```bash
pytest -v tests/
```

## Test Coverage

The test suite covers:
- ✅ Token post-processing with delimiter merging (Tibetan)
- ✅ Token post-processing without merging (English, Chinese)
- ✅ Span chunking with realistic subword tokenization
- ✅ Multiple delimiter types per language
- ✅ Small segment sizes (2-10 tokens)
- ✅ Character-level and subword-level tokens
- ✅ Accurate character offset calculations
- ✅ Multi-language support (Tibetan, English, Chinese)

## Test Data Characteristics

### Realistic Tokenization
All tests use realistic token patterns that match actual Gemma3 tokenizer behavior:
- **Tibetan**: Character-level tokens (e.g., `"ཤ", "ཱ", "་"`) and subword tokens
- **English**: Subword splits (e.g., `"jump" + "s"`, `"happen" + "ed"`) with space tokens
- **Chinese**: Character-level tokens (e.g., `"我", "爱", "中", "国"`)

### Small Segment Sizes
Tests focus on small segment sizes (2-10 tokens) to thoroughly test delimiter detection and boundary conditions, which is more realistic for production use cases.

## Adding New Tests

When adding new functionality:
1. Add tests to the appropriate file based on functionality
2. Follow existing naming conventions (`test_<functionality>_<scenario>`)
3. Use realistic, character/subword-level tokens
4. Include docstrings explaining what is being tested
5. Test with small segment sizes (2-10 tokens)
6. Verify accurate character offset calculations
7. Update this README if adding new test files or major changes

