# Milvus Segment Generator

<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

Multi-language text segmentation library using the Gemma tokenizer from HuggingFace Transformers.

## Owner(s)

- [@ngawangtrinley](https://github.com/ngawangtrinley)
- [@mikkokotila](https://github.com/mikkokotila)
- [@evanyerburgh](https://github.com/evanyerburgh)


## Table of contents
<p align="center">
  <a href="#project-description">Project description</a> •
  <a href="#who-this-project-is-for">Who this project is for</a> •
  <a href="#project-dependencies">Project dependencies</a> •
  <a href="#instructions-for-use">Instructions for use</a> •
  <a href="#contributing-guidelines">Contributing guidelines</a> •
  <a href="#additional-documentation">Additional documentation</a> •
  <a href="#how-to-get-help">How to get help</a> •
  <a href="#terms-of-use">Terms of use</a>
</p>
<hr>

## Project description

Milvus Segment Generator helps you tokenize and segment text into fixed-size chunks with character-level span information. It supports multiple languages (Tibetan, English, Chinese) with language-specific delimiter handling and token post-processing rules.

## Features

- **Multi-language support**: Tibetan, English, and Chinese
- **Gemma tokenizer**: Uses HuggingFace Transformers' Gemma model tokenizer
- **Language-specific rules**: Custom delimiters and token merging for each language
- **Character spans**: Returns precise character offsets for each segment
- **JSON export**: Save segmentation results to JSON format

## Project dependencies

Before using Milvus Segment Generator, ensure you have:
* Python 3.8 or higher
* pip package manager
* HuggingFace account (for downloading Gemma model tokenizer)


## Instructions for use

### Installation

1. Clone the repository:
```bash
git clone https://github.com/OpenPecha/milvus_segment_generator.git
cd milvus_segment_generator
```

2. Install dependencies:
```bash
pip install -e .
```

This will install:
- `transformers>=4.30.0` - HuggingFace Transformers library
- `torch>=2.0.0` - PyTorch (required by transformers)

3. (Optional) For development, install dev dependencies:
```bash
pip install -e ".[dev]"
```

### Quick Start

```python
from milvus_segment_generator import segment_text, segment_text_to_json

# Segment Tibetan text
tibetan_text = "བཅོམ་ལྡན་འདས། དེ་བཞིན་གཤེགས་པ།"
spans = segment_text(tibetan_text, lang="tibetan", segment_size=2000)
print(spans)
# [{"span": {"start": 0, "end": 15}}, {"span": {"start": 15, "end": 30}}]

# Save to JSON file
segment_text_to_json(
    tibetan_text,
    lang="bo",
    output_path="output/segments.json",
    segment_size=2000
)
```

### Supported Languages

- **Tibetan**: `tibetan`, `bo`
- **English**: `english`, `en`
- **Chinese**: `chinese`, `zh`

### API Reference

#### `segment_text(text, lang, segment_size=1990)`

Tokenize and segment text into chunks.

**Parameters:**
- `text` (str): Input text to segment
- `lang` (str): Language code
- `segment_size` (int): Maximum tokens per segment (default: 1990)

**Returns:**
- List of dictionaries with `span` containing `start` and `end` character offsets

#### `segment_text_to_json(text, lang, output_path, segment_size=1990)`

Segment text and save to JSON file.

**Parameters:**
- `text` (str): Input text to segment
- `lang` (str): Language code
- `output_path` (str | Path): Output file path
- `segment_size` (int): Maximum tokens per segment (default: 1990)

**Returns:**
- Path object pointing to the created JSON file

### Troubleshooting

<table>
  <tr>
   <td><strong>Issue</strong></td>
   <td><strong>Solution</strong></td>
  </tr>
  <tr>
   <td>ImportError: No module named 'transformers'</td>
   <td>Install transformers: <code>pip install transformers torch</code></td>
  </tr>
  <tr>
   <td>HuggingFace authentication error</td>
   <td>Login to HuggingFace: <code>huggingface-cli login</code> or set <code>HF_TOKEN</code> environment variable</td>
  </tr>
  <tr>
   <td>ValueError: No delimiter found within window</td>
   <td>Your text segment doesn't contain any delimiters within the segment_size. Add appropriate punctuation or increase segment_size.</td>
  </tr>
  <tr>
   <td>Model download is slow</td>
   <td>The first run downloads the Gemma tokenizer (~500MB). Subsequent runs use cached version.</td>
  </tr>
</table>

### Environment Variables

- `HF_TOKEN`: HuggingFace API token for model access
- `TRANSFORMERS_CACHE`: Directory for caching downloaded models (default: `~/.cache/huggingface`)


## Contributing guidelines
If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).


## Additional documentation

For more information:
* [New API Documentation](README_NEW_API.md) - Detailed API reference and architecture
* [Test Documentation](tests/README.md) - Testing guidelines and structure
* [Examples](examples/) - Usage examples for different languages


## How to get help
* File an issue.
* Email us at openpecha[at]gmail.com.
* Join our [discord](https://discord.com/invite/7GFpPFSTeA).


## Terms of use
Milvus Segment Generator is licensed under the [MIT License](/LICENSE.md).
