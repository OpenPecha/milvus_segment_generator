"""Example usage of the milvus_segment_generator library."""

from pathlib import Path
from milvus_segment_generator import segment_text, segment_text_to_json, list_supported_languages


def main():
    # Show supported languages
    print("Supported languages:")
    print(list_supported_languages())
    print()
    
    # Example 1: Segment Tibetan text
    tibetan_text = "བཅོམ་ལྡན་འདས་མ་ཤེས་རབ་ཀྱི་ཕ་རོལ་ཏུ་ཕྱིན་པའི་སྙིང་པོ། །"
    
    print("Example 1: Tibetan text segmentation")
    spans = segment_text(tibetan_text, lang="tibetan", segment_size=100)
    print(f"Text: {tibetan_text}")
    print(f"Spans: {spans}")
    print()
    
    # Example 2: Save to JSON file
    print("Example 2: Save Tibetan segmentation to JSON")
    output_path = segment_text_to_json(
        tibetan_text,
        lang="bo",  # ISO 639-1 code also works
        output_path="tibetan_segments.json",
        segment_size=100
    )
    print(f"Saved to: {output_path}")
    print()
    
    # Example 3: English text
    english_text = "Hello world. This is a test. How are you?"
    
    print("Example 3: English text segmentation")
    spans = segment_text(english_text, lang="english", segment_size=50)
    print(f"Text: {english_text}")
    print(f"Spans: {spans}")
    print()
    
    # Example 4: Chinese text
    chinese_text = "你好世界。这是一个测试。你好吗？"
    
    print("Example 4: Chinese text segmentation")
    spans = segment_text(chinese_text, lang="zh", segment_size=50)
    print(f"Text: {chinese_text}")
    print(f"Spans: {spans}")
    print()
    
    # Example 5: Read from file and segment
    if Path("tests/data/input.txt").exists():
        print("Example 5: Segment from file")
        text = Path("tests/data/input.txt").read_text(encoding="utf-8")
        output_path = segment_text_to_json(
            text,
            lang="tibetan",
            output_path="tests/data/output.json",
            segment_size=1990
        )
        print(f"Segmented file saved to: {output_path}")


if __name__ == "__main__":
    main()

