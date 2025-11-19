"""Example usage of the milvus_segment_generator library."""

from pathlib import Path
from milvus_segment_generator import segment_text, segment_text_to_json, list_supported_languages


def get_segmented_text(text: str, lang: str, segment_size: int):
    spans, segments = segment_text(text, lang, segment_size)
    return spans, segments


if __name__ == "__main__":
    # text_dirs = list(Path("data/input").iterdir())
    # text_dirs.sort()
    # for text_dir in text_dirs:
    #     text_title = text_dir.stem
    #     text_path = list(text_dir.glob("*.txt"))[0]
    text = Path("data/input.txt").read_text(encoding="utf-8")
    text = text.replace("་ ", "")
    lang = 'bo'
    segment_size = 1990
    spans, segments = get_segmented_text(text, lang, segment_size)
    output_path = Path("data") / "output.txt"
    output_path.write_text(segments, encoding="utf-8")
    print(f"Done {text_title}")
    print("Done all")