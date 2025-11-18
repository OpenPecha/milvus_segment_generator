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
    text = "頂禮尊聖救度母頂禮救度速勇母目如電光剎那照三世界尊淚所生蓮花蕊開現端顏頂禮秋宵無垢月百聚圓滿容顏母如千星宿俱時聚殊勝威光熠射母頂禮金藍水中生蓮花手中嚴飾母布施精進難行靜忍辱禪定行境母頂禮猶如佛頂髻勝伏一切行境母無餘波羅密多行圓滿佛子所依母頂禮杜達拉吽字遍及欲方虛空母蓮足遍履七世界悉能無餘勾召母頂禮釋梵自在天火風神眾供養母部多起屍及尋香藥叉眾等禮讚母頂禮哲沛咒音吼於彼惑障能摧母左足展踏右足屈智火烈焰熾然母頂禮怖畏杜列音勇猛魔軍盡摧母於蓮花面作顰眉無餘摧滅眾敵母頂禮三寶所表徵手印心間嚴飾母無餘各方輪圓飾自身光聚熾耀母頂禮威嚴頭飾光以及歡悅迭現母最喜笑聲杜達拉魔及世間降伏母頂禮大地眾守護諸神皆得召伏母面搖顰眉心吽字度脫一切貧苦母頂禮猶如盡劫火熾然火鬘中住母半趺坐中轉法輪喜悅遍摧怨敵母頂禮以手心鎮壓且以足踩大地母目現顰眉吽字光七惡逐次摧壞母頂禮入安樂柔善涅槃寂靜行境母真實梭哈嗡字等極能滅大罪障母頂禮喜悅法輪轉其怨敵身盡摧母十字真言妙嚴佈明咒吽光度脫母頂禮速疾足踩踏吽相種子字性母須彌比傑曼達拉三世間眾撼動母頂禮如天海行相月亮手中執持母念誦達拉及沛字諸毒無餘盡除母頂禮天眾王及天人非人等所依母威德歡悅之堅鎧鬥諍惡夢滅除母頂禮圓滿如日月雙眼明光普照母念誦哈拉杜達拉惡毒疫癘盡除母頂禮三自性莊嚴真實息災力具母魑魅起屍藥叉等最極迅速摧壞母此乃贊頌根本咒禮敬文共二十一"
    text = text.replace("\n", "")
    text = text.replace("་ ", "")
    lang = 'zh'
    segment_size = 1990
    spans, segments = get_segmented_text(text, lang, segment_size)
    output_path = Path("data/output") / f"{text_title}.txt"
    output_path.write_text(segments, encoding="utf-8")
    print(f"Done {text_title}")
    print("Done all")