import polars as pl
import yake
import nltk
from nltk import pos_tag, word_tokenize
from pathlib import Path

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")

pl.Config.set_fmt_table_cell_list_len(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(5000)
pl.Config.set_tbl_rows(-1)

review_path = Path(__file__).parent / "data" / "task1_business_review.parquet"
tip_path = Path(__file__).parent / "data" / "task1_business_tip.parquet"

output_path = Path(__file__).parent / "data" / "task1"

if not output_path.exists():
    output_path.mkdir()

review_df = pl.read_parquet(review_path)
tip_df = pl.read_parquet(tip_path)

tip_df = tip_df.filter(tip_df["business_id"].is_in(review_df["business_id"]))

# done to keep other cols
review_df = review_df.group_by("business_id").agg(
    pl.first("name"),
    pl.first("state"),
    pl.first("stars"),
    pl.first("review_count"),
    pl.first("categories"),
    pl.col("text").str.concat("\n"),
)

tip_df = tip_df.group_by("business_id").agg(
    pl.first("name"),
    pl.first("state"),
    pl.first("stars"),
    pl.first("review_count"),
    pl.first("categories"),
    pl.col("text").str.concat("\n"),
)


def filter_pos(keywords):
    """Remove keywords that contain verbs and adjectives and adverbs."""
    tokenized_keywords = [word_tokenize(kw) for kw in keywords]
    tagged_keywords = [pos_tag(tokens) for tokens in tokenized_keywords]

    filtered_words: list[str] = []
    for kw, tag in zip(keywords, tagged_keywords):
        if " " in kw:
            filtered_words.append(kw)
        elif not (any((t[1].startswith("VB") or t[1].startswith("JJ") or t[1].startswith("RB")) for t in tag)):
            filtered_words.append(kw)

    return filtered_words


def filter_keywords(keywords):
    """Remove lower n-grams if they appear in longer n-grams."""
    banned_keywords = {"time", "place", "location", "drive", "great", "good", "minutes", "minute", "times"}
    filtered_keywords = [kw for kw in keywords if (kw.lower() not in banned_keywords) and (not any(kw.lower() in longer_kw.lower() and kw.lower() != longer_kw.lower() for longer_kw in set(keywords)))]
    return filtered_keywords


def extract_keywords_yake(text, max_keywords=20) -> list[str]:
    custom_kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=max_keywords, windowsSize=4)
    # second param is importance, lower is more important
    keywords = [kw[0] for kw in custom_kw_extractor.extract_keywords(text)]
    keywords = filter_pos(filter_keywords(keywords))
    return keywords


def apply_keyword_extraction(df):
    return df.with_columns(
        pl.col("text").map_elements(extract_keywords_yake, return_dtype=pl.List(pl.String)).alias("keywords"),
    )


review_df = apply_keyword_extraction(review_df.head(1000))

tip_df = tip_df.filter(tip_df["business_id"].is_in(review_df["business_id"])).head(1000)
tip_df = apply_keyword_extraction(tip_df)

review_df.write_parquet(output_path / "review_keywords_task1.parquet")
tip_df.write_parquet(output_path / "tip_keywords_task1.parquet")
