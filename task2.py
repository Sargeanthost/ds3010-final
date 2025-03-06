from transformers import pipeline
import spacy
import polars as pl
from pathlib import Path

spacy.require_gpu()

sentiment_pipeline = pipeline("sentiment-analysis", model="LiYuan/amazon-review-sentiment-analysis", batch_size=8)

nlp = spacy.load("en_core_web_md")


def extract_negative_sentences(text: str):
    """Extract full sentences with negative sentiment."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents] # collect them into a list

    if not sentences: 
        # This shouldn't ever happen on review. tips seem to be way shorter so may not have anything negative
        return []

    sentiments = sentiment_pipeline(sentences)

    negative_sentences = [sent for sent, sentiment in zip(sentences, sentiments) if sentiment["label"] in {"1 star", "2 stars"} and sentiment["score"] > 0.75]

    return negative_sentences


def apply_absa_extraction(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a new column 'absa' containing negative sentences."""
    return df.with_columns(pl.col("text").map_elements(extract_negative_sentences, return_dtype=pl.List(pl.Utf8())).alias("absa"))


task2_review_path = Path(__file__).parent / "data" / "task2" / "task_2_review.parquet"
task2_tip_path = Path(__file__).parent / "data" / "task2" / "task_2_tip.parquet"

review_df = pl.read_parquet(task2_review_path)
tip_df = pl.read_parquet(task2_tip_path)

review_df = apply_absa_extraction(review_df.head(1)) # change to what you need, but takes a long time. max 100
review_df.write_parquet("./data/task2/task2_absa_review.parquet")

tip_df = apply_absa_extraction(tip_df.head(1)) # change to what you need, but takes a long time. max 100
tip_df.write_parquet("./data/task2/task2_absa_tip.parquet")
