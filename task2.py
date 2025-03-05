# task 2
from transformers import pipeline
import spacy
import polars as pl
from read_df import get_dfs

spacy.require_gpu()


sentiment_pipeline = pipeline("sentiment-analysis", model="LiYuan/amazon-review-sentiment-analysis")

nlp = spacy.load("en_core_web_md")


def extract_adj_noun_phrases(doc):
    """Extract adjective-noun phrases (e.g., 'terrible service')."""
    phrases = []
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            phrases.append(f"{token.text} {token.head.text}")
    return phrases


def extract_verb_phrases(doc):
    """Extract verb-based sentiment phrases (e.g., 'waiter ignored us')."""
    phrases = []
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in {"ROOT", "xcomp"}:
            subject = [child.text for child in token.children if child.dep_ == "nsubj"]
            obj = [child.text for child in token.children if child.dep_ in {"dobj", "attr"}]
            if subject and obj:
                phrases.append(f"{' '.join(subject)} {token.text} {' '.join(obj)}")
    return phrases


def extract_sentiment_for_chunks(text: str):
    """Extract noun chunks, adjective-noun phrases, and verb-based phrases for sentiment analysis."""
    doc = nlp(text)
    results = []

    phrases = set()
    phrases.update(chunk.text for chunk in doc.noun_chunks)
    phrases.update(extract_adj_noun_phrases(doc))
    phrases.update(extract_verb_phrases(doc))

    sentiments = sentiment_pipeline(list(phrases))

    results = [phrase for phrase, sentiment in zip(phrases, sentiments) if (sentiment["label"] == "1 star" or sentiment["label"] == "2 stars") and sentiment["score"] > 0.75]

    return results


def apply_absa_extraction(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a new column absa that is a list of negative phrases in the text"""
    return df.with_columns(pl.col("text").map_elements(extract_sentiment_for_chunks, return_dtype=pl.List(pl.Utf8())).alias("absa"))


# MAKE SURE that you do not run this right after running task 1, as the dataframes are modified in task 1 and need to be reinitilized by running the second previous cell
review_df, tip_df = get_dfs()
review_df = apply_absa_extraction(review_df.head(1))
print(review_df[["name", "stars", "absa"]])

# for chunk, label, score in sentiments:
#     print(f"Chunk: '{chunk}' -> Sentiment: {label} (Probability: {score:.2f})")
