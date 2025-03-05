import polars as pl
from pathlib import Path
pl.Config.set_fmt_table_cell_list_len(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(5000)
pl.Config.set_tbl_rows(-1)

# Load Processed Data for Evaluation
review_keywords_path = Path("./data/task 1/review_keywords_task1.parquet")
tip_keywords_path = Path("./data/task 1/tip_keywords_task1.parquet")

review_keywords_df = pl.read_parquet(review_keywords_path)
tip_keywords_df = pl.read_parquet(tip_keywords_path)

# Start with 10 Businesses for Evaluation
review_keywords_df = review_keywords_df.head(10)
tip_keywords_df = tip_keywords_df.head(10)

def compute_accuracy(keywords, categories):
    """Calculate accuracy as the fraction of extracted keywords that match business categories."""
    if not keywords or not categories:
        return 0.0  # No valid accuracy if either is missing

    # Convert both to lowercase and remove spaces to standardize comparison
    keywords_set = set(kw.lower().strip() for kw in keywords)  # Clean extracted keywords
    categories_set = set(c.lower().strip() for c in categories.replace(", ", ",").split(","))  # Clean categories

    match_count = len(keywords_set & categories_set)  # Count matching words
    total_keywords = len(keywords_set) if keywords_set else 1  # Avoid division by zero

    return match_count / total_keywords  # Accuracy ratio

# Compute Accuracy for Reviews
review_keywords_df = review_keywords_df.with_columns(
    pl.struct(["keywords", "categories"]).map_elements(
        lambda row: compute_accuracy(row["keywords"], row["categories"]),
        return_dtype=pl.Float64  # Ensure Polars handles float return properly
    ).alias("accuracy")
)

# Compute Accuracy for Tips
tip_keywords_df = tip_keywords_df.with_columns(
    pl.struct(["keywords", "categories"]).map_elements(
        lambda row: compute_accuracy(row["keywords"], row["categories"]),
        return_dtype=pl.Float64
    ).alias("accuracy")
)

# Compute Overall Accuracy
overall_accuracy_reviews = review_keywords_df["accuracy"].mean()
overall_accuracy_tips = tip_keywords_df["accuracy"].mean()

# Print Full DataFrames for Review and Tips (Selected Columns Only)
print("\nReview Keywords DataFrame:")
print(review_keywords_df.select(["business_id", "categories", "keywords", "accuracy"]))

print("\nTip Keywords DataFrame:")
print(tip_keywords_df.select(["business_id", "categories", "keywords", "accuracy"]))

# Print Business-Level Accuracy
print("\nBusiness-Level Accuracy Results:")
print(review_keywords_df.select(["business_id", "accuracy"]))

# Print Overall Accuracy
print("\nOverall Accuracy Across Businesses:")
print(f"Reviews Accuracy: {overall_accuracy_reviews:.2%}")
print(f"Tips Accuracy: {overall_accuracy_tips:.2%}")
