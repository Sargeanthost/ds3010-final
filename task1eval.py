import polars as pl
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

pl.Config.set_fmt_table_cell_list_len(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(5000)
pl.Config.set_tbl_rows(-1)

# Load Processed Data for Evaluation
review_keywords_path = Path("./data/task1/review_keywords_task1.parquet")
tip_keywords_path = Path("./data/task1/tip_keywords_task1.parquet")

review_keywords_df = pl.read_parquet(review_keywords_path)
tip_keywords_df = pl.read_parquet(tip_keywords_path)

# Start with 10 Businesses for Evaluation
review_keywords_df = review_keywords_df
tip_keywords_df = tip_keywords_df

def preprocess_text(text):
    """Tokenize, lowercase, and lemmatize words for better comparison."""
    words = word_tokenize(text.lower().strip())  # Tokenize into words
    return set(lemmatizer.lemmatize(word) for word in words)  # Convert to base form (e.g., "nails" → "nail")

def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2) if set1 | set2 else 1  # Avoid division by zero
    return intersection / union

def fuzzy_match(word1, word2, threshold=0.75):
    """Fuzzy match two words using Levenshtein-based similarity."""
    return SequenceMatcher(None, word1, word2).ratio() >= threshold

def compute_accuracy(keywords, categories):
    """
    Improved Accuracy Calculation:
    - Uses **lemmatization** to match plural/singular variations.
    - Matches **multi-word phrases** by Jaccard similarity.
    - Uses **fuzzy matching** to handle slight variations.
    - Penalizes only **missing words in categories** but not extra keywords.
    """
    if not keywords or not categories:
        return 0.0  # No valid accuracy if either is missing

    # Process keywords and categories for better comparison
    keywords_set = preprocess_text(" ".join(keywords))  # Convert list to string for tokenization
    categories_set = preprocess_text(categories.replace(", ", ","))  # Clean categories
    
    # Exact match count
    exact_matches = len(keywords_set & categories_set)

    # Fuzzy matches (words that are close but not exact)
    fuzzy_matches = sum(1 for k in keywords_set for c in categories_set if fuzzy_match(k, c))

    # Jaccard similarity for phrase-level match
    phrase_similarity = jaccard_similarity(keywords_set, categories_set)

    # Total score: Exact matches + Fuzzy matches + Weighted phrase similarity
    total_matches = exact_matches + fuzzy_matches + phrase_similarity

    # Normalize by the number of relevant category words
    total_relevant = len(categories_set) if categories_set else 1  # Avoid division by zero

    return total_matches / total_relevant  # Accuracy based on how many category words we match

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

# Compute Overall Accuracy and Save Results
overall_accuracy_reviews = review_keywords_df["accuracy"].mean()
overall_accuracy_tips = tip_keywords_df["accuracy"].mean()
review_keywords_df.write_parquet("./data/task1/reveiw_keywords_accuracy_task1.parquet")
tip_keywords_df.write_parquet("./data/task1/tip_keywords_accuracy_task1.parquet")

# Print Review Keywords DataFrame with Selected Columns
print("\nReview Keywords DataFrame:")
print(review_keywords_df.select(["business_id", "categories", "keywords", "accuracy"]))

# Print Tip Keywords DataFrame with Selected Columns
print("\nTip Keywords DataFrame:")
print(tip_keywords_df.select(["business_id", "categories", "keywords", "accuracy"]))

# Print Business-Level Accuracy
print("\nBusiness-Level Accuracy Results:")
print(review_keywords_df.select(["business_id", "accuracy"]))

# Print Overall Accuracy
print("\nOverall Accuracy Across Businesses:")
print(f"Reviews Accuracy: {overall_accuracy_reviews:.2%}")
print(f"Tips Accuracy: {overall_accuracy_tips:.2%}")
