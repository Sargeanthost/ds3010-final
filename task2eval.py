import polars as pl

pl.Config.set_fmt_table_cell_list_len(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(5000)
pl.Config.set_tbl_rows(-1)


def compute_accuracy(human_csv: str, absa_parquet: str):
    human_df = pl.read_csv(human_csv)
    absa_df = pl.read_parquet(absa_parquet)

    assert "business_id" in human_df.columns, "Missing 'business_id' in human CSV"
    assert "human" in human_df.columns, "Missing 'human' column in human CSV"
    assert "business_id" in absa_df.columns, "Missing 'business_id' in ABSA Parquet"
    assert "absa" in absa_df.columns, "Missing 'absa' column in ABSA Parquet"

    # Compute difference by exploding the list of sentences from each df and seeing if we can do a join on the human sentences, since we know these are correct.
    human_split_df = human_df.with_columns(pl.col("human").str.split(by=r"||").alias("human_sentence")).select(["business_id", "human_sentence"])
    human_exploded = human_split_df.explode("human_sentence")
    absa_exploded = absa_df.with_columns(pl.col("absa").list.eval(pl.element().str.strip_chars())).explode("absa").select(["business_id", "absa"]).rename({"absa": "absa_sentence"})

    # This will give us human sentences that were also in the absa sentence list
    matched = human_exploded.join(absa_exploded, left_on=["business_id", "human_sentence"], right_on=["business_id", "absa_sentence"])
    # Now we can compute how many remain in the matched vs how many were in the original exploded df
    human_exploded_count = human_exploded.group_by("business_id").agg(pl.col("human_sentence").count().alias("sentence_count"))
    matched_count = matched.group_by("business_id").agg(pl.col("human_sentence").count().alias("sentence_count"))
    # Take the ratio. We can do global accuracy outside this func.
    result_df = human_exploded_count.join(matched_count, on="business_id", how="inner")

    result_df = result_df.with_columns((pl.col("sentence_count_right") / pl.col("sentence_count")).alias("sentence_ratio"))
    result_df = result_df.select(["business_id", "sentence_ratio"])

    return result_df


accuracy_df = compute_accuracy("./data/task2/annotated_review_task2.csv", "./data/task2/task2_absa_review.parquet")
print(accuracy_df)
print(f"Global accuracy : {accuracy_df.with_columns(pl.col('sentence_ratio').mean().alias('global'))['global'][0]}")

# Optionally, save the results to a CSV file
# accuracy_df.write...
