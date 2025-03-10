import polars as pl

df = pl.read_parquet("./data/task2/task2_absa_review.parquet", parallel="none")
batch_size = 25
desired_batch = -1
# carlos: 0
# pooja: 25
# tim: 50
# josiah 75
if desired_batch >= 0:
    for i in range(0, len(df), batch_size):
        if i == desired_batch:
            df.slice(i, batch_size).with_columns(
                pl.col("text").str.slice(0, 3500)  # Truncate text column to 3500 characters. Each text can be over 32kb large
            )[["business_id", "name", "text"]].write_csv(f"annotated_review_slice{i}_task2.csv")

pl.read_parquet("./data/task2/task2_absa_tip.parquet", parallel="none")[["business_id", "name", "text"]].write_csv("annotated_tip_task2.csv")
