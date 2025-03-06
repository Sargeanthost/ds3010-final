import polars as pl

df = pl.read_parquet("./data/task2/task2_absa_review.parquet", parallel="none")
batch_size = 25
desired_batch = 0
for i in range(0, len(df), batch_size):
    if i == desired_batch:
        df.slice(i, batch_size).with_columns(
            pl.col("text").str.slice(0, 3000)  # Truncate text column to 1500 characters
        )[["business_id", "name", "text"]].write_csv("assigned_data.csv")
