from read_df import get_dfs
from pathlib import Path

task_2_review_path = Path("./data/task2/task_2_review.parquet")
task_2_tip_path = Path("./data/task2/task_2_tip.parquet")

review_df, tip_df = get_dfs()

review_df = review_df.head(100)
tip_df = tip_df.filter(tip_df["business_id"].is_in(review_df["business_id"]))

review_df.write_parquet(task_2_review_path)
tip_df.write_parquet(task_2_tip_path)
