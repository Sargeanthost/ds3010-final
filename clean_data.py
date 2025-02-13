import os

os.environ["POLARS_MAX_THREADS"] = (
    "7"  # Change this based on your number of system threads (logical processors in Windows) (cores*threads per core in Linux)
)

import polars as pl
from pathlib import Path

pl.Config.set_tbl_cols(900)
pl.Config.set_tbl_rows(900)

business_path = Path(__file__).parent / "data" / "yelp_academic_dataset_business.json"
review_path = Path(__file__).parent / "data" / "yelp_academic_dataset_review.json"
tip_path = Path(__file__).parent / "data" / "yelp_academic_dataset_tip.json"

task1_business_review_path = Path(__file__).parent / "data" / "task1_business_review.json"
task1_business_tip_path = Path(__file__).parent / "data" / "task1_business_tip.json"

business_df = pl.scan_ndjson(business_path)
review_df = pl.scan_ndjson(review_path)
tip_df = pl.scan_ndjson(tip_path)

# Examine data
print(
    business_df.filter(pl.col("review_count").ge(55))
    .group_by(pl.col("state"))
    .len()
    .sort(pl.col("len"))
    .collect(streaming=True)
)

# Filter data
business_df = business_df.filter(pl.col("state").eq("PA") & pl.col("review_count").ge(55))
review_df = review_df.filter(pl.col("text").str.len_chars().ge(25))
tip_df = tip_df.filter(pl.col("text").str.len_chars().get(5))

# write out . need to check if collecting here is fine.
business_df.join(review_df, on="business_id").unique().collect(streaming=True).write_parquet(task1_business_review_path)

business_df.join(tip_df, on="business_id").unique().collect(streaming=True).write_parquet(task1_business_tip_path)
