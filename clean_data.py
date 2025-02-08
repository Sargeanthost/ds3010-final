import os

os.environ["POLARS_MAX_THREADS"] = "15"  # Change this based on your number of system threads (logical processors in Windows) (cores*threads per core in Linux)

import polars as pl
from pathlib import Path

pl.Config.set_tbl_cols(900)
pl.Config.set_tbl_rows(900)

business_path = Path(__file__).parent / "data" / "yelp_academic_dataset_business.json"
review_path = Path(__file__).parent / "data" / "yelp_academic_dataset_review.json"

task1_business_review_path = Path(__file__).parent / "data" / "task1_business_review.json"

business_df = pl.scan_ndjson(business_path)
review_df = pl.scan_ndjson(review_path)


# Examine data
print(business_df.filter(pl.col("review_count").ge(35)).group_by(pl.col("state")).len().sort(pl.col("len")).collect(streaming=True))

# Get just PA business that have  or more reviews
business_df.filter(pl.col("state").eq("PA") & pl.col("review_count").ge(35)).join(review_df, on="business_id").unique().collect(streaming=True).write_ndjson(task1_business_review_path)

# Cant merge tip into business and review df since each businesses' tips would be be a column per tip per review, and take up massive amounts of memory.
# Can make business and tip df if needed
