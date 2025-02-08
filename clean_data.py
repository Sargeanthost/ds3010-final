import os 
os.environ["POLARS_MAX_THREADS"] = "7"

import polars as pl
from pathlib import Path

business_path = Path(__file__).parent / "data" / "yelp_academic_dataset_business.json"
review_path = Path(__file__).parent / "data" / "yelp_academic_dataset_review.json"
tip_path = Path(__file__).parent / "data" / "yelp_academic_dataset_tip.json"

clean_data_path = Path(__file__).parent / "data" / "california_dataset_all.json"

business_df = pl.scan_ndjson(business_path)
review_df = pl.scan_ndjson(review_path)
tip_df = pl.scan_ndjson(tip_path)


if clean_data_path.exists():
    print("Overwriting existing clean dataset.")

review_df.join(business_df, on="business_id").join(tip_df, on="business_id").filter(
    pl.col("state").eq("CA") & pl.col("review_count").ge(25)
).unique().collect(streaming=True).write_ndjson(clean_data_path)

print("Done writing ndjson")
