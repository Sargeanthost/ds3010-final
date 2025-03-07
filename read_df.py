import polars as pl
from pathlib import Path

pl.Config.set_fmt_table_cell_list_len(100)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(5000)
pl.Config.set_tbl_rows(-1)

review_path = Path(__file__).parent / "data" / "task1_business_review.parquet"
tip_path = Path(__file__).parent / "data" / "task1_business_tip.parquet"



def get_dfs() -> tuple[pl.DataFrame, pl.DataFrame]:
    review_df = pl.read_parquet(review_path)

    tip_df = pl.read_parquet(tip_path)

    tip_df = tip_df.filter(tip_df["business_id"].is_in(review_df["business_id"]))

    # done to keep other cols
    review_df = review_df.group_by("business_id").agg(
        pl.first("name"),
        pl.first("state"),
        pl.first("stars"),
        pl.first("review_count"),
        pl.first("categories"),
        pl.col("text").str.concat("\n"),
    )

    tip_df = tip_df.group_by("business_id").agg(
        pl.first("name"),
        pl.first("state"),
        pl.first("stars"),
        pl.first("review_count"),
        pl.first("categories"),
        pl.col("text").str.concat("\n"),
    )

    return review_df.sort("business_id"), tip_df.sort("business_id") #doing this for consistance