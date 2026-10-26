"""Turn reviewed coding exports into final labels -- no calculator state needed."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame


def resolve_labels(
    coding_df: DataFrame,
    completed: str | Path | DataFrame,
    item_id_col: str = "item_id",
    title_col: str | None = "title",
    text_col: str | None = "text",
) -> tuple[DataFrame, DataFrame]:
    """Build the final label per item from a coding export and a completed review file.

    `coding_df` is `<key>_coding.csv`: items the coders agreed on already
    have `resolved` filled in. `completed` is `<key>_to_review.csv` after
    discussion, with `resolved` filled in by hand. A hand-filled value is
    used only if it is numeric and one of the categories the coders
    voted with; blank, non-numeric, or out-of-range values are skipped.

    Returns `(resolved, skipped)`: `resolved` has every item from
    `coding_df` with just key/item_id/title/text/resolved/decision -- no
    coder votes or discussion notes -- and `resolved` left blank for
    skipped items. `decision` is "unanimous" when all coders agreed and
    "adjudicated" when the label had to come from the review file.
    `skipped` lists those items with the raw value that was rejected (NaN
    if the item was missing or blank).
    """
    completed_df = (
        completed if isinstance(completed, DataFrame) else pd.read_csv(completed)
    )

    meta_cols = {
        "key",
        item_id_col,
        "disagreement_level",
        "resolved",
    }
    coder_cols = [
        c for c in coding_df.select_dtypes("number").columns if c not in meta_cols
    ]
    categories = set(pd.unique(coding_df[coder_cols].to_numpy().ravel()))
    categories.discard(np.nan)

    raw_values = completed_df.set_index(item_id_col)["resolved"]
    disputed = coding_df["resolved"].isna()
    raw = coding_df[item_id_col].map(raw_values).where(disputed)
    discussed = pd.to_numeric(raw, errors="coerce")
    discussed = discussed.where(discussed.isin(categories))

    output_cols = ["key", item_id_col] + [
        c for c in (title_col, text_col) if c and c in coding_df.columns
    ]
    resolved = coding_df[output_cols].copy()
    resolved["resolved"] = coding_df["resolved"].fillna(discussed).astype("Int64")
    resolved["decision"] = disputed.map({True: "adjudicated", False: "unanimous"})

    skipped = coding_df.loc[disputed & discussed.isna(), ["key", item_id_col]]
    skipped["raw_value"] = raw[skipped.index]
    return resolved, skipped.reset_index(drop=True)


def strip_coder_columns(
    resolved_df: DataFrame,
    item_id_col: str = "item_id",
    title_col: str = "title",
    text_col: str = "text",
    doi_source: DataFrame | None = None,
    doi_col: str = "doi",
) -> DataFrame:
    """Strip a resolved export down to a shareable subset with no coder identities.

    Keeps item_id/title/abstract (renamed from text_col)/disagreement_level/
    resolved, and doi if `doi_source` (an item_id -> doi lookup) is given.
    Drops every per-coder vote column and any discussion columns (e.g.
    notes, Final Decision, Reason) -- those stay available in the coding
    and resolutions files this was built from.
    """
    export = resolved_df[
        [item_id_col, title_col, text_col, "disagreement_level", "resolved"]
    ].rename(columns={text_col: "abstract"})

    if doi_source is not None:
        export = export.merge(
            doi_source[[item_id_col, doi_col]].drop_duplicates(subset=item_id_col),
            on=item_id_col,
            how="left",
        )

    return export
