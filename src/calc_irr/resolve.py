"""Turn reviewed coding exports into final labels -- no calculator state needed."""

from pathlib import Path
from string import ascii_lowercase

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


# Kept in the resolved file whenever the coding file has them, along with
# every coder flag column (any_marked_<name>), which goes before coder_notes
DEFAULT_RESOLVED_COLUMNS = [
    "title_missing",
    "abstract_missing",
    "title_non_english",
    "abstract_non_english",
    "coder_notes",
]
FLAG_PREFIX = "any_marked_"


def anonymize_notes(notes: Series, coders: list[str]) -> Series:
    """Replace "name: note" prefixes in a `coder_notes` column with "coder_a: note" etc.

    Coders are lettered in name order, so the same coder gets the same letter
    in every key's file as long as the coder list is the same.
    """
    aliases = {
        name: f"coder_{ascii_lowercase[i]}" if i < 26 else f"coder_{i + 1}"
        for i, name in enumerate(sorted(map(str, coders)))
    }

    def anonymize(text: object) -> object:
        if pd.isna(text):
            return text
        parts = []
        for part in str(text).split(" | "):
            name, sep, note = part.partition(": ")
            parts.append(
                f"{aliases[name]}: {note}" if sep and name in aliases else part
            )
        return " | ".join(parts)

    return notes.map(anonymize)


def resolve_labels(
    coding_df: DataFrame,
    completed: str | Path | DataFrame,
    item_id_col: str = "item_id",
    title_col: str | None = "title",
    text_col: str | None = "text",
    extra_columns: list[str] | None = None,
) -> tuple[DataFrame, DataFrame]:
    """Build the final label per item from a coding export and a completed review file.

    `coding_df` is `<key>_coding.csv`: items the coders agreed on already
    have `resolved` filled in. `completed` is `<key>_to_review.csv` after
    discussion, with `resolved` filled in by hand. A hand-filled value is
    used only if it is numeric and one of the categories the coders
    voted with; blank, non-numeric, or out-of-range values are skipped.

    Returns `(resolved, skipped)`: `resolved` has every item from
    `coding_df` with key/item_id/title/text/resolved/decision, then whichever
    of DEFAULT_RESOLVED_COLUMNS (text-quality flags, any_marked_<name>
    coder flags, coder_notes) the coding file has -- no coder votes -- and `resolved`
    left blank for skipped items. `decision` is "unanimous" when all coders agreed and
    "adjudicated" when the label had to come from the review file.
    `skipped` lists those items with the raw value that was rejected (NaN
    if the item was missing or blank).

    `extra_columns` adds more columns after those, taken from the completed
    review file where it has them (e.g. a "Reason" column added during
    discussion) and from `coding_df` otherwise. `coder_notes` is anonymized
    on the way: coder names become coder_a, coder_b, ...
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
        c
        for c in coding_df.select_dtypes("number").columns
        if c not in meta_cols
        and not c.endswith(("_missing", "_non_english"))
        and not c.startswith(FLAG_PREFIX)
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

    flag_cols = [c for c in coding_df.columns if c.startswith(FLAG_PREFIX)]
    defaults = DEFAULT_RESOLVED_COLUMNS[:-1] + flag_cols + DEFAULT_RESOLVED_COLUMNS[-1:]
    for col in defaults:
        if col in coding_df.columns:
            resolved[col] = coding_df[col]

    for col in extra_columns or []:
        if col in resolved.columns:
            continue
        if col not in coding_df.columns and col not in completed_df.columns:
            raise KeyError(f"resolved column not found in coding or review file: {col}")
        values = Series(pd.NA, index=coding_df.index, dtype="object")
        if col in completed_df.columns:
            reviewed = completed_df.set_index(item_id_col)[col]
            values = coding_df[item_id_col].map(reviewed)
        if col in coding_df.columns:
            values = values.combine_first(coding_df[col])
        resolved[col] = values

    if "coder_notes" in resolved.columns:
        resolved["coder_notes"] = anonymize_notes(resolved["coder_notes"], coder_cols)

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
