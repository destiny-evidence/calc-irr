# how to identify difficult records
# nice display
# how to relay bad info?

from datetime import datetime
from enum import StrEnum
from html import escape
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import cohen_kappa_score


class KeyType(StrEnum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class IrrCalculator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: DataFrame | None = None
    keys: list[str] = Field(default_factory=list)
    is_wide: bool | None = True
    item_id: str | None = "item_id"
    coder_id: str | None = "user_id"
    n_coders: int | None = None
    long_df: dict[str, DataFrame] = Field(default_factory=dict)
    wide_df: dict[str, DataFrame] = Field(default_factory=dict)

    def add_wide(self, key: str, wide_df: DataFrame) -> None:
        """Register an already wide-format DataFrame for a variable, skipping collapse/to_wide."""
        self.wide_df[key] = wide_df

    def add_gold(
        self,
        key: str,
        resolved: str | Path | DataFrame,
        gold_coder_id: str = "gold",
        value_column: str = "resolved",
    ) -> None:
        """Merge a resolved/gold label source into `key` as an extra coder column.

        `resolved` (a CSV path or an already-loaded DataFrame) must have an
        item id column matching self.item_id and a `value_column` with the
        final label per item. Once merged, gold_coder_id behaves like any
        other coder for kappa, alpha, coder_summary, and the priority/diff
        export -- for checking a newly-added coder against labels that were
        already adjudicated, without recomputing consensus from scratch.
        """
        wide_df = self._ensure_wide(key)

        gold_df = resolved if isinstance(resolved, DataFrame) else pd.read_csv(resolved)
        if self.item_id not in gold_df.columns or value_column not in gold_df.columns:
            raise ValueError(
                f"gold source must have {self.item_id!r} and {value_column!r} columns"
            )

        gold_series = (
            gold_df[[self.item_id, value_column]]
            .dropna(subset=[value_column])
            .rename(columns={value_column: gold_coder_id})
        )

        self.wide_df[key] = wide_df.merge(gold_series, on=self.item_id, how="left")

    def merge_resolutions(
        self,
        coding_df: DataFrame,
        resolutions: str | Path | DataFrame,
        item_id_col: str | None = None,
    ) -> DataFrame:
        """Combine a `<key>_coding.csv` export with a completed `<key>_to_review.csv` into final resolved data.

        `coding_df` is the full per-key export (e.g. from `generate_report`'s
        `<key>_coding.csv`) -- every fully-coded item, with `resolved` already
        filled in for items all coders agreed on. `resolutions` (a CSV path or
        DataFrame) is the `<key>_to_review.csv` subset after human review, with
        `resolved` now filled in plus any extra columns added during
        discussion (e.g. notes, Final Decision, Reason). Rows in `resolutions`
        overwrite/extend the matching rows in `coding_df`; already-agreed rows
        pass through unchanged.
        """
        item_id_col = item_id_col or self.item_id

        resolutions_df = (
            resolutions
            if isinstance(resolutions, DataFrame)
            else pd.read_csv(resolutions)
        )

        merged = coding_df.set_index(item_id_col)
        updates = resolutions_df.set_index(item_id_col)

        for col in updates.columns:
            if col not in merged.columns:
                merged[col] = pd.NA

        merged.update(updates)

        return merged.reset_index()

    def collapse_one_hot_annotation(self, key: str) -> None:
        if self.df is None:
            raise ValueError("df must be provided to collapse one-hot annotations")

        id_columns = [self.item_id, self.coder_id]
        selected_columns = [col for col in self.df.columns if col.startswith(f"{key}|")]
        if not selected_columns:
            raise KeyError(f"Key not found: {key}")

        # Set meta columns
        result = self.df[id_columns].copy()

        # Single category: convert directly
        if len(selected_columns) == 1:
            result[key] = self.df[selected_columns[0]].astype(int)
            self.long_df[key] = result
            return

        # Multiple one-hot columns: collapse to category index
        values = self.df[selected_columns]
        has_annotation = values.notna().any(axis=1)
        result[key] = np.nan

        result.loc[has_annotation, key] = (
            values.loc[has_annotation]
            .idxmax(axis=1)
            .str.rsplit("|", n=1)
            .str[-1]
            .astype("Int64")
        )

        self.long_df[key] = result
        return

    def _ensure_wide(self, key: str) -> DataFrame:
        """Resolve key to its wide form, running whichever prior steps are needed."""
        if key in self.wide_df:
            return self.wide_df[key]
        if key not in self.long_df:
            self.collapse_one_hot_annotation(key)
        return self.to_wide(key)

    def to_wide(self, key: str) -> DataFrame:
        wide_df: DataFrame = (
            self.long_df[key]
            .pivot(index=self.item_id, columns=self.coder_id, values=key)
            .reset_index()
        )
        self.wide_df[key] = wide_df
        return wide_df

    def pairwise_cohen_kappa(self, key: str) -> DataFrame:
        """Compute Cohen's kappa for every pair of coders for the given variable."""
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]

        rows = []
        for coder_a, coder_b in combinations(coders, 2):
            pair_df = wide_df[[coder_a, coder_b]].dropna()
            kappa = cohen_kappa_score(pair_df[coder_a], pair_df[coder_b])
            rows.append(
                {
                    "coder_a": coder_a,
                    "coder_b": coder_b,
                    "n": len(pair_df),
                    "cohen_kappa": kappa,
                }
            )

        return DataFrame(rows)

    def krippendorff_alpha(
        self, key: str, level_of_measurement: KeyType | None = None
    ) -> float:
        """Compute Krippendorff's alpha across all coders for the given variable."""
        if level_of_measurement is None:
            level_of_measurement = KeyType.NOMINAL

        wide_df = self.wide_df[key]
        reliability_data = (
            wide_df.drop(columns=self.item_id).astype("float64").T.to_numpy()
        )

        return krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement.value,
        )

    def coder_summary(self, key: str) -> DataFrame:
        """Count items coded and missing per coder for the given variable."""
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]
        n_items = len(wide_df)

        rows = []
        for coder in coders:
            coded = int(wide_df[coder].notna().sum())
            rows.append(
                {
                    "coder": coder,
                    "items_coded": coded,
                    "missing": n_items - coded,
                    "% missing": round((n_items - coded) / n_items * 100, 2),
                }
            )

        return DataFrame(rows)

    def _resolve_n_coders(self, key: str) -> int:
        """Return self.n_coders if set, otherwise infer it from the coder columns."""
        if self.n_coders is not None:
            return self.n_coders
        wide_df = self.wide_df[key]
        return len([col for col in wide_df.columns if col != self.item_id])

    @staticmethod
    def _disagreement_label(
        distance: float, min_distance: float, max_distance: float
    ) -> str:
        if distance == min_distance:
            return "High"
        if distance == max_distance:
            return "Low"
        return "Medium"

    def assign_disagreement_level(
        self, key: str, task_type: KeyType = KeyType.NOMINAL
    ) -> DataFrame:
        """Rank fully-coded items by coder disagreement (lowest disagreement_score = most contentious = High).

        disagreement_score is modal share: the fraction of coders who picked
        the item's most-voted category (low share = votes are spread across
        categories = contentious). A 2-category (binary) variable is just a
        nominal variable with 2 categories, so the same formula applies.
        """
        if task_type != KeyType.NOMINAL:
            raise ValueError(
                f"assign_disagreement_level does not support task_type {task_type!r}"
            )

        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]
        n_coders = self._resolve_n_coders(key)

        votes = wide_df[coders]
        disagreement_df = wide_df[[self.item_id]].copy()
        disagreement_df["n_coded"] = votes.notna().sum(axis=1)
        disagreement_df = disagreement_df[disagreement_df["n_coded"] == n_coders].copy()
        fully_coded_votes = votes.loc[disagreement_df.index]

        disagreement_df["modal_share"] = (
            fully_coded_votes.apply(lambda row: row.value_counts().max(), axis=1)
            / n_coders
        )
        disagreement_df["disagreement_score"] = disagreement_df["modal_share"]

        max_score = disagreement_df["disagreement_score"].max()
        min_score = disagreement_df["disagreement_score"].min()

        disagreement_df["disagreement_level"] = disagreement_df[
            "disagreement_score"
        ].apply(lambda s: self._disagreement_label(s, min_score, max_score))
        return disagreement_df.sort_values("disagreement_score")

    def _resolve_abstracts_df(
        self,
        abstracts_df: DataFrame | None,
        item_text_column: str,
        item_title_column: str | None,
    ) -> DataFrame | None:
        """Return an item_id -> title/text lookup table, or None if unavailable."""
        if abstracts_df is not None:
            return abstracts_df
        if self.df is not None and item_text_column in self.df.columns:
            columns = [self.item_id]
            if item_title_column and item_title_column in self.df.columns:
                columns.append(item_title_column)
            columns.append(item_text_column)
            return self.df[columns].drop_duplicates(subset=self.item_id)
        return None

    @staticmethod
    def _dataframe_to_html_table(df: DataFrame) -> str:
        return df.to_html(
            index=False,
            classes="data-table",
            border=0,
            na_rep="—",
            float_format=lambda v: f"{v:.3f}",
        )

    def _build_coding_export(
        self, key: str, disagreement_df: DataFrame, abstracts_df: DataFrame | None
    ) -> DataFrame:
        """All items for a key, with disagreement_level, abstract text, each coder's vote,

        and a "resolved" column: the shared classification when all coders
        agree, blank otherwise (left for discussion to fill in).
        """
        assert self.item_id is not None
        item_id_col = self.item_id

        to_resolve = disagreement_df[[item_id_col, "disagreement_level"]].copy()
        to_resolve.insert(0, "key", key)

        if abstracts_df is not None:
            to_resolve = to_resolve.merge(abstracts_df, on=item_id_col, how="left")

        votes = self.wide_df[key]
        coder_cols = [c for c in votes.columns if c != item_id_col]
        to_resolve = to_resolve.merge(votes, on=item_id_col, how="left")

        all_agree = to_resolve[coder_cols].nunique(axis=1, dropna=False) == 1
        to_resolve["resolved"] = to_resolve[coder_cols].iloc[:, 0].where(all_agree)

        return to_resolve

    def _render_key_section(
        self, key: str, alpha: float, summary_df: DataFrame, kappa_df: DataFrame
    ) -> str:
        return f"""
        <section class="key-section">
            <h2>{escape(key)}</h2>
            <p class="alpha-stat">Krippendorff's alpha: <strong>{alpha:.3f}</strong></p>
            <h3>Coder completeness</h3>
            {self._dataframe_to_html_table(summary_df)}
            <h3>Pairwise Cohen's kappa</h3>
            {self._dataframe_to_html_table(kappa_df)}
        </section>
        """

    @staticmethod
    def _render_report_html(generated_at: str, key_sections: list[str]) -> str:
        template_path = Path(__file__).resolve().parent / "report_template.html"
        template = template_path.read_text()
        return template.replace("$generated_at", escape(generated_at)).replace(
            "$key_sections", "".join(key_sections)
        )

    def generate_report(
        self,
        output_path: str,
        key_types: dict[str, KeyType],
        priority_csv_dir: str | None = None,
        review_csv_dir: str | None = None,
        abstracts_df: DataFrame | None = None,
        item_text_column: str = "abstract",
        item_title_column: str | None = "title",
    ) -> str:
        """Write an HTML IRR report (alpha, coder completeness, kappa) for all keys to output_path.

        For each "nominal" key (2 categories = binary, or more), also writes:
        - <priority_csv_dir>/<key>_coding.csv: every fully-coded item, with
          disagreement_level, each coder's vote, and `resolved` pre-filled
          where all coders agree. The full audit record -- lives alongside
          the HTML report since both come out of this same run.
        - <review_csv_dir>/<key>_to_review.csv: just the disagreement rows
          from the coding file (where `resolved` is blank) -- the lean subset
          to actually send out for discussion. Defaults to priority_csv_dir
          if review_csv_dir isn't given.
        """
        resolved_abstracts = self._resolve_abstracts_df(
            abstracts_df, item_text_column, item_title_column
        )

        priority_keys = [k for k in self.keys if key_types[k] == KeyType.NOMINAL]
        if priority_keys and priority_csv_dir is None:
            raise ValueError(
                "priority_csv_dir is required when key_types includes a 'nominal' key"
            )
        priority_dir = Path(priority_csv_dir) if priority_csv_dir else None
        review_dir = Path(review_csv_dir) if review_csv_dir else priority_dir

        key_sections = []

        for key in self.keys:
            self._ensure_wide(key)
            task_type = key_types[key]

            alpha = self.krippendorff_alpha(key, task_type)
            summary = self.coder_summary(key)
            kappa = self.pairwise_cohen_kappa(key)
            key_sections.append(self._render_key_section(key, alpha, summary, kappa))

            if task_type == KeyType.NOMINAL:
                assert priority_dir is not None
                assert review_dir is not None
                disagreement = self.assign_disagreement_level(key, task_type=task_type)
                export = self._build_coding_export(
                    key, disagreement, resolved_abstracts
                )
                export.to_csv(priority_dir / f"{key}_coding.csv", index=False)

                to_resolve = export[export["resolved"].isna()]
                to_resolve.to_csv(review_dir / f"{key}_to_review.csv", index=False)

        html_doc = self._render_report_html(
            datetime.now().strftime("%Y-%m-%d %H:%M"), key_sections
        )

        with open(output_path, "w") as f:
            f.write(html_doc)

        return output_path
