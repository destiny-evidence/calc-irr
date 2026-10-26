from datetime import datetime
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import cohen_kappa_score

from calc_irr.key_type import KeyType
from calc_irr.report import (
    build_coding_export,
    render_key_section,
    render_report_html,
)

__all__ = ["IrrCalculator", "KeyType"]


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
                    "n_disagree": int((pair_df[coder_a] != pair_df[coder_b]).sum()),
                    "cohen_kappa": kappa,
                }
            )

        return DataFrame(rows)

    def disagreement_count(self, key: str) -> dict[str, int]:
        """Count fully-coded items where not every coder gave the same value.

        Uses the same rule as the `resolved` column in `<key>_coding.csv`, so
        for nominal keys `n_disagreements` equals the rows in `<key>_to_review.csv`.
        """
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]
        n_coders = self._resolve_n_coders(key)

        votes = wide_df[coders]
        fully_coded = votes[votes.notna().sum(axis=1) == n_coders]
        n_disagreements = int((fully_coded.nunique(axis=1) > 1).sum())

        return {
            "n_fully_coded": len(fully_coded),
            "n_disagreements": n_disagreements,
        }

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

    def class_distribution(self, key: str) -> DataFrame:
        """Count how often each class was chosen, per coder and across all coders.

        One row per class, one count column per coder, then `all` (total votes
        for the class) and `% of all` (its share of all non-missing votes).
        """
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]

        counts = DataFrame(
            {coder: wide_df[coder].dropna().value_counts() for coder in coders}
        )
        counts = counts.fillna(0).astype(int).sort_index()
        counts["all"] = counts.sum(axis=1)
        total = counts["all"].sum()
        share = counts["all"] / total * 100 if total else counts["all"] * 0.0
        counts["% of all"] = share.map(lambda v: f"{v:.1f}%")

        counts.index = counts.index.map(self._format_class)
        return counts.rename_axis("class").reset_index()

    def agreement_profile(self, key: str) -> tuple[DataFrame, DataFrame]:
        """Break fully-coded items down by how the coders' votes fell.

        Returns `(summary, breakdown)`:
        - summary: one "Unanimous <class>" row per class all coders agreed on,
          then a "Disagreement" row -- these add up to the fully-coded items.
        - breakdown: the disagreements grouped by vote pattern -- one column
          per coder with their vote, then how many items had that exact
          combination (e.g. alice=1, bob=0, carol=0 is an alice-only
          inclusion on a binary key), most common first.
        """
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]
        n_coders = self._resolve_n_coders(key)

        votes = wide_df[coders]
        fully_coded = votes[votes.notna().sum(axis=1) == n_coders]
        n_items = len(fully_coded)
        unanimous = fully_coded.nunique(axis=1) == 1

        def pct(n: int) -> str:
            return f"{n / n_items * 100:.1f}%" if n_items else "0.0%"

        summary_rows = [
            {
                "outcome": f"Unanimous {self._format_class(cls)}",
                "items": int(n),
                "% of fully coded": pct(n),
            }
            for cls, n in fully_coded[unanimous]
            .iloc[:, 0]
            .value_counts()
            .sort_index()
            .items()
        ]
        n_disagree = int((~unanimous).sum())
        summary_rows.append(
            {
                "outcome": "Disagreement",
                "items": n_disagree,
                "% of fully coded": pct(n_disagree),
            }
        )

        breakdown = (
            fully_coded[~unanimous]
            .map(self._format_class)
            .value_counts()
            .reset_index(name="items")
        )
        breakdown["% of disagreements"] = breakdown["items"].map(
            lambda n: f"{n / n_disagree * 100:.1f}%"
        )

        return DataFrame(summary_rows), breakdown

    def coder_vs_majority(self, key: str) -> DataFrame:
        """Per coder, how often their vote differs from the majority vote.

        The majority is the class chosen by more than half the coders. Only
        fully-coded items that have one are compared, so on a 3-class key an
        item where every coder picked something different is left out.
        """
        wide_df = self.wide_df[key]
        coders = [col for col in wide_df.columns if col != self.item_id]
        n_coders = self._resolve_n_coders(key)

        votes = wide_df[coders]
        fully_coded = votes[votes.notna().sum(axis=1) == n_coders]
        top_share = fully_coded.apply(lambda row: row.value_counts().max(), axis=1)
        has_majority = top_share > n_coders / 2
        compared = fully_coded[has_majority]
        majority = compared.mode(axis=1).iloc[:, 0]

        rows = []
        for coder in coders:
            n_compared = len(compared)
            differs = int((compared[coder] != majority).sum())
            rows.append(
                {
                    "coder": coder,
                    "items compared": n_compared,
                    "differs from majority": differs,
                    "% differs": f"{differs / n_compared * 100:.1f}%"
                    if n_compared
                    else "0.0%",
                }
            )

        return DataFrame(rows)

    @staticmethod
    def _format_class(value: object) -> object:
        """Show whole-number class codes as ints (pivoting turns them into floats)."""
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

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

    def generate_report(
        self,
        output_path: str,
        key_types: dict[str, KeyType],
        priority_csv_dir: str | None = None,
        review_csv_dir: str | None = None,
        abstracts_df: DataFrame | None = None,
        item_text_column: str = "abstract",
        item_title_column: str | None = "title",
        source_file: str | None = None,
        write_csvs: bool = True,
    ) -> str:
        """Write an HTML IRR report (alpha, disagreements, coder completeness, kappa) for all keys to output_path.

        For each "nominal" key (2 categories = binary, or more), also writes:
        - <priority_csv_dir>/<key>_coding.csv: every fully-coded item, with
          disagreement_level, each coder's vote, and `resolved` pre-filled
          where all coders agree. The full audit record -- lives alongside
          the HTML report since both come out of this same run.
        - <review_csv_dir>/<key>_to_review.csv: just the disagreement rows
          from the coding file (where `resolved` is blank) -- the lean subset
          to actually send out for discussion. Defaults to priority_csv_dir
          if review_csv_dir isn't given.

        `source_file`, if given, is shown in the report header so the report
        can be traced back to the export it was built from.

        With `write_csvs=False`, only the HTML report is written -- the coding
        and review CSVs are left untouched, so a review file already being
        edited isn't overwritten.
        """
        resolved_abstracts = self._resolve_abstracts_df(
            abstracts_df, item_text_column, item_title_column
        )

        key_sections = []
        priority_keys = [
            k for k in self.keys if write_csvs and key_types[k] == KeyType.NOMINAL
        ]
        if priority_keys and priority_csv_dir is None:
            raise ValueError(
                "priority_csv_dir is required when key_types includes a 'nominal' key"
            )
        priority_dir = Path(priority_csv_dir) if priority_csv_dir else None
        review_dir = Path(review_csv_dir) if review_csv_dir else priority_dir

        for key in self.keys:
            self._ensure_wide(key)
            task_type = key_types[key]

            alpha = self.krippendorff_alpha(key, task_type)
            disagreements = self.disagreement_count(key)
            summary = self.coder_summary(key)
            kappa = self.pairwise_cohen_kappa(key)
            # Interval/ratio keys can take many distinct values, so per-class tables aren't useful there
            is_categorical = task_type in (KeyType.NOMINAL, KeyType.ORDINAL)
            distribution = self.class_distribution(key) if is_categorical else None
            profile = self.agreement_profile(key) if is_categorical else None
            majority = self.coder_vs_majority(key) if is_categorical else None
            key_sections.append(
                render_key_section(
                    key,
                    alpha,
                    disagreements,
                    summary,
                    kappa,
                    distribution,
                    profile,
                    majority,
                )
            )

            if write_csvs and task_type == KeyType.NOMINAL:
                assert priority_dir is not None
                assert review_dir is not None
                disagreement = self.assign_disagreement_level(key, task_type=task_type)
                export = build_coding_export(
                    key,
                    disagreement,
                    self.wide_df[key],
                    self.item_id,
                    resolved_abstracts,
                )
                export.to_csv(priority_dir / f"{key}_coding.csv", index=False)

                to_resolve = export[export["resolved"].isna()]
                to_resolve.to_csv(review_dir / f"{key}_to_review.csv", index=False)

        html_doc = render_report_html(
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            key_sections,
            source_file,
        )

        with open(output_path, "w") as f:
            f.write(html_doc)

        return output_path
