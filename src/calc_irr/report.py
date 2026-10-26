"""HTML rendering and coding-export helpers for IrrCalculator.generate_report."""

from html import escape
from pathlib import Path

from pandas import DataFrame


def dataframe_to_html_table(df: DataFrame) -> str:
    return df.to_html(
        index=False,
        classes="data-table",
        border=0,
        na_rep="—",
        float_format=lambda v: f"{v:.3f}",
    )


def build_coding_export(
    key: str,
    disagreement_df: DataFrame,
    votes: DataFrame,
    item_id_col: str,
    abstracts_df: DataFrame | None,
    flags_df: DataFrame | None = None,
) -> DataFrame:
    """All items for a key (`votes` is its wide coder-vote table), with disagreement_level, abstract text, each coder's vote,

    and a "resolved" column: the shared classification when all coders
    agree, blank otherwise (left for discussion to fill in). With
    `flags_df` (from `coder_flags`), also its `any_marked_<name>` True/False
    columns right after the abstract text.
    """
    to_resolve = disagreement_df[[item_id_col, "disagreement_level"]].copy()
    to_resolve.insert(0, "key", key)

    if abstracts_df is not None:
        to_resolve = to_resolve.merge(abstracts_df, on=item_id_col, how="left")

    if flags_df is not None:
        to_resolve = to_resolve.merge(flags_df, on=item_id_col, how="left")
        for col in flags_df.columns.drop(item_id_col):
            to_resolve[col] = to_resolve[col].astype("boolean").fillna(False)
            to_resolve[col] = to_resolve[col].astype(bool)

    coder_cols = [c for c in votes.columns if c != item_id_col]
    to_resolve = to_resolve.merge(votes, on=item_id_col, how="left")

    all_agree = to_resolve[coder_cols].nunique(axis=1, dropna=False) == 1
    to_resolve["resolved"] = to_resolve[coder_cols].iloc[:, 0].where(all_agree)

    return to_resolve


def render_key_section(
    key: str,
    alpha: float,
    disagreements: dict[str, int],
    summary_df: DataFrame,
    kappa_df: DataFrame,
    distribution_df: DataFrame | None = None,
    profile: tuple[DataFrame, DataFrame] | None = None,
    majority_df: DataFrame | None = None,
) -> str:
    n_items = disagreements["n_fully_coded"]
    n_disagree = disagreements["n_disagreements"]
    pct = n_disagree / n_items * 100 if n_items else 0.0

    # (heading, table, caption); None tables are skipped so numbering stays continuous
    tables: list[tuple[str, DataFrame | None, str]] = [
        (
            "Coder completeness",
            summary_df,
            "How many items each coder labeled and how many they left blank.",
        ),
        (
            "Class distribution",
            distribution_df,
            "How often each class was picked, per coder and overall -- shows "
            "class imbalance and coders who lean toward one class.",
        ),
    ]
    if profile is not None:
        profile_summary, profile_breakdown = profile
        tables.append(
            (
                "Agreement vs. disagreement",
                profile_summary,
                "Fully-coded items where all coders picked the same class "
                "(per class), versus items where at least one coder differed.",
            )
        )
        if not profile_breakdown.empty:
            tables.append(
                (
                    "Disagreements by vote pattern",
                    profile_breakdown,
                    "Each disagreement grouped by how every coder voted -- e.g. "
                    "a row where only one coder has a 1 is that coder's "
                    "solo inclusion.",
                )
            )
    if majority_df is not None:
        n_no_majority = n_items - (
            int(majority_df["items compared"].iloc[0]) if len(majority_df) else 0
        )
        caption = (
            "Per coder, how often their vote differs from the class picked by "
            "more than half the coders."
        )
        if n_no_majority:
            caption += (
                f" {n_no_majority} fully-coded item(s) had no majority and "
                f"are left out."
            )
        tables.append(("Coder vs. majority", majority_df, caption))
    tables.append(
        (
            "Pairwise Cohen's kappa",
            kappa_df,
            "Agreement between each pair of coders, corrected for chance "
            "(1 = perfect, 0 = no better than chance), on items both coded.",
        )
    )

    blocks = []
    for i, (heading, df, caption) in enumerate(
        [t for t in tables if t[1] is not None], start=1
    ):
        blocks.append(
            f"<h3>{i}. {escape(heading)}</h3>\n"
            f'            <p class="table-note">{escape(caption)}</p>\n'
            f"            {dataframe_to_html_table(df)}"
        )
    tables_html = "\n            ".join(blocks)

    return f"""
        <section class="key-section">
            <h2>{escape(key)}</h2>
            <p class="alpha-stat">Krippendorff's alpha: <strong>{alpha:.3f}</strong></p>
            <p class="alpha-stat">Disagreements: <strong>{n_disagree}</strong>
                of {n_items} fully-coded items ({pct:.1f}%)</p>
            {tables_html}
        </section>
        """


def render_report_html(
    generated_at: str, key_sections: list[str], source_file: str | None = None
) -> str:
    template_path = Path(__file__).resolve().parent / "report_template.html"
    template = template_path.read_text()
    source_line = (
        f'<p class="source-file">Source: <code>{escape(source_file)}</code></p>'
        if source_file
        else ""
    )
    return (
        template.replace("$generated_at", escape(generated_at))
        .replace("$source_line", source_line)
        .replace("$key_sections", "".join(key_sections))
    )


_TEXT_QUALITY_LABELS = {
    "title_missing": "Title missing",
    "abstract_missing": "Abstract missing",
    "title_non_english": "Title contains non-English",
    "abstract_non_english": "Abstract contains non-English",
}


def render_text_quality_summary(counts: dict[str, int], n_items: int) -> str:
    """Report section with how many items each text check flagged."""
    rows = DataFrame(
        [
            {
                "check": _TEXT_QUALITY_LABELS.get(col, col),
                "items flagged": n,
                "% of items": f"{n / n_items * 100:.1f}%" if n_items else "—",
            }
            for col, n in counts.items()
        ]
    )
    note = (
        f"Automatic checks on the title and abstract of all {n_items} items. "
        "Per-item True/False flags are in the coding and review CSVs."
    )
    if "abstract_non_english" not in counts:
        note += " The non-English check was skipped (language model unavailable)."
    return f"""
        <section class="key-section">
            <h2>Text quality</h2>
            <p class="table-note">{escape(note)}</p>
            {dataframe_to_html_table(rows)}
        </section>
        """


def render_flag_summary(name: str, key: str, n_flagged: int, n_items: int) -> str:
    """Report section with how many unique items any coder marked with a flag."""
    pct = f" ({n_flagged / n_items * 100:.1f}% of {n_items} items)" if n_items else ""
    return f"""
        <section class="key-section">
            <h2>Items marked {escape(name)}</h2>
            <p class="alpha-stat"><strong>{n_flagged}</strong> unique item(s) were
                marked {escape(name)} by at least one coder{escape(pct)}.</p>
            <p class="table-note">From the {escape(key)} question; a blank
                answer counts as not marked. The per-item any_marked_{escape(name)}
                column is in the coding and review CSVs.</p>
        </section>
        """
