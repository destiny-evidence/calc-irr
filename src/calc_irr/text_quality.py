"""Per-item text checks (missing title/abstract, non-English text) -- no calculator state needed."""

import os
import re
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

LANGUAGE_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)

# Compared after lowercasing and stripping surrounding brackets/periods, so
# "[No abstract available.]" matches "no abstract available".
DEFAULT_PLACEHOLDERS = [
    "",
    "-",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "no abstract",
    "no abstract available",
    "abstract not available",
    "not available",
]

# Abstracts are checked sentence by sentence, so a German paragraph inside an
# English abstract is still caught. Titles are one short line, checked whole,
# with a stricter confidence cutoff since short text is easier to misdetect.
# Text below the word minimum is too short to detect reliably and is skipped.
ABSTRACT_MIN_WORDS = 5
ABSTRACT_MIN_CONFIDENCE = 0.6
TITLE_MIN_WORDS = 3
TITLE_MIN_CONFIDENCE = 0.8

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def default_model_path() -> Path:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_dir / "calc-irr" / "lid.176.ftz"


def load_language_model(model_path: str | Path | None = None):
    """Load fastText's language-ID model, downloading it (~1MB) on first use."""
    import fasttext

    path = Path(model_path) if model_path else default_model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")
        urllib.request.urlretrieve(LANGUAGE_MODEL_URL, tmp_path)
        tmp_path.rename(path)
    return fasttext.load_model(str(path))


def is_missing(
    text: Series,
    placeholders: list[str] | None = None,
    min_length: int = 1,
) -> Series:
    """True where text is blank, a placeholder like "N/A", or shorter than min_length characters."""
    known = {p.lower() for p in (placeholders or DEFAULT_PLACEHOLDERS)}
    normalized = (
        text.fillna("")
        .astype(str)
        .str.strip()
        .str.strip("[]().")
        .str.strip()
        .str.lower()
    )
    return (normalized.isin(known) | (normalized.str.len() < min_length)).astype(bool)


def _non_english(
    model, segments: list[list[str]], min_words: int, min_confidence: float
) -> list[bool]:
    """True per item if any of its segments is confidently detected as a language other than English."""
    flat = [
        (i, s)
        for i, item_segments in enumerate(segments)
        for s in item_segments
        if len(s.split()) >= min_words
    ]
    flags = [False] * len(segments)
    if not flat:
        return flags
    labels, probs = model.predict([s for _, s in flat], k=1)
    for (i, _), label, prob in zip(flat, labels, probs):
        if label[0] != "__label__en" and prob[0] >= min_confidence:
            flags[i] = True
    return flags


def has_non_english(text: Series, model, is_title: bool = False) -> Series:
    """True where the text contains non-English: per sentence for abstracts, whole text for titles."""
    clean = text.fillna("").astype(str)
    if is_title:
        segments = [[t.replace("\n", " ").strip()] for t in clean]
        flags = _non_english(model, segments, TITLE_MIN_WORDS, TITLE_MIN_CONFIDENCE)
    else:
        segments = [
            [s.strip() for s in _SENTENCE_SPLIT.split(t) if s.strip()] for t in clean
        ]
        flags = _non_english(
            model, segments, ABSTRACT_MIN_WORDS, ABSTRACT_MIN_CONFIDENCE
        )
    return Series(flags, index=text.index, dtype=bool)


def text_quality_flags(
    abstracts_df: DataFrame,
    item_id_col: str,
    title_col: str | None,
    text_col: str,
    model_path: str | Path | None = None,
    placeholders: list[str] | None = None,
) -> DataFrame:
    """One row per item: item_id plus True/False title_missing, abstract_missing,
    title_non_english, abstract_non_english.

    Title columns are left out if there is no title column. If the language
    model can't be loaded (e.g. offline on first run), the two non_english
    columns are left out with a warning rather than failing the report. A
    missing title/abstract counts as False for non_english.
    """
    has_title = bool(title_col) and title_col in abstracts_df.columns
    flags = abstracts_df[[item_id_col]].copy()

    if has_title:
        flags["title_missing"] = is_missing(abstracts_df[title_col], placeholders)
    flags["abstract_missing"] = is_missing(
        abstracts_df[text_col], placeholders, min_length=20
    )

    try:
        model = load_language_model(model_path)
    except Exception as e:
        warnings.warn(f"Skipping non-English check, language model unavailable: {e}")
        return flags

    if has_title:
        flags["title_non_english"] = has_non_english(
            abstracts_df[title_col], model, is_title=True
        )
    flags["abstract_non_english"] = has_non_english(abstracts_df[text_col], model)
    return flags


def text_quality_counts(flags: DataFrame) -> dict[str, int]:
    """Number of flagged items per check column."""
    return {
        col: int(pd.to_numeric(flags[col]).sum())
        for col in flags.columns
        if col.endswith(("_missing", "_non_english"))
    }
