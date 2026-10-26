# calc-irr

Compute inter-rater reliability (Krippendorff's alpha, pairwise Cohen's kappa)
from coder annotations, and generate an HTML report for every key plus, for
nominal keys, per-key CSVs of every item labeled by disagreement_level
(High/Medium/Low), with agreed items already filled in under `resolved`.

## Install

```bash

# stable install:
uv add git+https://github.com/you/calc-irr

# as a standalone CLI tool:
uv tool install --editable /path/to/calc-irr

# while actively developing calc-irr, point at your local clone:
uv add --editable /path/to/calc-irr
```

## CLI usage

```bash
calc-irr init [path]        # write a starter config (default: config.yaml); errors if it exists
calc-irr init --force       # overwrite an existing config
calc-irr run <config.yaml>  # generate the report and coding/review CSVs
calc-irr run <config.yaml> --report-only  # regenerate just the HTML report; CSVs untouched
calc-irr resolve <config.yaml>  # combine agreed labels with the completed review file
```

The template written by `init` lives at `src/calc_irr/config_template.yaml`
and ships with the package, so `calc-irr init` works from any project once
you've installed calc-irr.

## Config file

```yaml
# Path to the CSV with one row per (item, coder) annotation.
input: annotations.csv

# Where to write the HTML report.
output: output/report.html

# Directory to write <key>_coding.csv files for nominal keys.
# Required only if any key below is "nominal".
priority_dir: output

# Directory to write <key>_to_review.csv files (the disagreement subset of
# each coding file) for nominal keys. Optional -- defaults to priority_dir.
# review_dir: output

# Column names in the input CSV.
coder_id: user_id
item_id: item_id

# Columns used to pull abstract text/title into the coding/review CSVs (optional).
item_text_column: abstract
item_title_column: title

# Restrict to these coder ids only (optional; omit to use everyone in the CSV).
# coders:
#   - alice
#   - bob

# Merge resolved/gold labels in as an extra coder, per key (optional). Each
# path is a CSV with an item id column and a `resolved` column. Useful for
# checking a newly-added coder against labels that were already adjudicated.
# gold:
#   example_key: path/to/example_key_resolved.csv
# gold_coder_id: gold

# Each key to report on, mapped to its type: nominal, ordinal, interval, or ratio.
keys:
  example_key: nominal
```

## Output

- **HTML report** (`output`): the input file it was built from, then per
  key, Krippendorff's alpha, number of
  disagreements (fully-coded items where coders didn't all agree), coder
  completeness, class distribution (votes per class, per coder and overall),
  agreement vs. disagreement (unanimous items per class, then disagreements
  broken down by vote combination, one column per coder), coder vs. majority
  (per coder, number and % of items where their vote differs from the class
  chosen by more than half the coders) -- these three for nominal/ordinal keys
  only -- and pairwise Cohen's kappa with per-pair disagreement counts.
  Sections are numbered within each key, each with a one-line caption saying
  what it shows. After all the keys: a "Text quality" summary (items with a
  missing title/abstract or non-English text, see below), then for each of
  `flags` with `report: true`, a line giving how many unique items at least
  one coder marked (a blank answer counts as not marked).
- **Coding CSV** (`<priority_dir>/<key>_coding.csv`, one per nominal key):
  every fully-coded item with its `disagreement_level` (`High`/`Medium`/`Low`, by
  degree of coder disagreement), abstract text, each coder's vote, and a
  `resolved` column — pre-filled with the shared classification when all
  coders agree, left blank when they don't. With `flags` set, also one
  `any_marked_<name>` column per flag after the abstract text: True if at
  least one coder gave the flag's answer (e.g. `any_marked_difficult` for
  "is_difficult = 1", `any_marked_ineligible` for "is_eligible = 0"). The
  full audit record.
  Also four True/False text-quality columns right after the abstract:
  `title_missing`, `abstract_missing` (blank, a placeholder like "N/A", or
  under 20 characters), `title_non_english` and `abstract_non_english`
  (abstracts are checked sentence by sentence with fastText, so partly
  non-English abstracts count). The language model (~1MB) is downloaded to
  `~/.cache/calc-irr/` on first run; if that fails, the non-English columns
  are skipped with a warning. Set `text_quality: false` to turn these off.
  With `notes_column` set, also a `coder_notes` column before `resolved`
  (each coder's free-text note as "coder: note", joined by " | "); it's for
  discussion only, so resolved files leave it out.
- **To-review CSV** (`<priority_dir>/<key>_to_review.csv`, one per nominal key):
  just the disagreement rows from the coding CSV (where `resolved` is
  blank) — the lean file to actually send out for discussion. Once it comes
  back with `resolved` filled in (plus whatever discussion columns, e.g.
  notes/decision/reason), run `calc-irr resolve` to get the final resolved
  data (see "Resolving disagreements" below), rather than hand-saving a
  third full copy.

## Resolving disagreements

After discussion, save the completed review file as
`<resolved_dir>/<key>_to_review_completed.csv`, with the final value typed
into `resolved` for each row, and run `calc-irr resolve <config.yaml>`. It
writes `<resolved_dir>/<key>_resolved.csv`: every item from
`<key>_coding.csv` with just `key`, `item_id`, title, text, `resolved` and
`decision` (`unanimous` if all coders agreed, `adjudicated` if the label came
from the review file) -- no coder votes or discussion notes. Items the coders agreed on keep their
label; discussed items take the hand-filled value if it is numeric and one of
the categories the coders used. Anything else (blank, text like `Exclude`, a
typo like `5`) leaves `resolved` blank for that row, and the item ids are
printed so they can be followed up.

## Validating a coder against already-resolved labels

Set `coders:` to just the new coder(s) and add a `gold:` mapping (key ->
path to that key's resolved CSV) to the config; `calc-irr run` merges the
resolved labels in as an extra coder (named by `gold_coder_id`, default
`gold`) before computing kappa/alpha and writing the coding/review CSVs, so
the new coder is checked directly against the adjudicated labels instead of
against other raw coders.

Equivalent from the Python API: `IrrCalculator.add_gold(key, resolved_csv)`
merges a resolved/gold label CSV into a key's wide-format data as an extra
coder column.

## Python API

```python
from calc_irr import IrrCalculator, KeyType

calculator = IrrCalculator(df=df, keys=["example_key"], coder_id="user_id", item_id="item_id")
calculator.generate_report(
    output_path="output/report.html",
    key_types={"example_key": KeyType.NOMINAL},
    priority_csv_dir="output",
)
```
