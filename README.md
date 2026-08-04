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

- **HTML report** (`output`): per key, Krippendorff's alpha, coder
  completeness, and pairwise Cohen's kappa.
- **Coding CSV** (`<priority_dir>/<key>_coding.csv`, one per nominal key):
  every fully-coded item with its `disagreement_level` (`High`/`Medium`/`Low`, by
  degree of coder disagreement), abstract text, each coder's vote, and a
  `resolved` column — pre-filled with the shared classification when all
  coders agree, left blank when they don't. The full audit record.
- **To-review CSV** (`<priority_dir>/<key>_to_review.csv`, one per nominal key):
  just the disagreement rows from the coding CSV (where `resolved` is
  blank) — the lean file to actually send out for discussion. Once it comes
  back with `resolved` filled in (plus whatever discussion columns, e.g.
  notes/decision/reason), combine it with the coding CSV via
  `IrrCalculator.merge_resolutions()` to get the final resolved data, rather
  than hand-saving a third full copy.

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
