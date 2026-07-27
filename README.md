# calc-irr

Compute inter-rater reliability (Krippendorff's alpha, pairwise Cohen's kappa)
from coder annotations, and generate an HTML report for every key plus, for
nominal keys, per-key CSVs of every item labeled by disagreement priority
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
calc-irr run <config.yaml>  # generate the report and priority CSVs
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

# Directory to write <key>_priority.csv files for nominal keys.
# Required only if any key below is "nominal".
priority_dir: output

# Column names in the input CSV.
coder_id: user_id
item_id: item_id

# Columns used to pull abstract text/title into the priority CSV (optional).
item_text_column: abstract
item_title_column: title

# Restrict to these coder ids only (optional; omit to use everyone in the CSV).
# coders:
#   - alice
#   - bob

# Each key to report on, mapped to its type: nominal, ordinal, interval, or ratio.
keys:
  example_key: nominal
```

## Output

- **HTML report** (`output`): per key, Krippendorff's alpha, coder
  completeness, and pairwise Cohen's kappa.
- **Priority CSV** (`<priority_dir>/<key>_priority.csv`, one per nominal key):
  every fully-coded item with its `priority` (`High`/`Medium`/`Low`, by
  degree of coder disagreement), abstract text, each coder's vote, and a
  `resolved` column — pre-filled with the shared classification when all
  coders agree, left blank when they don't so it can be filled in after
  discussion. This same file can then double as training/eval data once
  every `resolved` value is filled in.

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
