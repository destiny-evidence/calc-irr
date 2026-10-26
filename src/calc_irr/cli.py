import argparse
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml

from calc_irr import IrrCalculator, KeyType, resolve_labels

_DEFAULTS = {
    "coder_id": "user_id",
    "item_id": "item_id",
    "item_text_column": "abstract",
    "item_title_column": "title",
    "coders": None,
    "priority_dir": None,
    "review_dir": None,
    "resolved_dir": None,
    "gold": None,
    "gold_coder_id": "gold",
}

_TEMPLATE_NAME = "config_template.yaml"


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    for field in ("input", "output", "keys"):
        if field not in raw:
            raise ValueError(f"config is missing required field: {field!r}")

    try:
        raw["keys"] = {
            key: KeyType(task_type) for key, task_type in raw["keys"].items()
        }
    except ValueError as e:
        valid_types = [t.value for t in KeyType]
        raise ValueError(f"{e}; must be one of {valid_types}") from e

    return {**_DEFAULTS, **raw}


def _run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    df = pd.read_csv(config["input"])

    # NACSOS exports sometimes carry stray leading/trailing whitespace on the
    # coder id (e.g. "rokshana.samad "), which then fails to match the config's
    # `coders` list and silently drops that coder from the report.
    coder_col = df[config["coder_id"]]
    if pd.api.types.is_string_dtype(coder_col) or coder_col.dtype == object:
        df[config["coder_id"]] = coder_col.str.strip()

    if config["coders"]:
        df = df[df[config["coder_id"]].isin(config["coders"])]

    key_types: dict[str, KeyType] = config["keys"]

    write_csvs = not args.report_only

    Path(config["output"]).parent.mkdir(parents=True, exist_ok=True)
    if write_csvs and config["priority_dir"]:
        Path(config["priority_dir"]).mkdir(parents=True, exist_ok=True)
    if write_csvs and config["review_dir"]:
        Path(config["review_dir"]).mkdir(parents=True, exist_ok=True)

    calculator = IrrCalculator(
        df=df,
        keys=list(key_types),
        coder_id=config["coder_id"],
        item_id=config["item_id"],
    )

    if config["gold"]:
        for key, resolved_path in config["gold"].items():
            calculator.add_gold(
                key, resolved_path, gold_coder_id=config["gold_coder_id"]
            )

    output_path = calculator.generate_report(
        output_path=config["output"],
        key_types=key_types,
        priority_csv_dir=config["priority_dir"],
        review_csv_dir=config["review_dir"],
        item_text_column=config["item_text_column"],
        item_title_column=config["item_title_column"],
        source_file=config["input"],
        write_csvs=write_csvs,
    )
    if config["gold"]:
        print(f"Merged gold labels for keys: {', '.join(config['gold'])}")
    print(f"Wrote report to {output_path}")
    if not write_csvs:
        return
    if config["priority_dir"]:
        print(f"Wrote coding CSVs to {config['priority_dir']}")
    if config["review_dir"] or config["priority_dir"]:
        print(
            f"Wrote to-review CSVs to {config['review_dir'] or config['priority_dir']}"
        )


def _resolve(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if not config["priority_dir"] or not config["resolved_dir"]:
        raise ValueError(
            "resolve needs both 'priority_dir' and 'resolved_dir' in the config"
        )

    priority_dir = Path(config["priority_dir"])
    resolved_dir = Path(config["resolved_dir"])
    item_id = config["item_id"]

    for key, task_type in config["keys"].items():
        if task_type != KeyType.NOMINAL:
            continue

        completed_path = resolved_dir / f"{key}_to_review_completed.csv"
        if not completed_path.exists():
            print(f"{key}: no {completed_path} yet -- skipping")
            continue

        coding_df = pd.read_csv(priority_dir / f"{key}_coding.csv")
        resolved, skipped = resolve_labels(
            coding_df,
            completed_path,
            item_id_col=item_id,
            title_col=config["item_title_column"],
            text_col=config["item_text_column"],
        )

        output_path = resolved_dir / f"{key}_resolved.csv"
        resolved.to_csv(output_path, index=False)

        n_resolved = int(resolved["resolved"].notna().sum())
        n_adjudicated = n_resolved - int(coding_df["resolved"].notna().sum())
        print(
            f"{key}: {n_resolved} of {len(coding_df)} items resolved "
            f"({n_adjudicated} by adjudication) -> {output_path}"
        )
        if not skipped.empty:
            print(f"  left blank ({len(skipped)}, no usable value in 'resolved'):")
            for row in skipped.itertuples(index=False):
                value = "blank" if pd.isna(row.raw_value) else repr(row.raw_value)
                print(f"    {getattr(row, item_id)}  ({value})")


def _init(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} already exists; pass --force to overwrite"
        )

    template_text = resources.files("calc_irr").joinpath(_TEMPLATE_NAME).read_text()
    output_path.write_text(template_text)
    print(f"Wrote template config to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an inter-rater reliability report from a YAML config."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Generate a report from a config file"
    )
    run_parser.add_argument("config", help="Path to a YAML config file")
    run_parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write only the HTML report; leave the coding/review CSVs untouched",
    )
    run_parser.set_defaults(func=_run)

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Combine agreed labels with a completed review file into <key>_resolved.csv",
    )
    resolve_parser.add_argument("config", help="Path to a YAML config file")
    resolve_parser.set_defaults(func=_resolve)

    init_parser = subparsers.add_parser("init", help="Write a starter config file")
    init_parser.add_argument(
        "output",
        nargs="?",
        default="config.yaml",
        help="Path to write the config template to (default: config.yaml)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output path if it already exists",
    )
    init_parser.set_defaults(func=_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
