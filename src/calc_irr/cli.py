import argparse
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml

from calc_irr import IrrCalculator, KeyType

_DEFAULTS = {
    "coder_id": "user_id",
    "item_id": "item_id",
    "item_text_column": "abstract",
    "item_title_column": "title",
    "coders": None,
    "priority_dir": None,
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
    if config["coders"]:
        df = df[df[config["coder_id"]].isin(config["coders"])]

    key_types: dict[str, KeyType] = config["keys"]

    Path(config["output"]).parent.mkdir(parents=True, exist_ok=True)
    if config["priority_dir"]:
        Path(config["priority_dir"]).mkdir(parents=True, exist_ok=True)

    calculator = IrrCalculator(
        df=df,
        keys=list(key_types),
        coder_id=config["coder_id"],
        item_id=config["item_id"],
    )
    output_path = calculator.generate_report(
        output_path=config["output"],
        key_types=key_types,
        priority_csv_dir=config["priority_dir"],
        item_text_column=config["item_text_column"],
        item_title_column=config["item_title_column"],
    )
    print(f"Wrote report to {output_path}")
    if config["priority_dir"]:
        print(f"Wrote priority CSVs to {config['priority_dir']}")


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
    run_parser.set_defaults(func=_run)

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
