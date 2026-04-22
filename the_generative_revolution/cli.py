"""CLI entry points for the Book 3 companion demos."""

from __future__ import annotations

import argparse
import json
from typing import Callable

from examples.anomaly_screening_flow import run_demo as run_anomaly_screening_demo
from examples.multimodal_creative_assistant import run_demo as run_creative_assistant_demo
from examples.product_imaging_diffusion import run_demo as run_product_imaging_demo

DemoFn = Callable[[int], dict[str, object]]

_DEMOS: dict[str, DemoFn] = {
    "product-imaging": run_product_imaging_demo,
    "anomaly-screening": run_anomaly_screening_demo,
    "creative-assistant": run_creative_assistant_demo,
}


def run_named_demo(name: str, seed: int = 0) -> dict[str, object]:
    """Run a named practitioner demo and return its JSON-serializable result."""
    if name not in _DEMOS:
        valid = ", ".join(sorted(_DEMOS))
        raise ValueError(f"unknown demo {name!r}; expected one of: {valid}")
    return _DEMOS[name](seed=seed)


def _emit(result: dict[str, object]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgr-demo",
        description="Run practitioner demos from The Generative Revolution companion.",
    )
    parser.add_argument(
        "demo",
        choices=sorted(_DEMOS),
        help="Which practitioner demo to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic demo output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _emit(run_named_demo(args.demo, seed=args.seed))


def product_imaging_main() -> None:
    _emit(run_named_demo("product-imaging", seed=0))


def anomaly_screening_main() -> None:
    _emit(run_named_demo("anomaly-screening", seed=0))


def creative_assistant_main() -> None:
    _emit(run_named_demo("creative-assistant", seed=0))
