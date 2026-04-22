"""CLI entry points for the Book 3 companion demos."""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Callable

DemoFn = Callable[[int], dict[str, object]]

_DEMO_MODULES: dict[str, str] = {
    "product-imaging": "the_generative_revolution.examples.product_imaging_diffusion",
    "anomaly-screening": "the_generative_revolution.examples.anomaly_screening_flow",
    "creative-assistant": "the_generative_revolution.examples.multimodal_creative_assistant",
}


def _load_demo(name: str) -> DemoFn:
    module = importlib.import_module(_DEMO_MODULES[name])
    return module.run_demo


def run_named_demo(name: str, seed: int = 0) -> dict[str, object]:
    """Run a named practitioner demo and return its JSON-serializable result."""
    if name not in _DEMO_MODULES:
        valid = ", ".join(sorted(_DEMO_MODULES))
        raise ValueError(f"unknown demo {name!r}; expected one of: {valid}")
    return _load_demo(name)(seed=seed)


def _emit(result: dict[str, object]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgr-demo",
        description="Run practitioner demos from The Generative Revolution companion.",
    )
    parser.add_argument(
        "demo",
        choices=sorted(_DEMO_MODULES),
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
