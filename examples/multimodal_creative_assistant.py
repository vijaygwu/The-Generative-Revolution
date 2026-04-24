"""Compatibility wrapper for the package-scoped creative-assistant demo."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from the_generative_revolution.examples.multimodal_creative_assistant import main, run_demo

__all__ = ["main", "run_demo"]


if __name__ == "__main__":
    main()
