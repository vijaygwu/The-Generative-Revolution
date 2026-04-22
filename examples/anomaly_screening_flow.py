"""Compatibility wrapper for the package-scoped anomaly-screening demo."""

from the_generative_revolution.examples.anomaly_screening_flow import main, run_demo

__all__ = ["main", "run_demo"]


if __name__ == "__main__":
    main()
