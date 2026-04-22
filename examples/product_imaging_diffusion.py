"""Compatibility wrapper for the package-scoped product-imaging demo."""

from the_generative_revolution.examples.product_imaging_diffusion import main, run_demo

__all__ = ["main", "run_demo"]


if __name__ == "__main__":
    main()
