# Quick Start

This companion is intentionally small: it gives you runnable reference
implementations for the book plus a few practitioner-style demos you can inspect
in under 30 minutes.

## 30-Minute Path

1. Create an environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

2. Run the runtime smoke test:

```bash
python tests/validate_runtime_stack.py
```

3. Run one practitioner workflow from `examples/`:

```bash
tgr-demo product-imaging
tgr-demo anomaly-screening
tgr-demo creative-assistant
```

You can still run the underlying scripts directly from `examples/` if you want
to inspect the raw implementation.

4. Compare your run with the saved expected outputs in `artifacts/expected/`.
   The companion also includes matching notebooks in `notebooks/` if you want a
   more exploratory workflow. Those notebooks are intentionally committed with
   cleared outputs.

## What Each Example Shows

- `examples/product_imaging_diffusion.py`
  An untrained retail product-imaging smoke test that exercises a single
  null-conditioned diffusion loss pass and classifier-free-guided DDIM sampling.

- `examples/anomaly_screening_flow.py`
  An untrained industrial anomaly-screening smoke test that keeps calibration
  separate from the evaluated reference and anomaly batches while exercising
  flow log-likelihood scores and threshold-based triage.

- `examples/multimodal_creative_assistant.py`
  A synthetic multimodal creative-assistant loop that combines retrieval,
  discrete latents, guidance, and evaluation.

The anomaly-screening example demonstrates score plumbing and thresholding
mechanics only. It intentionally skips model fitting, so treat it as a
split-aware smoke test rather than a trained detector.

## Recommended Reading Order

1. Chapter 24 + `examples/product_imaging_diffusion.py`
2. Chapter 23 + `examples/anomaly_screening_flow.py`
3. Chapter 25 + `examples/multimodal_creative_assistant.py`

## After That

- Run the full validation suite:

```bash
python tests/test_examples.py
python tests/test_expected_artifacts.py
python tests/test_package_api.py
python tests/test_notebook_hygiene.py
python tests/test_advanced_generative.py
python tests/test_metrics.py
python tests/test_vae.py
python tests/test_gan.py
python tests/test_flows.py
python tests/test_diffusion.py
```

- Open the corresponding `src/*.py` files to see the maintained reference
  implementations behind each demo.
