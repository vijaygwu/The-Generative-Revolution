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

`src/` mirrors the printed chapter snippets for inspection. The commands below
run through the installable `the_generative_revolution/` package, which is the
runtime surface behind `tgr-demo` and `python -m the_generative_revolution`.

2. Run the runtime smoke test:

```bash
python tests/validate_runtime_stack.py
```

3. Run one practitioner workflow from the companion repo root:

```bash
tgr-demo product-imaging
tgr-demo anomaly-screening
tgr-demo creative-assistant

# equivalent package entry points
python -m the_generative_revolution product-imaging --seed 0
python -m the_generative_revolution anomaly-screening --seed 0
python -m the_generative_revolution creative-assistant --seed 0
```

If you want to inspect the raw wrappers instead, the same demos also run
directly from an uninstalled checkout:

```bash
python examples/product_imaging_diffusion.py
python examples/anomaly_screening_flow.py
python examples/multimodal_creative_assistant.py
```

4. Compare your run with the saved expected outputs in `artifacts/expected/`.
   The companion also includes matching notebooks in `notebooks/` if you want a
   more exploratory workflow. Those notebooks are intentionally committed with
   cleared outputs.

5. Open `operator_assets/` if you want the printable decision and review tools
   that match the book's practitioner layer. Start with:
   - `operator_assets/gcledo_scorecard.md`
   - `operator_assets/benchmark_brief_retail_imaging.md`
   - `operator_assets/design_clinics/retail_product_imaging.md`
   - `operator_assets/ship_no_ship_checklist.md`

   Scope note: the shipped operator assets cover the reusable Book 3
   field-manual layer, including evaluation packs and design clinics for
   imaging, anomaly screening, and multimodal workflows. Book Chapter 7's voice
   evaluation bundle and design clinic remain book-only in this release.

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

1. Book Chapter 1 for the generative-model family map and decision criteria that
   the rest of the companion code assumes
2. Book Chapter 2 + `src/vae.py` and `tests/test_vae.py`
3. Book Chapter 3 + `src/gan.py` and `tests/test_gan.py`
4. Book Chapter 4 + `examples/anomaly_screening_flow.py` and
   `tests/test_flows.py`
5. Book Chapter 5 + `examples/product_imaging_diffusion.py` and
   `tests/test_diffusion.py`
6. Book Chapter 6 + `examples/multimodal_creative_assistant.py`
7. Book Chapter 7 for the voice decision framework; it remains book-only in this
   companion release, so there is no mirrored voice module, eval pack, or design
   clinic yet
8. Book Chapter 8 + `src/metrics.py` and `operator_assets/README.md` for the
   evaluation bundle, field-manual assets, and release-gate checklists

## Demo-First Shortcut

If you already know the model families and want the fastest hands-on path, use
this shorter sequence instead:

1. Book Chapter 5 + `examples/product_imaging_diffusion.py`
2. Book Chapter 4 + `examples/anomaly_screening_flow.py`
3. Book Chapter 6 + `examples/multimodal_creative_assistant.py`

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

- Open the corresponding `src/*.py` files to compare against the printed
  chapter snippets, and `the_generative_revolution/*.py` for the installable
  package modules behind each demo.
- Open `operator_assets/` if you want the field-manual templates rather than
  more implementation detail. The `design_clinics/` subfolder is the fastest
  path if you want a concrete product memo rather than a blank worksheet.
