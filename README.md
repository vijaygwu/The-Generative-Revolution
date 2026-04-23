# The Generative Revolution

Companion code for **"The Generative Revolution"** (Book 3 of *The AI Engineer's Library*) by Dr. Vijay Raghavan.

License: [MIT](LICENSE)

## Repository Structure

```text
The-Generative-Revolution/
├── QUICKSTART.md           # 30-minute practitioner path through the repo
├── artifacts/expected/     # Canonical demo outputs for quick comparison
├── examples/               # Product-style runnable demos built on the modules
├── notebooks/              # Lightweight exploratory companions for key workflows
├── operator_assets/        # Printable field-manual templates from the book
├── src/                    # Maintained runnable reference implementations
│   ├── vae.py              # Variational autoencoder examples
│   ├── gan.py              # GAN baseline plus WGAN-GP helper components
│   ├── flows.py            # RealNVP-style normalizing flows
│   ├── diffusion.py        # DDPM / DDIM building blocks
│   ├── advanced_generative.py  # Book Chapter 6 quantization and CFG helpers
│   ├── metrics.py          # Evaluation helpers such as FID
│   ├── utils.py            # Small shared utilities
│   └── __init__.py         # Lazy exports for the companion API
├── tests/                  # Direct script validation tests
├── requirements.txt        # Runtime plus notebook/test dependencies for the companion
└── README.md
```

## Chapters Covered

| Book Chapter | Source Chapter | Topic | Companion Module |
|--------------|----------------|-------|------------------|
| 1 | ch20 | Introduction to Generative Models | Orientation only; no mirrored code block |
| 2 | ch21 | Variational Autoencoders | `src/vae.py` |
| 3 | ch22 | Generative Adversarial Networks | `src/gan.py` |
| 4 | ch23 | Normalizing Flows | `src/flows.py` |
| 5 | ch24 | Diffusion Models | `src/diffusion.py` |
| 6 | ch25 | Advanced Generative Models | `src/advanced_generative.py`, `src/metrics.py`, `src/utils.py` |
| 7 | ch26 | Audio and Voice Models | Book-only chapter in this release; no mirrored companion module or operator-asset pack yet |

## Quick Start

```bash
git clone https://github.com/vijaygwu/The-Generative-Revolution.git
cd The-Generative-Revolution

python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

If you want the fastest path into the repo, start with [QUICKSTART.md](QUICKSTART.md).

## Practitioner Workflows

The companion now includes lightweight product-style demos in `examples/`:

- `examples/product_imaging_diffusion.py`
  An untrained retail product-imaging smoke test for Book Chapter 5 with a single
  null-conditioned loss pass and classifier-free-guided DDIM sampling
- `examples/anomaly_screening_flow.py`
  An untrained anomaly-screening smoke test for Book Chapter 4 that keeps
  calibration separate from the evaluated reference and anomaly batches
- `examples/multimodal_creative_assistant.py`
  A retrieval-guidance-evaluation workflow for Book Chapter 6

These scripts use synthetic inputs so they run without external datasets while
still exercising the maintained reference implementations.

The anomaly-screening example is intentionally a smoke test: it demonstrates
score computation and split-aware thresholding, but it does not fit the flow on
real nominal data.

The saved reference outputs for those demos live under `artifacts/expected/`,
and the corresponding exploratory notebooks live under `notebooks/`.
The notebooks are intentionally committed with cleared outputs so the repo
stays diff-friendly.

If you want the operator layer from the book rather than another code path,
open `operator_assets/`. That folder mirrors the most reusable practitioner
tools from Book 3 through Book Chapter 6 (source ch25): the G-CLEDO scorecard,
benchmark brief, failure triage sheet, design clinics, evaluation packs, ship
checklist, economics sheet, and a reusable modelopsy template. Book Chapter 7's
voice evaluation bundle and design clinic currently remain book-only; this
companion release does not yet ship a voice eval pack or voice clinic.

For a package-style workflow, you can also run:

```bash
tgr-demo product-imaging
tgr-demo anomaly-screening
tgr-demo creative-assistant
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy and SciPy for numerical helpers

## Testing

```bash
python tests/validate_runtime_stack.py
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

GitHub Actions runs the same companion-only test sequence on pushes and pull
requests via `.github/workflows/companion-ci.yml`.

For release prep, use [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
