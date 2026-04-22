# Release Checklist

Use this list before the first public release or any tagged update of the
Book 3 companion repo.

## 1. Repo Hygiene

- Confirm `.gitignore` is active and no cache/build artifacts are staged.
- Confirm `README.md` and `QUICKSTART.md` still match the current package and CLI.
- Confirm `artifacts/expected/` reflects the current canonical outputs for the demos.

## 2. Validation

Run the companion-only checks:

```bash
pip install -e ".[dev]"
python tests/validate_runtime_stack.py
python tests/test_examples.py
python tests/test_expected_artifacts.py
python tests/test_package_api.py
python tests/test_advanced_generative.py
python tests/test_metrics.py
python tests/test_vae.py
python tests/test_gan.py
python tests/test_flows.py
python tests/test_diffusion.py
```

Run the full Book 3 integration check from the manuscript repo:

```bash
cd ../../
./scripts/validate_book3_generative.sh
```

## 3. Manual Decisions

- Confirm the chosen MIT license still matches your publishing intent for the
  companion repo.
- Confirm the public GitHub repo name, description, and topics.
- Keep notebooks committed with cleared outputs unless you intentionally choose
  a different publishing policy later.

## 4. Release Prep

- The current initial public release target is `0.1.0`. Bump the version in
  `pyproject.toml` only if you intentionally want a different release number.
- Verify GitHub Actions is green on `.github/workflows/companion-ci.yml`.
- Tag the release and draft release notes describing:
  - package install path
  - CLI demos
  - notebooks
  - expected-output artifacts

## 5. Optional Nice-to-Haves

- Add a short changelog if you expect multiple public releases.
- Add badges to the README after the public GitHub repo URL is final.
