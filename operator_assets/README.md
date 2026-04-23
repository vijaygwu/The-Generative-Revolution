# Operator Assets

These assets turn the companion from a runnable code repo into a small working
kit for practitioners.

Current scope note: this folder ships the reusable field-manual layer plus the
Book Chapter 4-6 workflow-specific clinics and evaluation packs (source
ch23-ch25). Book Chapter 7's voice evaluation bundle and design clinic remain
book-only in this release, so there is no voice-specific eval pack or voice
clinic in `operator_assets/` yet.

Use them with the book, not instead of the book:

- `gcledo_scorecard.md`
  Reusable `Goal / Control / Latency / Data / Eval / Ops` model-family sheet
- `benchmark_brief_retail_imaging.md`
  The recurring benchmark contract used across Book 3
- `failure_triage_sheet.md`
  Symptom-to-intervention worksheet
- `review_rubric.md`
  General human-review rubric for product-facing evaluation
- `design_clinics/`
  Selected short product memos from the book's practitioner layer; current repo
  clinics stop before Book Chapter 7's voice workflow
- `eval_pack_product_imaging.md`
  Minimum viable evaluation pack for prompt-conditioned imaging
- `eval_pack_anomaly_screening.md`
  Minimum viable evaluation pack for anomaly workflows
- `eval_pack_multimodal_assistant.md`
  Minimum viable evaluation pack for composed assistants
- `ship_no_ship_checklist.md`
  Release gate checklist
- `economics_tradeoff_sheet.md`
  Qualitative economics worksheet for family comparisons
- `modelopsy_template.md`
  Short postmortem template for documenting family switches

Recommended order:

1. Fill out `gcledo_scorecard.md`
2. Write the workflow in `benchmark_brief_retail_imaging.md` or adapt it
3. Use the relevant `eval_pack_*.md`
4. Adapt the nearest memo in `design_clinics/`
5. Run `ship_no_ship_checklist.md`
6. If the model family fails, document the switch with `modelopsy_template.md`
