# Design Clinic: Retail Product Imaging

## Brief

A merchandising team needs prompt-conditioned product imagery for internal
review, light editing for seasonal refreshes, and a path to brand-safe
deployment. Latency matters, but seconds are acceptable if control and image
usefulness stay high.

## Best First Family

Start with `latent diffusion`.

Why:
- the workflow is generation-first
- prompt fidelity matters
- editing and structural control are likely to grow in importance
- the product will probably need policy and review layers later

## Tempting Wrong Turn

- `VAE`
  Good for embeddings and reconstruction, but usually the wrong lead family
  once the job becomes generation-first and control-heavy.
- `GAN`
  Can produce a sharp early demo, but control and roadmap flexibility usually
  become the real bottlenecks.

## Minimum Evaluation Pack

Use `eval_pack_product_imaging.md`:
- alignment proxy
- realism proxy
- diversity across seeds
- p95 latency on a fixed prompt suite
- human review for identity preservation and usefulness
- adversarial prompt set for policy edges

## Ship / No-Ship Call

Ship only if:
- the fixed review panel is stable
- prompt fidelity does not require brittle guidance extremes
- the serving profile fits the declared latency budget
- edits preserve product identity

Do not ship if:
- the checkpoint only looks good on handpicked prompts
- edit quality breaks under realistic workflows
- safety or policy handling exists only as a promise

## Switch Trigger

Move from a single-family diffusion baseline to a composed system when:
- retrieval becomes necessary
- brand-policy review becomes mandatory
- structured edit controls are now part of the product contract
- human review and escalation are not optional cleanup steps anymore
