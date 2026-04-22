# Design Clinic: Low-Latency Constrained Generator

## Brief

A product team needs a narrow-domain generator inside an interactive UI with a
hard latency budget, visually consistent outputs, and limited tolerance for
expensive serving infrastructure.

## Best First Family

Start with a `narrow GAN baseline` or a `distilled diffusion baseline`.

Why:
- latency is a hard constraint, not a preference
- the domain is constrained enough that one-pass generation may still win
- the right decision depends on measured p95 latency and retrain stability

## Tempting Wrong Turn

- `Full diffusion`
  Strong on quality and control, but often the wrong default when the
  experience requires near-immediate feedback.
- `Permanent GAN lock-in`
  The first benchmark may be fast, but the tuning burden can erase the gain if
  the roadmap expands or retrains become fragile.

## Minimum Evaluation Pack

Adapt `eval_pack_product_imaging.md` with extra emphasis on:
- p95 latency
- consistency across seeds on a fixed panel
- compact human review for perceived quality
- retrain-to-retrain stability

## Ship / No-Ship Call

Ship only if:
- the chosen serving mode stays inside the hard latency budget
- quality remains acceptable under that fast path
- the main failure modes can be reproduced on demand

Do not ship if:
- most requests need a slow fallback
- retrains regularly reset visual quality

## Switch Trigger

Switch away from GANs when stability work dominates roadmap velocity or when
control requirements outgrow the narrow-domain setup.

Switch away from distilled diffusion when the compression needed for latency
destroys consistency, realism, or editability.
