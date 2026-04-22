# Design Clinic: Industrial Anomaly Screening

## Brief

A manufacturing team needs a reviewed anomaly-screening workflow, not a pretty
generator. The real constraint is ranking suspicious cases under a fixed
reviewer budget while keeping nuisance variation from flooding the queue.

## Best First Family

Start with a `flow-based` or `feature-space scoring` workflow.

Why:
- the real product surface is a score and threshold
- calibration matters more than sample beauty
- reviewer usefulness is the key outcome

## Tempting Wrong Turn

- `Diffusion` or `GAN`
  These can produce visually compelling outputs, but that does not solve the
  screening job. A generation-first family is usually the wrong default when
  the workflow needs stable ranking and reviewed calibration.

## Minimum Evaluation Pack

Use `eval_pack_anomaly_screening.md`:
- threshold stability
- precision at the real review budget
- miss cost by defect subtype
- drift checks by acquisition slice
- reviewed panel of top alerts, borderline alerts, and known misses

## Ship / No-Ship Call

Ship only if:
- one threshold produces an acceptable reviewer load
- top-ranked alerts correspond to review-worthy events
- nuisance variation does not dominate the queue

Do not ship if:
- the dashboard metric looks good but reviewers still drown in false positives
- calibration changes meaningfully across acquisition slices

## Switch Trigger

Switch from raw likelihood scoring to a cleaner feature-space ranking workflow
when reviewed false positives are dominated by nuisance variation.

Switch away from the family entirely when no threshold makes the review loop
economically tolerable.
