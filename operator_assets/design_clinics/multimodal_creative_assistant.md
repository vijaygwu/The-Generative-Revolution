# Design Clinic: Multimodal Creative Assistant

## Brief

A product team wants an assistant that retrieves references, proposes creative
variants, explains why they fit the brief, and escalates uncertain cases for
review. This is a workflow with retrieval, generation, policy checks, and
handoff logic.

## Best First Family

Start with a `composed system`.

Why:
- retrieval, generation, and review are different jobs
- provenance matters
- stagewise debugging matters
- a fallback path is part of the product, not a later patch

## Tempting Wrong Turn

- `Single multimodal generator`
  A monolithic system can look impressive in a demo, but once provenance,
  policy handling, and stagewise debugging matter, it becomes hard to operate
  and harder to trust.

## Minimum Evaluation Pack

Use `eval_pack_multimodal_assistant.md`:
- retrieval relevance
- generation quality
- end-task completion
- refusal and policy behavior
- stagewise latency
- end-to-end human review for relevance, grounding, usefulness, and compliance

## Ship / No-Ship Call

Ship only if:
- each stage passes separately
- provenance is logged well enough to reproduce bad runs
- there is a defined fallback for uncertainty or policy-sensitive outputs

Do not ship if:
- the workflow can only be described with one blended quality score
- failures cannot be localized to a stage

## Switch Trigger

Move from a monolithic assistant to explicit retrieval, generation, and review
gates as soon as the team cannot say which stage is wrong.
