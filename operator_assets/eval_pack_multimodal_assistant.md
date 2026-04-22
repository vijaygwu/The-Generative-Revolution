# Minimum Viable Evaluation Pack: Multimodal Assistant

## Stage Metrics

- Retrieval relevance
- Generation quality
- End-task completion
- Refusal / escalation rate
- Stagewise latency

## Human Review

- End-to-end session review
- Grounding
- Usefulness
- Policy compliance

## Policy and Robustness

- Red-team prompts that target stage handoffs
- Fallback behavior under uncertainty
- Provenance logging for each stage

## Ship Gate

- Each stage passes separately
- Provenance is logged
- A fallback path exists for uncertainty or policy hits

## Common Trap

- One blended score that hides whether retrieval, generation, or review is failing
