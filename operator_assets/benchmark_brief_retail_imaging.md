# Benchmark Brief: Retail Product Imaging

Use this as the recurring benchmark contract from Book 3 or adapt it for your
own product.

## Goal

Generate catalog-style product images from short prompts and support later
background changes, seasonal variants, and channel-specific edits while
preserving product identity.

## Control

- Text conditioning matters
- Editability matters
- Identity preservation matters

## Latency

- Internal reviewers can tolerate seconds per image
- Minutes per image are too slow for iteration
- The serving policy must declare a real p95 budget

## Data

- Curated historical catalog images
- Product attributes and metadata
- Small reviewed prompt suite
- Small reviewed edit suite

## Minimum Evaluation Bundle

- Alignment proxy
- Realism proxy
- Diversity across seeds
- Product-identity review
- p95 latency
- Human review on a fixed panel

## Ops Constraints

- Freeze sampler and guidance defaults
- Log prompt, seed, sampler, and policy actions
- Keep an escalation path for policy-sensitive or low-confidence outputs
