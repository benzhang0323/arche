# Online Softmax Specification

## Summary

Compute softmax using a running reduction over the input that maintains
numerically stable normalization state.

## Conceptual setting

Many GPU kernels process softmax inputs incrementally rather than in one full
materialized pass.

A common abstraction is:

scores: [batch, N]

The reduction is performed row-wise over the last dimension.

## Inputs

### scores

Input tensor containing softmax scores.

Conceptually:

[batch, N]

Implementations may operate on equivalent layouts so long as the same row-wise
softmax semantics are preserved.

## Outputs

### probs

Softmax probabilities with the same shape as the input.

Conceptually:

[batch, N]

Each output row is the normalized exponential transform of the corresponding
input row.

## Reduction semantics

For each row, softmax is defined as:

probs[i] = exp(scores[i]) / sum_j exp(scores[j])

A numerically stable online implementation maintains running normalization state
while processing the row incrementally.

A common formulation maintains:

- a running maximum
- a running exponential sum

When a new tile or element is processed, the normalization state is updated so
that the final output matches the stable softmax result.

## Correctness semantics

A correct implementation must:

- produce the same row-wise softmax as a stable reference implementation
- preserve numerical stability under incremental processing
- avoid out-of-bounds accesses at row boundaries

## Scope

This module defines only the online softmax reduction pattern.

It does not define:

- score computation
- masking
- value accumulation
- fused attention scheduling