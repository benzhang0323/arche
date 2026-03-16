# Welford Variance Specification

## Summary

Compute row-wise mean and variance using numerically stable running updates.

## Conceptual setting

Many GPU kernels compute statistics incrementally rather than in one fully
materialized pass.

A common abstraction is:

x: [B, N]

The computation is performed row-wise over the last dimension.

## Inputs

### x

Input tensor containing values whose statistics are computed.

Conceptually:

[B, N]

Implementations may operate on equivalent layouts so long as the same row-wise
statistics semantics are preserved.

## Outputs

### mean

Row-wise means.

Conceptually:

[B]

### var

Row-wise variances.

Conceptually:

[B]

Each output row contains the mean and variance of the corresponding input row.

## Update semantics

For each row, the implementation maintains running state while processing the
row incrementally.

A common formulation maintains:

- count
- running mean
- running second-moment accumulator

The final variance is derived from this running state.

## Correctness semantics

A correct implementation must:

- produce the same row-wise mean and variance as a reference implementation
- preserve numerical stability under incremental processing
- avoid out-of-bounds accesses at row boundaries

## Scope

This module defines only the Welford-style running statistics pattern.

It does not define:

- normalization output transformation
- affine scaling or shifting
- fused downstream operators
- multi-stage cross-block reduction