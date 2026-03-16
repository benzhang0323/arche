# Welford Variance

## Purpose

`welford_variance` captures the numerically stable running-statistics pattern
used to compute mean and variance over a reduction domain.

This pattern appears frequently in GPU kernels that compute normalization
statistics or other row-wise summary values while processing inputs
incrementally.

## Context

Mean and variance are common reductions in ML kernels.

In high-performance GPU kernels, however, inputs are often processed in tiles or
streaming chunks rather than as one fully materialized reduction domain. In
this setting, the implementation must maintain running statistics while
preserving numerical stability.

Examples include:

- normalization statistics
- row-wise mean/variance computation
- streaming statistics over tiles
- fused kernels with embedded statistical reduction

## Arche role

This module captures the structural numerics pattern underlying Welford-style
running variance computation.

It focuses on the stable update structure itself rather than on the larger
normalization operator in which it may be embedded.