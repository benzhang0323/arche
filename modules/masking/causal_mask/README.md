# Causal Mask

## Purpose

`causal_mask` captures the masking pattern that prevents a query position from
attending to future key positions.

This pattern appears throughout autoregressive attention kernels and sequence
models where computation must respect causal ordering.

## Context

In causal sequence models, each query position may only attend to keys at the
same position or earlier positions.

As a result, kernels often apply a causal masking step before softmax so that
future positions do not contribute to the final attention result.

Examples include:

- autoregressive self-attention
- decode attention
- masked training-time attention
- triangular attention score masking

## Arche role

This module captures the structural masking pattern that enforces causal order
over a 2D query-key score layout.

It focuses on the masking operation itself rather than on the surrounding
attention computation.