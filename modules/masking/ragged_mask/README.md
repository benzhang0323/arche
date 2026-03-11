# Ragged Mask

## Purpose

`ragged_mask` captures the masking pattern used when different batch elements
have different valid sequence lengths.

This pattern appears in GPU kernels that operate on padded or packed tensors
where only a prefix of each sequence should participate in computation.

## Context

Many sequence workloads batch examples of different lengths into shared tensor
shapes.

As a result, kernels often need to apply a validity mask so that positions
beyond the true length of each sequence are excluded from downstream
computation.

Examples include:

- padded attention score tensors
- variable-length retrieval contexts
- packed sequence kernels
- decode kernels with batch-dependent context lengths

## Arche role

This module captures the structural masking pattern that enforces per-batch
validity over a query-key score layout.

It focuses on the masking operation itself rather than on the surrounding
attention or retrieval computation.