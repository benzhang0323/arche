# Online Softmax

## Purpose

`online_softmax` captures the reduction pattern used to compute softmax in a
streaming or tiled manner without materializing a full-row maximum and
normalization sum in separate passes over the data.

This pattern appears frequently inside high-performance attention kernels and
other reductions where numerical stability must be preserved under incremental
processing.

## Context

Softmax is often written as a reduction over a row or score vector.

In high-performance GPU kernels, however, the input is frequently processed in
tiles or streaming chunks rather than as one contiguous vector. In this setting,
the implementation must maintain running normalization state while preserving
numerical stability.

Examples include:

- tiled attention kernels
- streaming score normalization
- blockwise softmax
- fused attention reductions

## Arche role

This module captures the structural reduction pattern underlying online softmax.

It focuses on the stable running reduction itself rather than on the larger
fused operator in which the reduction may be embedded.