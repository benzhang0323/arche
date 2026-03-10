# Batched Sparse Row Gather

## Purpose

Batched sparse row gather retrieves rows from a source tensor using a batch of
index lists.

The operation is conceptually simple but appears frequently inside GPU kernels
that must retrieve irregular subsets of rows from large tensors.

## Context

Many GPU workloads operate on tensors where rows correspond to tokens, items,
or feature vectors.

In sparse workloads, kernels often need to gather a subset of these rows based
on dynamically generated indices.

Examples include:

- sparse attention
- mixture-of-experts routing
- top-k retrieval
- block-sparse kernels

## Arche role

This module captures the structural memory access pattern for batched sparse
row retrieval independent of the higher-level algorithm that generates the
indices.

It focuses solely on the indexed row retrieval step.