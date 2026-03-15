# Arche

A library of reusable structural modules for high-performance GPU kernels.

Arche is a research-oriented library for organizing and implementing recurring
structural patterns that appear in modern GPU kernels.

Rather than centering full fused operators, Arche focuses on smaller structural
units such as irregular memory access, masking and predication, reductions,
and numerically sensitive accumulation.

These units appear repeatedly inside high-performance kernels and form the
building blocks of larger fused operators.

## Scope

Arche is built around a simple idea:

> many high-performance GPU kernels are composed of a small set of recurring
> structural modules.

Examples include:

- sparse and paged memory access
- edge and validity masking
- block and streaming reductions
- mixed-precision accumulation

These modules appear across attention, retrieval, sparse workloads,
and other performance-critical GPU code.

## Goals

Arche is intended to provide:

- a practical taxonomy of GPU kernel modules
- per-module specifications and implementation notes
- backend implementations, starting with Triton
- reference implementations for correctness checking
- lightweight infrastructure for evaluation and benchmarking

## Non-Goals

Arche is not intended to be:

- a fused operator library
- a collection of unrelated kernel snippets
- a backend-specific taxonomy tied only to Triton
- a claim of complete coverage over all GPU kernels

## Repository Layout

- `docs/` — conceptual documents and project design
- `modules/` — module definitions and taxonomy
- `implementations/` — backend implementations
- `references/` — correctness-oriented reference implementations

## Current Modules

### Memory

- `paged_kv_gather`
- `batched_sparse_row_gather`
- `layout_transform_copy`

### Masking

- `causal_mask`
- `ragged_mask`
- `block_edge_mask`

### Reductions

- `block_reduce_sum`
- `online_softmax`

Each module includes a specification, implementation notes, a Triton
implementation, and a PyTorch reference.

## Status

Arche is currently in early development.