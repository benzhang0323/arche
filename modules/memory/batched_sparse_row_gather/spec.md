# Batched Sparse Row Gather Specification

## Summary

Gather rows from a token-major tensor using batched index lists.

## Conceptual setting

Many GPU kernels operate on tensors where rows represent tokens or feature
vectors.

A common abstraction is:

rows: [num_rows, dim]

Batched sparse retrieval requests rows using an index tensor:

indices: [batch, K]

The module defines the pattern that gathers rows according to these index
requests.

## Inputs

### rows

Source tensor containing rows that may be gathered.

Conceptually:

[num_rows, dim]

### indices

Index tensor specifying which rows to retrieve.

Conceptually:

[batch, K]

Each entry refers to a row of the source tensor.

Invalid entries may be represented using a sentinel such as `-1`.

## Outputs

### gathered

Tensor containing gathered rows.

Conceptually:

[batch, K, dim]

Invalid indices must not trigger out-of-bounds reads. Their outputs are masked
according to module policy (typically zero-filled).

## Retrieval semantics

Conceptually:

row = rows[index]

Batched interpretation:

gathered[b, k] = rows[indices[b, k]]

Implementations may perform retrieval using vectorized loads or other backend
specific mechanisms so long as the same row semantics are preserved.

## Validity semantics

An index is valid if:

- it is not the invalid sentinel
- it is non-negative
- it refers to an in-range row

Invalid entries must produce masked output rows without performing invalid
memory accesses.

## Scope

This module defines only batched sparse row retrieval.

It does not define:

- attention score computation
- reduction operations
- masking logic beyond index validity
- fused operator scheduling