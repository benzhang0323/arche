# Block Edge Mask Specification

## Summary

Apply a validity mask to a block or tile so that only in-bounds elements
participate in computation.

## Conceptual setting

Many GPU kernels operate on fixed-size blocks over a logical tensor region whose
dimensions may not be exact multiples of the block shape.

A common abstraction is:

x: [M, N]

A kernel processes one block at a time using offsets into the logical row and
column dimensions.

## Inputs

### x

Input tensor over which the block mask is applied.

Conceptually:

[M, N]

Implementations may operate on equivalent layouts so long as the same block-edge
validity semantics are preserved.

## Outputs

### out

Masked output tensor with the same logical shape as the input.

Conceptually:

[M, N]

In-bounds positions preserve their original values. Out-of-bounds positions are
replaced with a specified fill value.

## Mask semantics

For a block with row offsets `offs_m` and column offsets `offs_n`, validity is
defined by:

valid = (offs_m < M) & (offs_n < N)

Only valid positions participate in the load, computation, or store.

## Correctness semantics

A correct implementation must:

- preserve values at in-bounds positions
- mask out positions outside the logical tensor extent
- avoid out-of-bounds accesses at block edges

## Scope

This module defines only the block-edge masking pattern.

It does not define:

- the larger computation using the mask
- reduction semantics
- layout transformation logic
- fused downstream operators