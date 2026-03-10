# Layout Transform Copy Specification

## Summary

Copy data from a source tensor into a destination tensor while changing its
logical layout.

## Conceptual setting

Many GPU kernels move data not only to copy it, but also to transform its
layout into a form better suited for downstream computation.

A common example is a 2D transpose:

src: [M, N]
dst: [N, M]

More generally, the pattern represents copying data between equivalent logical
views with different memory traversal or writeout structure.

## Inputs

### src

Source tensor containing the input layout.

A common concrete instance is:

[M, N]

### layout mapping

A rule that maps source coordinates to destination coordinates.

For a transpose-like case:

dst[j, i] = src[i, j]

Implementations may realize other layout transforms so long as they preserve
the intended element mapping.

## Outputs

### dst

Destination tensor storing the transformed layout.

For a transpose-like case:

[N, M]

## Transform semantics

The module is defined by an elementwise mapping between source and destination
coordinates.

A common transpose interpretation is:

dst[j, i] = src[i, j]

Implementations may use tiling, staging, or vectorized loads provided the same
logical mapping is preserved.

## Correctness semantics

A correct implementation must:

- preserve all source values
- write each output element to the correct transformed location
- avoid out-of-bounds accesses at layout boundaries

## Scope

This module defines only the memory movement pattern associated with layout
transformation.

It does not define:

- reductions
- masking beyond boundary checks
- fused arithmetic
- downstream compute stages