# Module Taxonomy

Arche organizes GPU kernel modules into a small, practical taxonomy.

The taxonomy is intended to group recurring kernel structures in a way
that is useful for implementation, evaluation, and discussion.

## Memory

Modules in this category capture non-trivial memory access structures,
including irregular gathers, paged access patterns, sparse selection,
and layout movement.

These structures are often central to kernel difficulty because memory
behavior strongly influences both performance and implementation
complexity.

## Masking

Masking modules capture predication, validity checks, boundary
handling, and sparsity-aware execution.

These structures frequently appear alongside irregular memory access
and are often necessary to preserve correctness without sacrificing
performance.

## Reductions

Reduction modules capture accumulation across tiles, blocks, or streams
of values.

This includes both simple block reductions and more structured online
or streaming updates that appear in numerically sensitive kernels.

## Numerics

Numerics modules capture precision management and stability-preserving
computation, such as mixed-precision accumulation.

These modules become important when kernel correctness depends not only
on memory access and scheduling, but also on how values are represented
and accumulated.

## Notes

The taxonomy is expected to evolve as Arche grows. Early versions
prioritize a small number of grounded categories rather than broad or
overly fine-grained coverage.