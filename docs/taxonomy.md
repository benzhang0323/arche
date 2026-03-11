# Kernel Taxonomy

Arche currently organizes modules into four categories:

- memory
- masking
- reductions
- numerics

These categories reflect recurring structural patterns in high-performance GPU
kernels.

## Memory

Memory modules capture access, movement, and layout transformation patterns.

Examples include:

- paged row retrieval
- sparse batched gather
- layout-changing copy

## Masking

Masking modules capture validity and boundary constraints that determine which
elements participate in downstream computation.

Examples include:

- causal masking
- ragged validity masking
- edge masking

## Reductions

Reduction modules capture aggregation patterns over blocks, rows, or streaming
state.

Examples include:

- block reduction
- online softmax
- segmented reduction

## Numerics

Numerics modules capture precision-sensitive accumulation and stability
patterns.

Examples include:

- fp32 accumulation templates
- stable mixed-precision softmax
- variance accumulation