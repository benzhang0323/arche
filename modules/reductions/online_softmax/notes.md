# Online Softmax Notes

## Systems Context

Online softmax appears in GPU kernels that normalize rows of scores while
processing them incrementally.

This is especially important in attention kernels, where scores are often
generated and reduced in tiles rather than as one fully materialized matrix.

## Why This Pattern Is Hard

A naive softmax implementation typically requires separate passes to compute a
row maximum, exponentials, and a normalization sum.

In tiled or streaming kernels, however, the reduction must be updated online
while preserving the same final result as a stable full-row softmax.

This requires careful handling of running normalization state.

## Performance Considerations

### Incremental Reduction

Online softmax allows kernels to process tiles or chunks without storing the
entire row at once.

### Numerical Stability

The reduction must track a running maximum and rescale the running sum when a
new maximum is observed.

### Fusion Opportunities

In larger kernels, online softmax is often fused with score generation and
value accumulation. Arche isolates it here as a standalone reduction pattern.

### Boundary Handling

Tile-based implementations must handle partial tiles at the end of each row.

## Correctness Requirements

Implementations must guarantee:

- correct row-wise normalization
- correct running-maximum updates
- correct rescaling of the running sum
- safe handling of row boundaries