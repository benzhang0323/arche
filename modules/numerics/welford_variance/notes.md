# Welford Variance Notes

## Systems Context

Welford-style running statistics appear in GPU kernels that compute means and
variances over rows, blocks, or streams of values.

This pattern is especially important in normalization-related kernels, where
variance must be computed efficiently while preserving numerical stability.

## Why This Pattern Is Hard

A naive variance computation often uses formulas based on separate passes over
the data or on subtracting large nearby quantities.

In lower precision or long reductions, this can become numerically unstable.

Welford's method restructures the update into a running form that tracks:

- count
- mean
- second-moment accumulator

This makes it much more robust for practical GPU kernels.

## Performance Considerations

### Running Updates

Welford variance supports incremental processing of values or tiles without
requiring full materialization of intermediate statistics.

### Numerical Stability

The update avoids the instability of naive variance formulas based on
difference-of-squares style computation.

### Composition

In larger kernels, Welford-style updates often appear inside normalization,
layer statistics, and fused reductions. Arche isolates it here as a standalone
numerics pattern.

### Boundary Handling

Tile-based implementations must safely handle partial tiles near row endings.

## Correctness Requirements

Implementations must guarantee:

- correct row-wise mean computation
- correct row-wise variance computation
- numerically stable running updates
- safe handling of row boundaries