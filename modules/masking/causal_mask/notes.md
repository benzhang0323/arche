# Causal Mask Notes

## Systems Context

Causal masking appears in attention kernels that enforce autoregressive
ordering.

Although the masking rule is simple, it often sits directly on the critical
path of attention because it modifies score tensors before normalization.

## Why This Pattern Is Hard

The causal rule is easy to express mathematically, but GPU implementations must
apply it efficiently across large score tensors while respecting tensor
boundaries and minimizing unnecessary control overhead.

This pattern also interacts closely with downstream softmax kernels, since the
choice of masked value affects numerical behavior.

## Performance Considerations

### Predicate Overhead

Masking introduces per-element validity checks, which may reduce throughput if
implemented inefficiently.

### Boundary Handling

Tile-based implementations must handle partial tiles at the edges of the query
and key dimensions.

### Numerical Interaction

Causal masking is often implemented by writing `-inf` or a very negative value.
This must be chosen to behave correctly under downstream softmax.

### Fusion Opportunities

In larger kernels, causal masking is often fused into score computation or
softmax. Arche isolates it here as a standalone structural pattern.

## Correctness Requirements

Implementations must guarantee:

- correct causal validity checks
- preservation of all unmasked entries
- correct masked values for all invalid future positions
- safe handling of boundary tiles