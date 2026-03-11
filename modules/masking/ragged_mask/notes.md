# Ragged Mask Notes

## Systems Context

Ragged masking appears in GPU kernels that process variable-length sequences in
batched tensor layouts.

Although the rule is simple, the pattern is common because many workloads pad
or pack sequence data into uniform tensor shapes while preserving only a
prefix of valid positions for each batch element.

## Why This Pattern Is Hard

The masking rule depends on batch-local sequence metadata rather than purely on
tensor coordinates.

This means the implementation must combine score-tensor traversal with
per-batch validity checks while still handling tensor boundaries efficiently.

Ragged masking also interacts closely with downstream reductions and softmax,
since masked values must be chosen to preserve correct numerical behavior.

## Performance Considerations

### Batch-Dependent Predicates

Validity depends on `lengths[b]`, so neighboring batch elements may follow
different masking patterns.

### Predicate Overhead

Per-element validity checks can reduce throughput if implemented
inefficiently.

### Boundary Handling

Tile-based implementations must handle partial tiles at the edges of the query
and key dimensions.

### Numerical Interaction

Ragged masking is often implemented by writing `-inf` or a very negative
value. This must behave correctly under downstream softmax or reduction.

## Correctness Requirements

Implementations must guarantee:

- correct per-batch validity checks
- preservation of all unmasked entries
- correct masked values for all invalid positions
- safe handling of tensor boundaries