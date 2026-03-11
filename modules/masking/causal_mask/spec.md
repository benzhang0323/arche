# Causal Mask Specification

## Summary

Apply a causal masking rule over query-key score matrices so that positions may
not attend to future positions.

## Conceptual setting

Many attention kernels operate on score tensors whose last two dimensions
represent query and key positions.

A common abstraction is:

scores: [batch, Q, K]

Causal masking modifies the scores so that entries corresponding to future key
positions are masked.

## Inputs

### scores

Input score tensor.

Conceptually:

[batch, Q, K]

Implementations may also operate on equivalent layouts so long as the same
query-key masking semantics are preserved.

### mask policy

A rule defining which positions are invalid.

For the standard causal case:

position `(q, k)` is valid if `k <= q`

Invalid positions are masked to a designated value, typically `-inf`.

## Outputs

### masked_scores

Tensor with the same shape as the input scores.

Conceptually:

[batch, Q, K]

All valid positions preserve their original values. All invalid positions are
set to the masked value.

## Mask semantics

For a standard causal mask:

masked_scores[b, q, k] =
- scores[b, q, k]   if k <= q
- masked_value      otherwise

A common masked value is negative infinity so that masked positions contribute
zero probability after softmax.

## Correctness semantics

A correct implementation must:

- preserve all valid score entries
- mask all invalid future positions
- avoid out-of-bounds accesses at tensor boundaries

## Scope

This module defines only the causal masking pattern.

It does not define:

- score computation
- softmax
- value accumulation
- fused attention scheduling