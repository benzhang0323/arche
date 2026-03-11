# Ragged Mask Specification

## Summary

Apply a per-batch validity mask over score tensors so that positions beyond the
valid sequence length are masked.

## Conceptual setting

Many kernels operate on tensors whose last dimensions represent query and key
positions, while each batch element may have a different valid key range.

A common abstraction is:

scores: [batch, Q, K]
lengths: [batch]

The module defines the pattern that masks key positions beyond the valid length
for each batch element.

## Inputs

### scores

Input score tensor.

Conceptually:

[batch, Q, K]

Implementations may operate on equivalent layouts so long as the same masking
semantics are preserved.

### lengths

Per-batch valid lengths.

Conceptually:

[batch]

Each entry defines the number of valid key positions for the corresponding
batch element.

### mask policy

A position `(b, q, k)` is valid if:

k < lengths[b]

Invalid positions are masked to a designated value, typically `-inf`.

## Outputs

### masked_scores

Tensor with the same shape as the input scores.

Conceptually:

[batch, Q, K]

All valid positions preserve their original values. All invalid positions are
set to the masked value.

## Mask semantics

For a standard ragged mask:

masked_scores[b, q, k] =
- scores[b, q, k]   if k < lengths[b]
- masked_value      otherwise

A common masked value is negative infinity so that masked positions contribute
zero probability after softmax.

## Correctness semantics

A correct implementation must:

- preserve all valid score entries
- mask all invalid positions beyond each batch element's valid length
- avoid out-of-bounds accesses at tensor boundaries

## Scope

This module defines only the ragged validity masking pattern.

It does not define:

- score computation
- causal ordering
- softmax
- value accumulation
- fused attention scheduling