# Paged KV Gather Specification

## Summary
Gather rows from paged KV storage using an index representation that refers to token rows in block-organized memory.

## Conceptual setting
KV state is stored in fixed-size pages rather than in one fully contiguous sequence-major buffer.

A common abstraction is:

kv_pages: [num_pages, page_size, dim]

Flattening the first two dimensions yields a token-row view:

[num_pages * page_size, dim]

This module defines the retrieval pattern that gathers rows from paged storage based on requested token positions.

## Inputs

### kv_pages
Paged KV storage.

Conceptually:

[num_pages, page_size, dim]

Implementations may use specialized physical layouts so long as they preserve the same row-retrieval semantics.

### token_indices
Indices selecting token rows from paged storage.

Conceptually:

[..., K]

Valid entries identify requested token rows. Depending on the system, these entries may represent:

- flattened token indices over paged storage
- values decoded into page and offset
- indices derived from a block-mapping scheme

Invalid entries may be represented using a sentinel such as `-1`.

## Outputs

### gathered
Gathered KV rows corresponding to the requested token indices.

Conceptually:

[..., K, dim]

Invalid entries must not trigger out-of-bounds reads. Their outputs are masked according to module policy (typically zero-filled).

## Retrieval semantics

Flattened interpretation:

flat_kv = kv_pages.reshape(num_pages * page_size, dim)
row = flat_kv[token_idx]

Equivalent page interpretation:

page = token_idx // page_size
offset = token_idx % page_size
row = kv_pages[page, offset, :]

Implementations may realize retrieval through either representation.

## Validity semantics
A token index is valid if:

- it is not the invalid sentinel
- it is non-negative
- it refers to an in-range row of paged storage

Invalid indices must produce masked output rows without reading invalid memory.

## Scope
This module defines only paged KV row retrieval.

It does not define:

- QK score computation
- softmax
- value accumulation
- fused attention scheduling