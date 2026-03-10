# Paged KV Gather

## Purpose
`paged_kv_gather` captures the indexed row-retrieval pattern used when key or value state is stored in fixed-size pages rather than in one fully contiguous token-major buffer.

## Context
In modern inference kernels, KV state is often organized as paged storage to improve memory management and support dynamic context handling. As a result, requested token rows are not always accessed through simple contiguous traversal. Instead, kernels interpret an index representation over paged storage and retrieve the corresponding rows safely and efficiently.

## What this module represents
`paged_kv_gather` is not a full attention kernel. It isolates the memory-side pattern that:

1. interprets token requests over paged KV storage
2. resolves the corresponding storage row
3. gathers the requested key or value rows
4. applies validity masking for invalid requests

Depending on the system, requested rows may be represented through:

- flattened token indices over paged storage
- explicit page and offset decoding
- indices derived from a block-mapping structure

## Why it matters
This pattern appears in high-performance inference kernels because paged KV storage improves memory management while introducing irregular memory access and nontrivial indexing semantics.

The retrieval operation lies on the critical path between:

- attention query processing
- KV-cache access
- downstream QK or LV computation

## Arche role
In Arche, `paged_kv_gather` is categorized as a **memory module**. It focuses on indexed row retrieval from paged KV storage independent of the surrounding attention math.

## Out of scope
This module does not define:

- query–key score computation
- softmax
- value-weighted accumulation
- output writeback
- full fused attention execution