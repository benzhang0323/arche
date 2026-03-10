# Paged KV Gather Notes

## Systems Context
Paged KV retrieval appears inside inference kernels that operate over block-organized KV cache state. It is especially relevant when kernels process a query against context rows stored in paged memory.

The retrieval step is conceptually simple but performance-critical.

## Why This Pattern Is Hard
Unlike dense contiguous loads, paged KV retrieval introduces indirection between requested token positions and physical storage rows.

That indirection may appear as:

- flattened token indices over paged storage
- explicit page and offset decoding
- indices produced from a block-mapping structure

This makes the access pattern less regular than standard contiguous sequence-major traversal.

## Performance Considerations

### Irregular Access
Access locality depends on how requested token rows map onto physical pages. Nearby logical requests may not correspond to nearby addresses.

### Coalescing
Memory performance depends heavily on how neighboring threads access rows within the same page.

### Divergence
Invalid token masking may introduce control divergence in sparse retrieval workloads.

### Key vs Value Layout
Key and value caches may use slightly different layouts to optimize downstream math kernels, though row retrieval semantics remain the same.

## Correctness Requirements
Implementations must guarantee:

- correct row resolution
- correct page/offset decoding when applicable
- safe handling of invalid indices
- no out-of-bounds reads
- deterministic masked output for invalid requests