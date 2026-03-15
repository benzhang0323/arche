# Block Edge Mask

## Purpose

`block_edge_mask` captures the masking pattern used to handle partial tiles at
the boundaries of a tensor.

This pattern appears frequently in GPU kernels that process data in fixed-size
blocks while operating on tensor dimensions that are not exact multiples of the
block shape.

## Context

Many high-performance GPU kernels are written around regular block or tile
shapes.

However, the logical tensor extent often ends partway through the final block.
In this setting, the kernel must identify which positions are valid and mask out
the rest.

Examples include:

- tiled matrix-style kernels
- boundary-safe loads and stores
- copy kernels over rectangular regions
- reductions over partial blocks
- fused kernels with tail tiles

## Arche role

This module captures the structural masking pattern underlying block-edge
boundary handling.

It focuses on the validity pattern itself rather than on the larger operator in
which the mask is embedded.