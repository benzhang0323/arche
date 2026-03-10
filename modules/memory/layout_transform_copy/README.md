# Layout Transform Copy

## Purpose

`layout_transform_copy` captures the memory movement pattern where data is read
from one logical layout and written into another.

The operation is conceptually simple, but it appears frequently inside GPU
kernels that must stage, reorder, or reinterpret data before downstream
computation.

## Context

High-performance GPU kernels often depend on intermediate layout transforms to
improve memory access behavior, enable vectorized loads, or match the expected
layout of a subsequent kernel stage.

Examples include:

- transposing row-major and column-major views
- staging data for tiled compute
- reordering tensors for more efficient downstream access
- materializing layout changes between pipeline stages

## Arche role

This module captures the structural pattern of copying data while changing its
logical layout.

It focuses on memory movement and layout reinterpretation rather than on any
particular fused operator or downstream computation.