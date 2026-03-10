# Design Principles

Arche is guided by a small set of principles intended to keep the project
focused, practical, and grounded in real GPU kernel design.

## 1. Focus on recurring kernel structure

Each module should capture a computational structure that appears across
multiple high-performance GPU kernels. Modules should not represent
one-off tricks or implementation artifacts.

## 2. Prefer modules over fused operators

Arche focuses on reusable structural components rather than complete
fused operators. A module should isolate a meaningful piece of kernel
behavior that can appear across different workloads.

## 3. Emphasize performance-critical structure

The most valuable modules are often the ones that make GPU kernels
difficult to write or optimize. Examples include irregular memory
access patterns, masking, reductions, and mixed-precision accumulation.

## 4. Keep modules minimal but meaningful

Modules should be small enough to remain understandable, but substantial
enough to represent real implementation concerns in GPU kernels.

## 5. Separate concept from backend

Modules represent conceptual structures. Backend implementations such
as Triton or CUDA are concrete realizations of those modules and should
not define the module itself.

## 6. Support evaluation and referenceability

Modules should be easy to specify, test, benchmark, and compare against
reference implementations. Arche is intended not only as code, but also
as a structured artifact for studying GPU kernel design.

## 7. Stay practical

The taxonomy should remain compact and grounded in real kernels. Arche
does not aim to construct an exhaustive ontology of GPU computation,
but rather a practical library of reusable kernel structures.