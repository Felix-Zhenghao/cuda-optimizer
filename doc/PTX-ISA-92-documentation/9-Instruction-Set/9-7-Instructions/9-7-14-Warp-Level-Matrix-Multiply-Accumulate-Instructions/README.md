# 9.7.14. Warp-Level Matrix Multiply-Accumulate Instructions

- **9-7-14-1-Matrix-Shape** — MxNxK shape table for all wmma/mma types (dense, sparse, block-scaled) across PTX ISA versions.
- **9-7-14-2-Matrix-Data-types** — Supported multiplicand and accumulator data types for all MMA variants.
- **9-7-14-3-Block-Scaling-formmasync** — Block scaling mechanism for mxf8f6f4/mxf4/mxf4nvf4 mma variants with scale matrix layout and selector details.
- **9-7-14-4-Matrix-multiply-accumulate-operation-usingwmmainstructions** — wmma instruction family: fragment layouts, storage, wmma.load, wmma.store, and wmma.mma.
- **9-7-14-5-Matrix-multiply-accumulate-operation-usingmmainstruction** — mma instruction family: per-shape fragment layouts (17 sub-sections), mma.sync, ldmatrix, stmatrix, movmatrix.
- **9-7-14-6-Matrix-multiply-accumulate-operation-usingmmaspinstruction-with-sparse-matrix-A** — Sparse mma.sp family: 2:4 sparsity storage, per-type sparse fragment layouts, and mma.sp instruction reference.
