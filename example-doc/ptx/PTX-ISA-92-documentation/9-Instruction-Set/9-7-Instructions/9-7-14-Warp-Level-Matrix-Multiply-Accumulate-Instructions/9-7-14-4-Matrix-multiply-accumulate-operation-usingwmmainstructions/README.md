# 9.7.14.4. Matrix Multiply-Accumulate Operation Using wmma Instructions

- **9.7.14.4.1. Matrix Fragments for WMMA** -- Per-thread fragment distribution across warp for all WMMA shapes and types.
- **9.7.14.4.2. Matrix Storage for WMMA** -- Row-major/column-major layouts and alignment for WMMA matrices.
- **9.7.14.4.3. Warp-level Matrix Load: wmma.load** -- Collective warp-level load of matrix fragments from memory.
- **9.7.14.4.4. Warp-level Matrix Store: wmma.store** -- Collective warp-level store of result matrix to memory.
- **9.7.14.4.5. Warp-level MMA: wmma.mma** -- Warp-level multiply-accumulate D = A*B + C instruction.
