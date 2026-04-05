# 9.7.14.4. Matrix Multiply-Accumulate using `wmma` Instructions

- **9-7-14-4-1-Matrix-Fragments-for-WMMA** — Per-thread fragment register layouts for all wmma data types and matrix roles.
- **9-7-14-4-2-Matrix-Storage-for-WMMA** — Row/column-major storage, stride semantics, and alignment requirements.
- **9-7-14-4-3-Warp-level-Matrix-Load-Instructionwmmaload** — `wmma.load` instruction reference for loading A/B/C matrices.
- **9-7-14-4-4-Warp-level-Matrix-Store-Instructionwmmastore** — `wmma.store` instruction reference for writing result matrix D.
- **9-7-14-4-5-Warp-level-Matrix-Multiply-and-Accumulate-Instructionwmmamma** — `wmma.mma` instruction reference for D = A*B+C across all supported types.
