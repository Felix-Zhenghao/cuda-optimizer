# 9.7.19.5. Miscellaneous Instructions: setmaxnreg

Documents the `setmaxnreg` instruction for dynamically adjusting per-thread register count at warp level. Uses `.inc` to request or `.dec` to release registers via a per-CTA register pool. Requires `.sync.aligned` qualifiers and uniform execution across all warps in a warpgroup. Operand must be 24-256, multiple of 8. Supports `sm_90a`/`sm_100a`/`sm_120a` and related architectures, PTX ISA 8.0.
