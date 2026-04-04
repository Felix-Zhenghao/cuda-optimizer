# 9.7.18.2.4. SIMD Video Instructions: vset4

Documents the `vset4` instruction for quad byte SIMD comparison. Performs four-way parallel comparison (eq, ne, lt, le, gt, ge) on byte elements selected from source operands. Results are unsigned. Supports secondary SIMD merge or accumulate operations with configurable byte mask. Requires `sm_30`+, PTX ISA 3.0.
