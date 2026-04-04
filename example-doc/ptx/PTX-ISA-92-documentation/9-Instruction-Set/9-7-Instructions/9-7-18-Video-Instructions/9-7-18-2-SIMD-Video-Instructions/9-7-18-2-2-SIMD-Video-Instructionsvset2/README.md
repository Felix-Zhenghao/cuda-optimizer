# 9.7.18.2.2. SIMD Video Instructions: vset2

Documents the `vset2` instruction for dual half-word SIMD comparison. Performs two-way parallel comparison (eq, ne, lt, le, gt, ge) on half-word elements selected from source operands. Results are unsigned. Supports secondary SIMD merge or accumulate operations with configurable half-word mask. Requires `sm_30`+, PTX ISA 3.0.
