# 9.7.18.2.3. SIMD Video Instructions: vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4

Documents quad byte SIMD video arithmetic instructions. Performs four-way parallel add, subtract, average, absolute difference, min, or max on byte elements selected from source operands. Supports optional saturation, secondary SIMD merge, or accumulate operations with configurable byte mask. Requires `sm_30`+, PTX ISA 3.0.
