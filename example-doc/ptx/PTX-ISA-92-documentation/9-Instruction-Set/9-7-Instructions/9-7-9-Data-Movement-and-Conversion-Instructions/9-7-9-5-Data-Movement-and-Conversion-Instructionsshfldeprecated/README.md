# 9.7.9.5. Data Movement and Conversion Instructions: shfl (deprecated)

The deprecated `shfl` instruction exchanges 32-bit register data between threads within a warp using four modes: .up, .down, .bfly (butterfly), and .idx (indexed). Computes a source lane index from operands and mode, copying data if in range. Deprecated in PTX ISA 6.0 in favor of shfl.sync; removed for sm_70+ in PTX ISA 6.4. Requires sm_30+.
