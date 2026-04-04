# 9.7.9.6. Data Movement and Conversion Instructions: shfl.sync

The `shfl.sync` instruction exchanges 32-bit register data between threads within a warp, with explicit synchronization via a membermask. Supports four modes: .up, .down, .bfly (butterfly), and .idx (indexed). All participating threads must execute with the same membermask before proceeding. Replaces the deprecated shfl instruction. Requires sm_30+. Introduced in PTX ISA 6.0.
