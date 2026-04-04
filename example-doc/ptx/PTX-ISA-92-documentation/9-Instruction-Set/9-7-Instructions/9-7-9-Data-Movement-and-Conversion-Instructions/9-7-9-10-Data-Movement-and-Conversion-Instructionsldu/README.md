# 9.7.9.10. Data Movement and Conversion Instructions: ldu

The `ldu` instruction loads read-only data into a register from global memory where the address is guaranteed to be uniform across all threads in the warp. Supports scalar and vector loads (.v2, .v4) with various types from .b8 to .b128. The uniform address requirement enables hardware optimizations. Introduced in PTX ISA 2.0.
