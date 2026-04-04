# 9.7.16.4.3. Zero-Column Mask Descriptor

Describes the zero-column mask descriptor used to zero out specific columns of matrix B during MMA operations regardless of shared memory values. The mask size depends on B's column count. Supports per-column masking at 16-column granularity. Provided as a 32-bit register operand.
