# 9.7.8.7. Logic and Shift Instructions: shf

The `shf` (funnel shift) instruction shifts a 64-bit value formed by concatenating two 32-bit operands left or right by a specified amount, extracting the most-significant (left) or least-significant (right) 32 bits. Supports clamp and wrap modes. Useful for multi-word shifts and 32-bit rotate operations. Requires sm_32+. Introduced in PTX ISA 3.1.
