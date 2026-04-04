# 9.7.8.8-9. Logic and Shift Instructions: shl, shr

Covers two shift instructions. `shl` shifts bits left with zero-fill on the right. `shr` shifts bits right, filling with the sign bit for signed types or zero for unsigned/untyped. Both clamp shift amounts to the register width. The shift amount operand b must always be 32-bit. Supported on all architectures. Introduced in PTX ISA 1.0.
