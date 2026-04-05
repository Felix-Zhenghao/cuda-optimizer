# 5.5.7. Swizzling Modes

Describes shared memory swizzling modes for tensor copy: no swizzle, 32B (256B-aligned, pair-swap pattern), 64B (512B-aligned, 4-row XOR pattern), 96B (256B-aligned), and 128B (1024B-aligned, 8x8 XOR pattern). The 128B mode supports four atomicity sub-modes: 16B, 32B, 32B+8B-flip (cluster copy only), and 64B. Swizzle base offset is computed from destination address alignment.
