# 9.7.1.13 Integer Arithmetic Instructions: max

Documents the PTX `max` instruction for finding the maximum of two integer values. Supports signed/unsigned types including packed SIMD variants (u16x2, s16x2, u8x4, s8x4). Includes `.relu` modifier to clamp negative results to zero. Requires sm_90+ for SIMD, sm_120f for byte-level SIMD.
