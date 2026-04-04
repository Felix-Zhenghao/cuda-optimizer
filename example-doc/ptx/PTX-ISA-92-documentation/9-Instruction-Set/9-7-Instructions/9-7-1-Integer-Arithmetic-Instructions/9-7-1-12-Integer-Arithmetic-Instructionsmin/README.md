# 9.7.1.12 Integer Arithmetic Instructions: min

Documents the PTX `min` instruction for finding the minimum of two integer values. Supports signed/unsigned types including packed SIMD variants (u16x2, s16x2, u8x4, s8x4). Includes `.relu` modifier to clamp negative results to zero. Requires sm_90+ for SIMD, sm_120f for byte-level SIMD.
