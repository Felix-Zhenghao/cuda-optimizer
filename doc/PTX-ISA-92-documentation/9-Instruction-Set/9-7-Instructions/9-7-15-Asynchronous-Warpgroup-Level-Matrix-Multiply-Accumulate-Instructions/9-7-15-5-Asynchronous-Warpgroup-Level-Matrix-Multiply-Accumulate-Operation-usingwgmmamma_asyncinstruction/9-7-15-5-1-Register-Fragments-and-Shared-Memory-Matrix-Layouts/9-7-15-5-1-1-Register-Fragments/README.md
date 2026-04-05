Describes per-thread register fragment layouts for accumulator D and multiplicand A in `wgmma.mma_async` operations across different shape/type combinations.

- **9-7-15-5-1-1-1-to-9-7-15-5-1-1-2**: Register fragments for m64nNk16 (f16/bf16) and m64nNk8 (tf32) shapes
- **9-7-15-5-1-1-3**: Register fragments for m64nNk32 shapes (int8, FP8 types)
- **9-7-15-5-1-1-4**: Register fragments for m64nNk256 shapes (binary b1 type)
