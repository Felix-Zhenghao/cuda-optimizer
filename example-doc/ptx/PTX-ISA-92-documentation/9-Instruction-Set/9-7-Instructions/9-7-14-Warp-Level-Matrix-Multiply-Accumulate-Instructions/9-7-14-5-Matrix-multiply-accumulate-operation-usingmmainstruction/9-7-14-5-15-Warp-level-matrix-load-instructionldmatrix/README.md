# 9.7.14.5.15. Warp-level Matrix Load Instruction: ldmatrix

Documents the `ldmatrix` instruction for collectively loading matrices from shared memory for `mma` operations. Supports m8n8 and m16n8 shapes, x1/x2/x4 counts, optional transpose, and b16/b8 types. Each thread provides a source address; matrices are loaded cooperatively across the warp.
