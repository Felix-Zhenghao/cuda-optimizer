# 9.7.14.5.16. Warp-level Matrix Store Instruction: stmatrix

Documents the `stmatrix` instruction for collectively storing matrices to shared memory. Supports m8n8 and m16n8 shapes, x1/x2/x4 counts, optional transpose, and b16 type. Each thread provides a destination address; matrices are stored cooperatively across the warp from register fragments.
