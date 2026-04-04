# 9.7.14.5.17. Warp-level Matrix Transpose Instruction: movmatrix

Documents the `movmatrix` instruction for transposing an m8n8 matrix in registers across the warp. Supports b16 type with .trans qualifier. Each thread provides one source element and receives the transposed result. Requires sm_75+, PTX ISA version 7.5.
