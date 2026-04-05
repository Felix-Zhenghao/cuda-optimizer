# 5.4.4. Initializers

Covers PTX variable initializers (constant and global spaces only): C-style syntax with partial array initialization defaulting to zero. Names in initializers represent state-space addresses; `generic()` creates generic addresses. The `mask()` operator (PTX ISA 7.1+) extracts byte ranges from addresses. Function and kernel names may be used as initializers for function-pointer tables. `.f16`, `.f16x2`, and `.pred` types do not support initializers.
