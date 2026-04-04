# 9.7.9.14. Data Movement and Conversion Instructions: multimem.ld_reduce, multimem.st, multimem.red

Documents three multimem operations on multi-GPU memory addresses. `multimem.ld_reduce` loads from all memory locations and reduces results. `multimem.st` stores to all locations. `multimem.red` performs reduction on all locations. Supports integer (.add, .min, .max, .and, .or, .xor) and floating-point (.add, .min, .max) operations with various types including f16, bf16, e5m2, e4m3. Requires sm_90+. Introduced in PTX ISA 8.1.
