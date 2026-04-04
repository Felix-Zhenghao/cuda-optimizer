# 9.7.4.9 Half Precision Floating Point Instructions: tanh

Documents approximate `tanh` for f16, f16x2, bf16, and bf16x2 types. SIMD variants operate on packed half-word pairs. Maximum absolute error is 2^-10.987 for f16, 2^-8 for bf16. Requires sm_75+; bf16 variants require sm_90+.
