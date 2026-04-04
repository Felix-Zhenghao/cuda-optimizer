# 9.7.9.25.4.5. Data Movement and Conversion Instructions: multimem.cp.reduce.async.bulk

The `multimem.cp.reduce.async.bulk` instruction initiates element-wise asynchronous reduction from shared::cta to all GPU memory locations referenced by a multimem global address. Supports reduction operations (.add, .min, .max, .inc, .dec, .and, .or, .xor) with integer and floating-point types. Uses bulk_group completion. Has .relaxed.sys ordering. Requires sm_90+. Introduced in PTX ISA 9.1.
