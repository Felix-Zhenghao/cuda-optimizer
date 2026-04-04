# 9.7.9.25.5.3. Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor

The `cp.reduce.async.bulk.tensor` instruction performs asynchronous element-wise reduction on tensor data from shared::cta to global memory using tensor-map objects. Supports reduction operations (.add, .min, .max, .inc, .dec, .and, .or, .xor) with .tile and .im2col_no_offs load modes. Uses bulk_group completion. Requires sm_90+. Introduced in PTX ISA 8.0.
