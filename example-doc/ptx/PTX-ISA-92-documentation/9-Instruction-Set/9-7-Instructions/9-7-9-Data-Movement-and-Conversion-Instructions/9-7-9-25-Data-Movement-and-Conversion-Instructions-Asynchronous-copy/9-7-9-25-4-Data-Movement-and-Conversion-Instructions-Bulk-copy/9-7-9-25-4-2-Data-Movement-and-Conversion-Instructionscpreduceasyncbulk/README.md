# 9.7.9.25.4.2. Data Movement and Conversion Instructions: cp.reduce.async.bulk

The `cp.reduce.async.bulk` instruction initiates asynchronous element-wise reduction between arrays in different state spaces. Supports reduction operations (.add, .min, .max, .inc, .dec, .and, .or, .xor) with integer and floating-point types. Copy directions: shared::cta-to-shared::cluster (mbarrier) and shared::cta-to-global (bulk_group). Size must be 16-byte aligned. Requires sm_90+. Introduced in PTX ISA 8.0.
