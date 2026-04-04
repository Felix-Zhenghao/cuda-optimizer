# 9.7.9.25.4.3. Data Movement and Conversion Instructions: cp.async.bulk.prefetch

The `cp.async.bulk.prefetch` instruction provides a hint to asynchronously prefetch data from global memory into L2 cache. Size must be a multiple of 16 bytes with 16-byte alignment. Supports optional cache eviction policy via .L2::cache_hint. Treated as a weak memory operation and performance hint only. Requires sm_90+. Introduced in PTX ISA 8.0.
