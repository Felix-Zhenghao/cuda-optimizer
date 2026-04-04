# 9.7.9.15. Data Movement and Conversion Instructions: prefetch, prefetchu

The `prefetch` instruction brings a cache line from global or local memory into L1 or L2 cache. Supports eviction priority hints and tensormap prefetching for cp.async.bulk.tensor. The `prefetchu` instruction prefetches to the uniform cache using a generic address. Both are performance hints. Requires sm_20+; tensormap requires sm_90+. Introduced in PTX ISA 2.0.
