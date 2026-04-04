# 9.7.9.25.5.4. Data Movement and Conversion Instructions: cp.async.bulk.prefetch.tensor

The `cp.async.bulk.prefetch.tensor` instruction provides a hint to prefetch tensor data from global memory into L2 cache using tensor-map objects. Supports .tile, .tile::gather4, .im2col, .im2col::w, and .im2col::w::128 load modes with 1D-5D tensors. Optional cache policy via .L2::cache_hint. Performance hint only. Requires sm_90+. Introduced in PTX ISA 8.0.
