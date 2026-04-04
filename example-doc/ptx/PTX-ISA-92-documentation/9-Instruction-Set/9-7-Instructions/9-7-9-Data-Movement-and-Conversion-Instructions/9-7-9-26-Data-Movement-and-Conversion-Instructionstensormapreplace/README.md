# 9.7.9.26. Data Movement and Conversion Instructions: tensormap.replace

The `tensormap.replace` instruction modifies individual fields of a 1024-bit tensor-map object in .global or .shared::cta memory. Replaceable fields include global_address, rank, box_dim, global_dim, global_stride, element_stride, elemtype, interleave_layout, swizzle_mode, swizzle_atomicity, and fill_mode. Supports .tile mode. Requires sm_90a or sm_100+. Introduced in PTX ISA 8.3.
