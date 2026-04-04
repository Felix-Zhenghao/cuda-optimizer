# 9.7.9.25.5.2. Data Movement and Conversion Instructions: cp.async.bulk.tensor

The `cp.async.bulk.tensor` instruction performs asynchronous tensor data copies using tensor-map objects. Supports directions: global-to-shared::cta, global-to-shared::cluster (with multicast), and shared::cta-to-global. Load modes include .tile, .tile::gather4/scatter4, .im2col, .im2col::w, and .im2col::w::128. Supports cta_group for CTA-pair signaling. Requires sm_90+. Introduced in PTX ISA 8.0.
