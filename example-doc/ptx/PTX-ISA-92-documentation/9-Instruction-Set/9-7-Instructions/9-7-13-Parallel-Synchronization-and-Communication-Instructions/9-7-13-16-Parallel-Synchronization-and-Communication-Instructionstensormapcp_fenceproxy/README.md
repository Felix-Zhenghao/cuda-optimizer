# 9.7.13.16. tensormap.cp_fenceproxy

A fused copy-and-fence operation that copies 128 bytes from shared::cta to global memory, then establishes a uni-directional proxy release from generic to tensormap proxy. Requires all warp threads to execute (.sync.aligned). Used to update tensor map objects before consumption.
