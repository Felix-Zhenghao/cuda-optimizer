# 3.1. A Set of SIMT Multiprocessors

NVIDIA GPUs consist of Streaming Multiprocessors (SMs) that execute thread blocks as warps using the SIMT model. Divergent warp branches are serialized. Resource limits (registers, shared memory) per SM determine how many blocks can run concurrently; insufficient resources prevent kernel launch.
