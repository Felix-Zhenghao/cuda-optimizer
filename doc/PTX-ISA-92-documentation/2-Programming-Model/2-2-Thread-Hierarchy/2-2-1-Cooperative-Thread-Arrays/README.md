# 2.2.1. Cooperative Thread Arrays

A Cooperative Thread Array (CTA) is a group of threads executing a kernel concurrently. Threads share a unique 3D identifier (tid), synchronize via barrier points, and execute in SIMT groups called warps. The warp size is 32 threads and is accessible via WARP_SZ.
