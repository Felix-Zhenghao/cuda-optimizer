# 4.14. Memory Synchronization Domains

Memory synchronization domains reduce unnecessary memory fence overhead by isolating traffic between kernel groups. Explains fence interference in multi-kernel workloads, how to assign kernels to logical domains to limit fence scope, and how to use domain launch attributes on streams, individual launches, and CUDA graph nodes.
