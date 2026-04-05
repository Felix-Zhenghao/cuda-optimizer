# 4.3. Stream-Ordered Memory Allocator

Stream-ordered memory allocator enables cudaMallocAsync/cudaFreeAsync to order allocations within streams, avoiding global synchronization. Covers memory pool creation, attribute tuning (release thresholds, access controls), pool sharing across streams and processes, best practices, and caveats around VRAM limits and IPC usage.
