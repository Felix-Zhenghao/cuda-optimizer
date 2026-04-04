# 9.7.16.6.5. Shared Memory Accesses

Explains that tcgen05.mma and tcgen05.cp shared memory accesses occur in the async proxy. Cross-proxy access to the same memory location requires `tcgen05.fence::before_thread_sync` followed by a thread-level fence. Provides the correct synchronization sequence.
