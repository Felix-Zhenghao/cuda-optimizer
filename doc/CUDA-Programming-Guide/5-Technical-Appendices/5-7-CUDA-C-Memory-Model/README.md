# 5.7. CUDA C++ Memory Model

Documents how CUDA C++ extends the standard C++ memory model with thread scopes (`thread_scope_system/device/block/thread`) to handle non-uniform synchronization costs across GPUs and CPUs. Covers atomicity conditions, data race definitions, and message-passing examples using `cuda::atomic_ref`.
