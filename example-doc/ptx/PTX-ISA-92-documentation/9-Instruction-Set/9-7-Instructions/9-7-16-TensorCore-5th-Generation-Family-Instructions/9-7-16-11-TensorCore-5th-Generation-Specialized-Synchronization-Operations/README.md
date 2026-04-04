# 9.7.16.11. TensorCore 5th Generation Specialized Synchronization Operations

Documents `tcgen05.fence` with two variants: `::before_thread_sync` creates an ordering point before a thread-level synchronization, and `::after_thread_sync` creates one after. Required for cross-proxy memory ordering between async tcgen05 and thread-level operations.
