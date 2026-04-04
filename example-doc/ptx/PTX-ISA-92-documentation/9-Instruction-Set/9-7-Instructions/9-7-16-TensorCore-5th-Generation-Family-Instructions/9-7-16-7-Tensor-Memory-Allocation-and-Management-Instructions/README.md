# 9.7.16.7. Tensor Memory Allocation and Management Instructions

Documents `tcgen05.alloc`, `tcgen05.dealloc`, and `tcgen05.relinquish_alloc_permit` for dynamic Tensor Memory management. Alloc returns a base address; dealloc frees it. Relinquish_alloc_permit releases allocation permission. All require synchronized CTA-level execution.
