# 8.8. Release and Acquire Patterns

Defines release and acquire patterns used for inter-thread synchronization. A release pattern makes prior operations visible to other threads; an acquire pattern makes other threads' operations visible. Patterns can be formed by release/acquire operations, fences combined with strong reads/writes. Atomic reductions do not form acquire patterns.
