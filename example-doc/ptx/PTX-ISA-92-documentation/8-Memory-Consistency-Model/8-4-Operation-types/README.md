# 8.4 Operation Types

Covers mmio and volatile operation semantics. Mmio operations preserve exact read/write counts for I/O device registers. Volatile operations are relaxed system-scope with instruction count preservation but allow operation merging; prefer strong operations for synchronization.
