# 9.7.9.25.3.1. Data Movement and Conversion Instructions: cp.async

The `cp.async` instruction initiates a non-blocking asynchronous copy from global to shared memory (4, 8, or 16 bytes). Supports .ca/.cg cache operators, optional src-size for partial copy with zero-fill, ignore-src predicate for zeroing destination, cache hints, and prefetch sizes. Completion tracked via async-group or mbarrier mechanisms. Requires sm_80+. Introduced in PTX ISA 7.0.
