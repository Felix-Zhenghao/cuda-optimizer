# 9.7.13.15.16. mbarrier.test_wait / mbarrier.try_wait

test_wait is non-blocking; try_wait may suspend the thread with an optional timeout. Both check mbarrier phase completion using either an opaque state from mbarrier.arrive or a parity bit. Provide acquire semantics guaranteeing visibility of memory operations from the completed phase.
