# mbarrier Instructions

- **Size/alignment/contents/lifecycle/phases/operations (15.1-15.12)** -- Mbarrier object fundamentals: type, lifecycle, phase model, arrive-on, expect-tx, and complete-tx operations; init, inval, expect_tx, complete_tx instructions.
- **mbarrier.arrive** -- Arrive-on operation with optional count, expect_tx, and noComplete modifiers.
- **mbarrier.arrive_drop** -- Permanently drops a thread from future phases and performs arrive-on.
- **cp.async.mbarrier.arrive** -- Links prior cp.async completion to mbarrier tracking.
- **mbarrier.test_wait / try_wait** -- Non-blocking and potentially-blocking phase completion checks.
- **mbarrier.pending_count** -- Queries pending arrival count from opaque mbarrier state.
