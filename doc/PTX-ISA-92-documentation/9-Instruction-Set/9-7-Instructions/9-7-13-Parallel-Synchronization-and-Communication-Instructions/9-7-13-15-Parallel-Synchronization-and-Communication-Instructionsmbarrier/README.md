# 9.7.13.15. mbarrier Instructions

- **9-7-13-15-1-to-9-7-13-15-12** — mbarrier object specification: size/alignment, internal state, lifecycle, init/inval, arrive-on semantics, and cluster-scope operations (sections 15.1-15.12).
- **9-7-13-15-13** — `mbarrier.arrive`: standard arrive-on with optional count, expect_tx, and noComplete variants.
- **9-7-13-15-14** — `mbarrier.arrive_drop`: arrive while decrementing expected arrival count for threads opting out.
- **9-7-13-15-15** — `cp.async.mbarrier.arrive`: link prior cp.async completions to mbarrier tracking.
- **9-7-13-15-16** — `mbarrier.test_wait` / `mbarrier.try_wait`: non-blocking and potentially-blocking phase completion checks.
- **9-7-13-15-17** — `mbarrier.pending_count`: extract pending arrival count from opaque mbarrier state token.
