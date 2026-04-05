# 9.7.13. Parallel Synchronization and Communication Instructions

- **9-7-13-1** — `bar`/`barrier`: CTA-level barrier sync, arrive, and reduction (popc/and/or) on 16 named barriers.
- **9-7-13-2** — `bar.warp.sync`: intra-warp barrier synchronization using a lane membermask.
- **9-7-13-3** — `barrier.cluster`: cluster-level arrive/wait synchronization for multi-CTA coordination.
- **9-7-13-4** — `membar`/`fence`: memory ordering fences at cta/cluster/gpu/sys scope with proxy variants.
- **9-7-13-5** — `atom`: atomic read-modify-write operations on global/shared memory for scalar and vector types.
- **9-7-13-6** — `red`: non-returning reduction operations on global/shared memory for scalar and vector types.
- **9-7-13-7** — `red.async`: asynchronous cluster-scoped reduction tracked by mbarrier completion.
- **9-7-13-8** — `vote` (deprecated): warp predicate reduction (.all/.any/.uni/ballot) without sync qualifier.
- **9-7-13-9** — `vote.sync`: synchronized warp predicate reduction/ballot using membermask.
- **9-7-13-10-to-9-7-13-11** — `match.sync` (value broadcast/compare across masked lanes) and `activemask` (active thread bitmask).
- **9-7-13-12** — `redux.sync`: warp-level synchronous reduction returning result to all participants.
- **9-7-13-13-to-9-7-13-14** — `griddepcontrol` (dependent-grid launch/wait control) and `elect.sync` (single-thread election from masked group).
- **9-7-13-15** — `mbarrier` family: 6 sub-sections covering the arrive/wait barrier primitive with async tracking.
- **9-7-13-16** — `tensormap.cp_fenceproxy`: fused shared-to-global copy with uni-directional proxy fence.
- **9-7-13-17** — `clusterlaunchcontrol.try_cancel`: asynchronously cancel a pending cluster launch.
- **9-7-13-18** — `clusterlaunchcontrol.query_cancel`: decode cancellation response from try_cancel.
