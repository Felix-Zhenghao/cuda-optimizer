# Parallel Synchronization and Communication Instructions

- **bar, barrier** -- CTA-level barrier synchronization with sync, arrive, and reduction modes.
- **bar.warp.sync** -- Warp-level barrier synchronization with membermask and memory ordering.
- **barrier.cluster** -- Cluster-level barrier with separate arrive/wait and configurable semantics.
- **membar / fence** -- Memory ordering enforcement with multiple scopes, semantics, and proxy variants.
- **atom** -- Atomic read-modify-write operations on global/shared memory with scope qualifiers.
- **red** -- Reduction operations on global/shared memory without returning original values.
- **red.async** -- Asynchronous reductions with mbarrier tracking or release semantics.
- **vote (deprecated)** -- Deprecated warp predicate reduction; replaced by vote.sync.
- **vote.sync** -- Synchronized warp predicate reduction with explicit membermask.
- **match.sync, activemask** -- Warp-level value matching and active thread mask queries.
- **redux.sync** -- Warp-level synchronized arithmetic and bitwise reductions.
- **griddepcontrol, elect.sync** -- Grid dependency control and deterministic leader election.
- **mbarrier** -- Asynchronous barrier object with phase tracking, transactions, and cluster support.
- **tensormap.cp_fenceproxy** -- Fused copy-and-fence for tensor map object updates.
- **clusterlaunchcontrol.try_cancel** -- Async cancellation request for not-yet-launched clusters.
- **clusterlaunchcontrol.query_cancel** -- Decode cancellation response for ctaid extraction.
