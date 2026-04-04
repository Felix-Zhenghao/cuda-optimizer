# 9.7.9.25.1. Completion Mechanisms for Asynchronous Copy Operations

Describes two completion mechanisms for async copy operations. The async-group mechanism uses commit/wait operations to group and track per-thread async operations in order. The mbarrier-based mechanism uses mbarrier objects to track completion via phase-based signaling, allowing multi-thread coordination. Both ensure results are visible only after completion is observed.
