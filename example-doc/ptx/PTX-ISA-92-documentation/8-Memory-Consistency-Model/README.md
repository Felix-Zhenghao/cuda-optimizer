# 8. Memory Consistency Model

- **8-1-Scope-and-applicability-of-the-model**: Atomicity limitations at system scope.
- **8-2-Memory-operations**: Overlap, aliases, multimem, vector/packed types, initialization.
- **8-3-State-spaces**: State space independence of consistency relations.
- **8-4-Operation-types**: Mmio and volatile operation semantics.
- **8-5-to-8-7-Scope**: Scopes, proxies, conflict, and data-race definitions.
- **8-8-Release-and-Acquire-Patterns**: Release and acquire synchronization patterns.
- **8-9-Ordering-of-memory-operations**: Program order, observation, causality, coherence, communication orders.
- **8-10-Axioms**: Formal axioms (coherence, atomicity, no-thin-air, SC-per-location, causality).
- **8-11-Special-Cases**: Reduction operations and acquire pattern limitations.
