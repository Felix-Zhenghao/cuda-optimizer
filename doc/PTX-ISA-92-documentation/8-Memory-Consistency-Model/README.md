# Memory-Consistency-Model

- **8-1-Scope-and-applicability-of-the-model** — Atomicity limitations for system-scope strong operations communicating with host CPU on certain hardware.
- **8-2-Memory-operations** — Definitions of overlapping, aliased, multimem addresses, vector/packed type modeling, and memory initialization.
- **8-3-State-spaces** — Memory consistency relations span all state spaces but side-effects are only visible within the same state space.
- **8-4-Operation-types** — mmio operations (MMIO registers, strict hardware constraints) and volatile operations (relaxed system-scope with instruction-count preservation).
- **8-5-to-8-7-Scope** — Four scopes (.cta/.cluster/.gpu/.sys), proxies and proxy fences, and conflict/data-race definitions with mixed-size limitations.
- **8-8-Release-and-Acquire-Patterns** — Release and acquire patterns formed by release/acquire operations or fences with strong reads/writes enabling inter-thread synchronization.
- **8-9-Ordering-of-memory-operations** — Program order, synchronization, causality order, coherence order, and communication order definitions.
- **8-10-Axioms** — Six formal axioms (Coherence, Fence-SC, Atomicity, No-Thin-Air, SC-Per-Location, Causality) governing valid executions.
- **8-11-Special-Cases** — Atomic reduction (red) instructions do not form acquire patterns; use atom instead for acquire semantics.
