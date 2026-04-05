# Axioms

- **8-10-1-to-8-10-3-Coherence** — Coherence, Fence-SC, and Atomicity axioms with litmus tests demonstrating morally strong operation guarantees.
- **8-10-4-No-Thin-Air** — Forbids self-satisfying speculative value cycles; permits load buffering without data dependencies.
- **8-10-5-Sequential-Consistency-Per-Location** — Pairwise morally strong overlapping operations are strictly sequentially consistent (CoRR litmus test).
- **8-10-6-Causality** — Communication order cannot contradict causality order; covers MP, CoWR alias proxy fence, and SB litmus tests.
