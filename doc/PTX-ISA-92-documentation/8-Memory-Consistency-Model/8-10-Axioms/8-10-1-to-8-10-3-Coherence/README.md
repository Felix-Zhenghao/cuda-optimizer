# Coherence

Merged from: 8-10-1-Coherence to 8-10-3-Atomicity

Covers three axioms: Coherence (writes ordered in causality must be ordered in coherence order), Fence-SC (Fence-SC order cannot contradict causality order), and Atomicity (morally strong conflicting operations use single-copy atomicity; atomic RMW operations cannot be interleaved with morally strong overlapping writes). Includes litmus tests illustrating atomicity guarantees.
