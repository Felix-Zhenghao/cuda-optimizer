# 9.5--9.6. Divergence of Threads in Control Constructs / Machine-Specific Semantics of 16-bit Code

Explains thread divergence and convergence in CTAs: threads diverge at conditional branches and re-converge automatically, with `.uni` suffix marking uniform (non-divergent) branches for optimization. Also covers machine-specific 16-bit instruction semantics -- on 32-bit GPUs, 16-bit registers are promoted to 32-bit, potentially exposing extra precision bits that can affect program behavior.
