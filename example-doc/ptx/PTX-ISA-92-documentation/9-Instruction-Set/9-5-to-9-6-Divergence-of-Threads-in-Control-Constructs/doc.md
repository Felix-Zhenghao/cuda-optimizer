## 9.5. [Divergence of Threads in Control Constructs](#divergence-of-threads-in-control-constructs)

Threads in a CTA execute together, at least in appearance, until they come to a conditional control
construct such as a conditional branch, conditional function call, or conditional return. If threads
execute down different control flow paths, the threads are called *divergent*. If all of the threads
act in unison and follow a single control flow path, the threads are called *uniform*. Both
situations occur often in programs.

A CTA with divergent threads may have lower performance than a CTA with uniformly executing threads,
so it is important to have divergent threads re-converge as soon as possible. All control constructs
are assumed to be divergent points unless the control-flow instruction is marked as uniform, using
the `.uni` suffix. For divergent control flow, the optimizing code generator automatically
determines points of re-convergence. Therefore, a compiler or code author targeting PTX can ignore
the issue of divergent threads, but has the opportunity to improve performance by marking branch
points as uniform when the compiler or author can guarantee that the branch point is non-divergent.

---

### 9.6.1. [Machine-Specific Semantics of 16-bit Code](#machine-specific-semantics-of-16-bit-code)

A PTX program may execute on a GPU with either a 16-bit or a 32-bit data path. When executing on a
32-bit data path, 16-bit registers in PTX are mapped to 32-bit physical registers, and 16-bit
computations are *promoted* to 32-bit computations. This can lead to computational differences
between code run on a 16-bit machine versus the same code run on a 32-bit machine, since the
promoted computation may have bits in the high-order half-word of registers that are not present in
16-bit physical registers. These extra precision bits can become visible at the application level,
for example, by a right-shift instruction.

At the PTX language level, one solution would be to define semantics for 16-bit code that is
consistent with execution on a 16-bit data path. This approach introduces a performance penalty for
16-bit code executing on a 32-bit data path, since the translated code would require many additional
masking instructions to suppress extra precision bits in the high-order half-word of 32-bit
registers.

Rather than introduce a performance penalty for 16-bit code running on 32-bit GPUs, the semantics of
16-bit instructions in PTX is machine-specific. A compiler or programmer may chose to enforce
portable, machine-independent 16-bit semantics by adding explicit conversions to 16-bit values at
appropriate points in the program to guarantee portability of the code. However, for many
performance-critical applications, this is not desirable, and for many applications the difference
in execution is preferable to limiting performance.
