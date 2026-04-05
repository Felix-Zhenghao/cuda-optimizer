# Programming Guidelines - Design Patterns

## Hierarchical Organization

The [CUTLASS 3.0 GEMM API](gemm_api_3x.html) document
explains CUTLASS 3.0's hierarchical organization,
based conceptually on parallelization strategy.
This differs from CUTLASS 2.x's approach,
which more closely mirrors the GPU hardware hierarchy
of thread blocks, warps, and threads.

## Design Patterns

CUTLASS aims for the highest performance possible on NVIDIA GPUs.
It also offers flexible components that can be assembled and customized
to solve new problems related to deep learning and linear algebra.
Given a tradeoff between simplicity and performance,
CUTLASS chooses performance.
Consequently, several design patterns are necessary
to yield a composable structure
while also satisfying these performance objectives.

### Templates

CUDA C++ templates and modern generic programming techniques enable CUTLASS device code to span a large design space.

This design space includes:

* Mixed precision arithmetic and data storage
* Kernels specialized for layout and problem size
* Support for kernel fusion

Moreover, templates provided a structured approach to collecting compile-time constants such as tile dimensions. These
must be template arguments to target static array allocation and take advantage of loop unrolling, constant folding,
and function inlining.

### Constant Memory

Several CUTLASS template classes exhibit a pattern in which problem-specific internal state is known at kernel
launch time and remains invariant throughout the execution of a kernel. For example, tile iterators compute several
offsets based on the strides of the input tensor that is added to an internal pointer when loading the elements
of a tile. These are computed from the tensor stride and never updated; the per-thread internal state consists
only of the internal global memory pointer.

CUTLASS can take advantage of this CUDA grid-invariant property by constructing the object in host code and passing
a composed parameters structure to the kernel. This confers two benefits: (1.) invariant state is held in constant
memory, and (2.) there is no overhead to compute the initial state by each thread.

The design pattern in CUTLASS is for classes with nontrivial constructors to define `struct Params` as an inner class
which contains grid-invariant state. These should define a constructor and an `initialize()` method. The `Params`
structure should also include a data member corresponding to each data member in the parent class, so these too can
be properly constructed in host code. The parent class should define a constructor which accepts `Params const &` as
its first argument.

### Composable Shared Memory

Shared memory requires explicit effort by the programmer to allocate and de-allocate. CUTLASS follows the paradigm
introduced by [CUB](https://nvlabs.github.io/cub/) to define composed structures for storing data intended to be held
in shared memory. Any object requiring shared memory storage for itself or its data members should define a child
structure called `SharedStorage`. This holds data needed by the class and also instantiates `SharedStorage`
objects for each data member.

To be consistent, this pattern defines a convention in which classes define internal shared memory storage requirements.
Classes should consider all SharedStorage structures to be opaque other than their own child class. When the lifetimes
of child objects are known to be non-overlapping, `union`s may be used to alias multiple SharedStorage objects to the same
shared memory region and reduce overall shared memory capacity. Developers should carefully note that C++ `union` rules
require that they only access the most recently written ("active") member of the `union`; this differs from C rules.

For host to device ABI compatibility, inheritance from a class is only permitted if the superclass is unique to the
child class. This is most easily achieved by templating the parent class by the child class (CRTP).

### Loop Unrolling

CUTLASS requires tiles of data to be stored in registers for high-bandwidth access. Simultaneously, high-throughput math instructions
must be issued concurrently with memory instructions to hide latency with relatively few concurrent threads. These objectives are
achieved by unrolling loops whose iteration counts are known at compile time.

Consequently, most loops within the CUTLASS GEMM implementation are specified by constant values and template arguments. The CUDA compiler
is able to unroll the loop bodies, map array elements to registers, and construct an efficient instruction schedule.

All loops expected to be unrolled should be annotated with `CUTLASS_PRAGMA_UNROLL` to explicitly direct the compiler
to unroll them.

```cpp
int const kN = 8;
Array<float, kN> x;                       // Array we would like to store in registers

CUTLASS_PRAGMA_UNROLL                     // Directs the CUDA compiler to unroll this loop.
for (int idx = 0; idx < kN; ++idx) {      // Loop has constant number of iterations.

  x[i] = float(idx);                      // Indirect access by induction variable results in
                                          // direct register access.
}
```
