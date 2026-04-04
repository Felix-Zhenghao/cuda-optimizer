## 3.2. [Independent Thread Scheduling](#independent-thread-scheduling)

On architectures prior to Volta, warps used a single program counter shared amongst all 32 threads
in the warp together with an active mask specifying the active threads of the warp. As a result,
threads from the same warp in divergent regions or different states of execution cannot signal each
other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or
mutexes can easily lead to deadlock, depending on which warp the contending threads come from.

Starting with the Volta architecture, *Independent Thread Scheduling* allows full concurrency
between threads, regardless of warp. With *Independent Thread Scheduling*, the GPU maintains
execution state per thread, including a program counter and call stack, and can yield execution at a
per-thread granularity, either to make better use of execution resources or to allow one thread to
wait for data to be produced by another. A schedule optimizer determines how to group active threads
from the same warp together into SIMT units. This retains the high throughput of SIMT execution as
in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at
sub-warp granularity.

*Independent Thread Scheduling* can lead to a rather different set of threads participating in the
executed code than intended if the developer made assumptions about warp-synchronicity of previous
hardware architectures. In particular, any warp-synchronous code (such as synchronization-free,
intra-warp reductions) should be revisited to ensure compatibility with Volta and beyond. See the
section on Compute Capability 7.x in the *Cuda Programming Guide* for further details.

---

## 3.3. [On-chip Shared Memory](#on-chip-shared-memory)

As illustrated by [Figure 4](#set-of-simt-multiprocessors-hardware-model), each multiprocessor has
on-chip memory of the four following types:

* One set of local 32-bit *registers* per processor,
* A parallel data cache or *shared memory* that is shared by all scalar processor cores and is where
  the shared memory space resides,
* A read-only *constant cache* that is shared by all scalar processor cores and speeds up reads from
  the constant memory space, which is a read-only region of device memory,
* A read-only *texture cache* that is shared by all scalar processor cores and speeds up reads from
  the texture memory space, which is a read-only region of device memory; each multiprocessor
  accesses the texture cache via a *texture unit* that implements the various addressing modes and
  data filtering.

The local and global memory spaces are read-write regions of device memory.
