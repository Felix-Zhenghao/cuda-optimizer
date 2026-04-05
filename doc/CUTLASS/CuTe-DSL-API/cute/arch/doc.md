# arch

The `cute.arch` module provides lightweight wrappers for NVVM Operation builders which implement CUDA built-in
device functions such as `thread_idx`. It integrates seamlessly with CuTe DSL types.

These wrappers enable source location tracking through the `@dsl_user_op`
decorator. The module includes the following functionality:

* Core CUDA built-in functions such as `thread_idx`, `warp_idx`, `block_dim`, `grid_dim`, `cluster_dim`, and related functions
* Memory barrier management functions including `mbarrier_init`, `mbarrier_arrive`, `mbarrier_wait`, and associated operations
* Low-level shared memory (SMEM) management capabilities, with `SmemAllocator` as the recommended interface
* Low-level tensor memory (TMEM) management capabilities, with `TmemAllocator` as the recommended interface

## API documentation

cutlass.cute.arch.make\_warp\_uniform( : *value: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32
:   Provides a compiler hint indicating that the specified value is invariant across all threads in the warp,
    which may enable performance optimizations.

    Parameters:
    :   **value** (*Int*) -- The integer value to be marked as warp-uniform.

    Returns:
    :   The input value, marked as warp-uniform.

    Return type:
    :   Int32

cutlass.cute.arch.elect\_one(*\**, *loc=None*, *ip=None*) → IfOpRegion
:   Elects one thread within a warp.

    ```
    with elect_one():
        # Only one thread in the warp executes the code in this context
        pass
    ```

cutlass.cute.arch.mbarrier\_init( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *cnt: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Initializes a mbarrier with the specified thread arrival count.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **cnt** (*Int*) -- The arrival count of the mbarrier

cutlass.cute.arch.mbarrier\_init\_fence(*\**, *loc=None*, *ip=None*) → None
:   A fence operation that applies to the mbarrier initializations.

cutlass.cute.arch.mbarrier\_arrive\_and\_expect\_tx( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *bytes: cutlass.cute.typing.Int*, : *peer\_cta\_rank\_in\_cluster=None*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Arrives on a mbarrier and expects a specified number of transaction bytes.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **bytes** (*Int*) -- The number of transaction bytes
        * **peer\_cta\_rank\_in\_cluster** -- An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTA's
          SMEM.

cutlass.cute.arch.mbarrier\_expect\_tx( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *bytes: cutlass.cute.typing.Int*, : *peer\_cta\_rank\_in\_cluster=None*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Expects a specified number of transaction bytes without an arrive.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **bytes** (*Int*) -- The number of transaction bytes
        * **peer\_cta\_rank\_in\_cluster** -- An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTA's
          SMEM.

cutlass.cute.arch.mbarrier\_wait( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Waits on a mbarrier with a specified phase.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **phase** (*Int*) -- The phase to wait for (either 0 or 1)

cutlass.cute.arch.mbarrier\_try\_wait( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean
:   Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **phase** (*Int*) -- The phase to wait for (either 0 or 1)

    Returns:
    :   A boolean value indicating whether the wait operation was successful

    Return type:
    :   Boolean

cutlass.cute.arch.mbarrier\_conditional\_try\_wait( : *cond*, : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *phase: cutlass.cute.typing.Int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean
:   Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    Parameters:
    :   * **cond** -- A boolean predicate
        * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **phase** (*Int*) -- The phase to wait for (either 0 or 1)

    Returns:
    :   A boolean value indicating whether the wait operation was successful

    Return type:
    :   Boolean

cutlass.cute.arch.mbarrier\_arrive( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *peer\_cta\_rank\_in\_cluster: cutlass.cute.typing.Int | None = None*, : *arrive\_count: cutlass.cute.typing.Int = 1*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Arrives on an mbarrier.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **peer\_cta\_rank\_in\_cluster** -- An optional CTA rank in cluster. If provided, the pointer to
          the mbarrier is converted to a remote address in the peer CTA's
          SMEM.

cutlass.cute.arch.lane\_idx(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32
:   Returns the lane index of the current thread within the warp.

cutlass.cute.arch.warp\_idx(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32
:   Returns the warp index within a CTA.

cutlass.cute.arch.thread\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the thread index within a CTA.

cutlass.cute.arch.block\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the number of threads in each dimension of the CTA.

cutlass.cute.arch.block\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the CTA identifier within a grid.

cutlass.cute.arch.grid\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the number of CTAs in each dimension of the grid.

cutlass.cute.arch.cluster\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the cluster identifier within a grid.

cutlass.cute.arch.cluster\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the number of clusters in each dimension of the grid.

cutlass.cute.arch.cluster\_size(*\**, *loc=None*, *ip=None*) → cutlass.cute.typing.Int32
:   Returns the number of CTA within the cluster.

cutlass.cute.arch.block\_in\_cluster\_idx( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the CTA index within a cluster across all dimensions.

cutlass.cute.arch.block\_in\_cluster\_dim( : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   Returns the dimensions of the cluster.

cutlass.cute.arch.block\_idx\_in\_cluster( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32
:   Returns the linearized identifier of the CTA within the cluster.

cutlass.cute.arch.barrier( : *\**, : *barrier\_id=None*, : *number\_of\_threads=None*, : *loc=None*, : *ip=None*, ) → None
:   Creates a barrier, optionally named.

cutlass.cute.arch.barrier\_arrive( : *\**, : *barrier\_id=None*, : *number\_of\_threads=None*, : *loc=None*, : *ip=None*, ) → None

cutlass.cute.arch.sync\_threads(*\**, *loc=None*, *ip=None*) → None
:   Synchronizes all threads within a CTA.

cutlass.cute.arch.sync\_warp( : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Performs a warp-wide sync with an optional mask.

cutlass.cute.arch.fence\_acq\_rel\_cta(*\**, *loc=None*, *ip=None*) → None
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_cluster(*\**, *loc=None*, *ip=None*) → None
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_gpu(*\**, *loc=None*, *ip=None*) → None
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.fence\_acq\_rel\_sys(*\**, *loc=None*, *ip=None*) → None
:   Fence operation with acquire-release semantics.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.arch.cp\_async\_commit\_group(*\**, *loc=None*, *ip=None*) → None
:   Commits all prior initiated but uncommitted cp.async instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group).

cutlass.cute.arch.cp\_async\_wait\_group(*n*, *\**, *loc=None*, *ip=None*) → None
:   Waits till only a specified numbers of cp.async groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).

cutlass.cute.arch.cp\_async\_bulk\_commit\_group(*\**, *loc=None*, *ip=None*) → None
:   Commits all prior initiated but uncommitted cp.async.bulk instructions.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group).

cutlass.cute.arch.cp\_async\_bulk\_wait\_group( : *group*, : *\**, : *read=None*, : *loc=None*, : *ip=None*, ) → None
:   Waits till only a specified numbers of cp.async.bulk groups are pending.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

cutlass.cute.arch.cluster\_wait(*\**, *loc=None*, *ip=None*) → None
:   A cluster-wide wait operation.

cutlass.cute.arch.cluster\_arrive(*\**, *aligned=None*, *loc=None*, *ip=None*) → None
:   A cluster-wide arrive operation.

cutlass.cute.arch.cluster\_arrive\_relaxed(*\**, *aligned=None*, *loc=None*, *ip=None*) → None
:   A cluster-wide arrive operation with relaxed semantics.

cutlass.cute.arch.vote\_ballot\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int32
:   Performs a ballot operation across the warp.

    It copies the predicate from each thread in mask into the corresponding bit position of
    destination register d, where the bit position corresponds to the thread's lane id.

    Parameters:
    :   * **pred** (*Boolean*) -- The predicate value for the current thread
        * **mask** (*Int**,* *optional*) -- A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)

    Returns:
    :   A 32-bit integer where each bit represents a thread's predicate value

    Return type:
    :   Int32

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_any\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean
:   True if source predicate is True for any non-exited threads in mask. Negate the source
    predicate to compute .none.

    Parameters:
    :   * **pred** (*Boolean*) -- The predicate value for the current thread
        * **mask** (*Int**,* *optional*) -- A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_all\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean
:   True if source predicate is True for all non-exited threads in mask. Negate the source
    predicate to compute .none.

    Parameters:
    :   * **pred** (*Boolean*) -- The predicate value for the current thread
        * **mask** (*Int**,* *optional*) -- A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

cutlass.cute.arch.vote\_uni\_sync( : *pred: cutlass.cute.typing.Boolean*, : *mask: cutlass.cute.typing.Int = 4294967295*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Boolean
:   True f source predicate has the same value in all non-exited threads in mask. Negating
    the source predicate also computes .uni

    Parameters:
    :   * **pred** (*Boolean*) -- The predicate value for the current thread
        * **mask** (*Int**,* *optional*) -- A 32-bit integer mask specifying which threads participate, defaults to all
          threads (0xFFFFFFFF)

    Returns:
    :   A boolean value indicating if the source predicate is True for all non-exited
        threads in mask

    Return type:
    :   Boolean

cutlass.cute.arch.warp\_redux\_sync( : *value: cutlass.cute.typing.Numeric*, : *kind: Literal['fmax', 'fmin', 'max', 'min', 'add', 'xor', 'or', 'and']*, : *mask\_and\_clamp: cutlass.cute.typing.Int = 4294967295*, : *\**, : *abs: bool | None = None*, : *nan: bool | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Perform warp-level reduction operation across threads.

    Reduces values from participating threads in a warp according to the specified operation.
    All threads in the mask receive the same result.

    Parameters:
    :   * **value** (*Numeric*) -- Input value to reduce
        * **kind** (*Literal**[**"add"**,* *"and"**,* *"max"**,* *"min"**,* *"or"**,* *"xor"**,* *"fmin"**,* *"fmax"**]*) -- Reduction operation. Supported operations:
          - Integer types (Int32/Uint32): "add", "and", "max", "min", "or", "xor"
          - Float types (Float32): "fmax", "fmin" (or "max"/"min" which auto-convert to "fmax"/"fmin")
        * **mask\_and\_clamp** (*Int*) -- Warp participation mask (default: FULL\_MASK = 0xFFFFFFFF)
        * **abs** (*bool*) -- Apply absolute value before reduction (float types only)
        * **nan** (*Optional**[**bool**]*) -- Enable NaN propagation for fmax/fmin operations (float types only)

    Returns:
    :   Reduced value (same for all participating threads)

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_max\_float32( : *ptr*, : *value: cutlass.cute.typing.Float32*, : *\**, : *positive\_only: bool = True*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32
:   Performs an atomic max operation on a float32 value in global memory.

    This implementation works correctly for non-negative values (>= 0) using direct bitcast.

    Parameters:
    :   * **ptr** -- Pointer to the memory location
        * **value** (*Float32*) -- The float32 value to compare and potentially store (should be >= 0 for correct results)
        * **positive\_only** (*bool*) -- If True (default), assumes input values are non-negative.
          This parameter is provided for API compatibility and future extensions.

    Returns:
    :   The old value at the memory location

    Return type:
    :   Float32

cutlass.cute.arch.atomic\_add( : *ptr*, : *val: cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value
:   Performs an atomic addition operation.

    Atomically adds val to the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Union**[**Numeric**,* *ir.Value**]*) -- Value to add (scalar Numeric or vector ir.Value)
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Union[Numeric, ir.Value]

cutlass.cute.arch.atomic\_and( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic bitwise AND operation.

    Atomically computes bitwise AND of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value for AND operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_or( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic bitwise OR operation.

    Atomically computes bitwise OR of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value for OR operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_xor( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic bitwise XOR operation.

    Atomically computes bitwise XOR of val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value for XOR operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_max( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic maximum operation.

    Atomically computes maximum of val and the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value for MAX operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_min( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic minimum operation.

    Atomically computes minimum of val and the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value for MIN operation
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_exch( : *ptr*, : *val: cutlass.cute.typing.Numeric*, : *\**, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic exchange operation.

    Atomically exchanges val with the value at memory location ptr and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location
        * **val** (*Numeric*) -- Value to exchange
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.atomic\_cas( : *ptr*, : *\**, : *cmp: cutlass.cute.typing.Numeric*, : *val: cutlass.cute.typing.Numeric*, : *sem: Literal['relaxed', 'release', 'acquire', 'acq\_rel'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs an atomic compare-and-swap (CAS) operation.

    Atomically compares the value at the memory location with cmp. If they are equal,
    stores val at the memory location and returns the old value.

    Parameters:
    :   * **ptr** -- Pointer to memory location. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **cmp** (*Numeric*) -- Value to compare against current memory value
        * **val** (*Numeric*) -- Value to store if comparison succeeds
        * **sem** (*Optional**[**Literal**[**"relaxed"**,* *"release"**,* *"acquire"**,* *"acq\_rel"**]**]*) -- Memory semantic ("relaxed", "release", "acquire", "acq\_rel")
        * **scope** (*Optional**[**Literal**[**"gpu"**,* *"cta"**,* *"cluster"**,* *"sys"**]**]*) -- Memory scope ("gpu", "cta", "cluster", "sys")

    Returns:
    :   Old value at memory location

    Return type:
    :   Numeric

cutlass.cute.arch.store( : *ptr*, : *val: cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value*, : *\**, : *level1\_eviction\_priority: Literal['evict\_normal', 'evict\_first', 'evict\_last', 'evict\_no\_allocate', 'evict\_unchanged'] | None = None*, : *cop: Literal['wb', 'cg', 'cs', 'wt'] | None = None*, : *ss: Literal['cta', 'cluster'] | None = None*, : *sem: Literal['relaxed', 'release'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *loc=None*, : *ip=None*, ) → None
:   Store a value to a memory location.

    Parameters:
    :   * **ptr** -- Pointer to store to. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **val** (*Union**[**Numeric**,* *ir.Value**]*) -- Value to store (scalar Numeric or vector ir.Value)
        * **level1\_eviction\_priority** -- L1 cache eviction policy string literal:
          "evict\_normal" : .level1::eviction\_priority = .L1::evict\_normal
          "evict\_first" : .level1::eviction\_priority = .L1::evict\_first
          "evict\_last" : .level1::eviction\_priority = .L1::evict\_last
          "evict\_no\_allocate" : .level1::eviction\_priority = .L1::no\_allocate
          "evict\_unchanged" : .level1::eviction\_priority = .L1::evict\_unchanged
        * **cop** -- Store cache modifier string literal:
        * **ss** -- Shared memory space string literal:
          "cta" : .ss = .shared::cta
          "cluster" : .ss = .shared::cluster
          None : .ss = .global
        * **sem** -- Memory semantic string literal:
        * **scope** -- Memory scope string literal:

cutlass.cute.arch.load( : *ptr*, : *dtype: type[cutlass.cute.typing.Numeric] | cutlass.\_mlir.ir.VectorType*, : *\**, : *sem: Literal['relaxed', 'acquire'] | None = None*, : *scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None*, : *level1\_eviction\_priority: Literal['evict\_normal', 'evict\_first', 'evict\_last', 'evict\_no\_allocate', 'evict\_unchanged'] | None = None*, : *cop: Literal['ca', 'cg', 'cs', 'lu', 'cv'] | None = None*, : *ss: Literal['cta', 'cluster'] | None = None*, : *level\_prefetch\_size: Literal['size\_64b', 'size\_128b', 'size\_256b'] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric | cutlass.\_mlir.ir.Value
:   Load a value from a memory location.

    Parameters:
    :   * **ptr** -- Pointer to load from. Supports:
          - ir.Value (LLVM pointer)
          - cute.ptr (\_Pointer instance)
        * **dtype** (*Union**[**type**[**Numeric**]**,* *ir.VectorType**]*) -- Data type to load. Can be:
          - Scalar: Numeric type class (Int8, Uint8, Int32, Float32, etc.)
          - Vector: ir.VectorType for vectorized load (e.g., ir.VectorType.get([4], Int64.mlir\_type))
        * **sem** -- Memory semantic string literal:
        * **scope** -- Memory scope string literal:
        * **level1\_eviction\_priority** -- L1 cache eviction policy string literal:
          "evict\_normal" : .level1::eviction\_priority = .L1::evict\_normal
          "evict\_first" : .level1::eviction\_priority = .L1::evict\_first
          "evict\_last" : .level1::eviction\_priority = .L1::evict\_last
          "evict\_no\_allocate" : .level1::eviction\_priority = .L1::no\_allocate
          "evict\_unchanged" : .level1::eviction\_priority = .L1::evict\_unchanged
        * **cop** -- Load cache modifier string literal:
        * **ss** -- Shared memory space string literal:
          "cta" : .ss = .shared::cta
          "cluster" : .ss = .shared::cluster
          None : .ss = .global
        * **level\_prefetch\_size** -- L2 cache prefetch size hint string literal:
          "size\_64b" : .level::prefetch\_size = .L2::64B
          "size\_128b" : .level::prefetch\_size = .L2::128B
          "size\_256b" : .level::prefetch\_size = .L2::256B

    Returns:
    :   Loaded value (scalar Numeric or vector ir.Value)

    Return type:
    :   Union[Numeric, ir.Value]

cutlass.cute.arch.popc( : *value: cutlass.cute.typing.Numeric*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Numeric
:   Performs a population count operation.

cutlass.cute.arch.fence\_proxy( : *kind: Literal['alias', 'async', 'async.global', 'async.shared', 'tensormap', 'generic']*, : *\**, : *space: Literal['cta', 'cluster'] | None = None*, : *loc=None*, : *ip=None*, ) → None
:   Fence operation to ensure memory consistency between proxies.

    Parameters:
    :   * **kind** (*Literal**[**"alias"**,* *"async"**,* *"async.global"**,* *"async.shared"**,* *"tensormap"**,* *"generic"**]*) -- Proxy kind string literal:
          - "alias" : Alias proxy
          - "async" : Async proxy
          - "async.global" : Async global proxy
          - "async.shared" : Async shared proxy
          - "tensormap" : Tensormap proxy
          - "generic" : Generic proxy
        * **space** (*Optional**[**Literal**[**"cta"**,* *"cluster"**]**]*) -- Shared memory space scope string literal (optional):
          - "cta" : CTA (Cooperative Thread Array) scope
          - "cluster" : Cluster scope

cutlass.cute.arch.warpgroup\_reg\_alloc(*reg\_count: int*, *\**, *loc=None*, *ip=None*) → None

cutlass.cute.arch.warpgroup\_reg\_dealloc(*reg\_count: int*, *\**, *loc=None*, *ip=None*) → None

cutlass.cute.arch.setmaxregister\_increase(*reg\_count: int*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.setmaxregister\_decrease(*reg\_count: int*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.fmax( : *a: float | cutlass.cute.typing.Float32*, : *b: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32

cutlass.cute.arch.rcp\_approx( : *a: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, )

cutlass.cute.arch.exp2( : *a: float | cutlass.cute.typing.Float32*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Float32

cutlass.cute.arch.cvt\_i8x4\_to\_f32x4(*src\_vec4*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_i8x2\_to\_f32x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_i8\_bf16(*src\_i8*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_i8x2\_to\_bf16x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_i8x4\_to\_bf16x4(*src\_vec4*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_f32x2\_bf16x2(*src\_vec2*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.alloc\_smem( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *size\_in\_elems: int*, : *alignment: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer
:   Statically allocates SMEM.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) -- The pointee type of the pointer.
        * **size\_in\_elems** (*int*) -- The size of the allocation in terms of number of elements of the
          pointee type
        * **alignment** (*int*) -- An optional pointer alignment for the allocation

    Returns:
    :   A pointer to the start of the allocation

    Return type:
    :   Pointer

cutlass.cute.arch.get\_dyn\_smem( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *alignment: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer
:   Retrieves a pointer to a dynamic SMEM allocation.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) -- The pointee type of the pointer.
        * **alignment** (*int*) -- An optional pointer alignment, the result pointer is offset appropriately

    Returns:
    :   A pointer to the start of the dynamic SMEM allocation with a correct
        alignement

    Return type:
    :   Pointer

cutlass.cute.arch.get\_dyn\_smem\_size(*\**, *loc=None*, *ip=None*) → int
:   Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time.
    This can be used for bounds checking during shared memory allocation.

    Returns:
    :   The size of dynamic shared memory in bytes

    Return type:
    :   int

cutlass.cute.arch.get\_max\_tmem\_alloc\_cols(*compute\_capability: str*) → int
:   Get the tensor memory capacity in columns for a given compute capability.

    Returns the maximum TMEM capacity in columns available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*str*) -- The compute capability string (e.g. "sm\_100", "sm\_103")

    Returns:
    :   The TMEM capacity in columns

    Return type:
    :   int

    Raises:
    :   **ValueError** -- If the compute capability is not supported

cutlass.cute.arch.get\_min\_tmem\_alloc\_cols(*compute\_capability: str*) → int
:   Get the minimum TMEM allocation columns for a given compute capability.

    Returns the minimum TMEM allocation columns available for the specified
    GPU compute capability.

    Parameters:
    :   **compute\_capability** (*str*) -- The compute capability string (e.g. "sm\_100", "sm\_103")

    Returns:
    :   The minimum TMEM allocation columns

    Return type:
    :   int

    Raises:
    :   **ValueError** -- If the compute capability is not supported

cutlass.cute.arch.retrieve\_tmem\_ptr( : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *alignment: int*, : *ptr\_to\_buffer\_holding\_addr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer
:   Retrieves a pointer to TMEM with the provided element type and alignment.

    Parameters:
    :   * **element\_type** (*Type**[**Numeric**]*) -- The pointee type of the pointer.
        * **alignment** (*int*) -- The alignment of the result pointer
        * **ptr\_to\_buffer\_holding\_addr** (*Pointer*) -- A pointer to a SMEM buffer holding the TMEM address of the
          start of the allocation allocation

    Returns:
    :   A pointer to TMEM

    Return type:
    :   Pointer

cutlass.cute.arch.alloc\_tmem( : *num\_columns: cutlass.cute.typing.Int*, : *smem\_ptr\_to\_write\_address: cutlass.cute.typing.Pointer*, : *is\_two\_cta=None*, : *\**, : *arch: str = 'sm\_100'*, : *loc=None*, : *ip=None*, ) → None
:   Allocates TMEM.

    Parameters:
    :   * **num\_columns** (*Int*) -- The number of TMEM columns to allocate
        * **smem\_ptr\_to\_write\_address** (*Pointer*) -- A pointer to a SMEM buffer where the TMEM address is written
          to
        * **is\_two\_cta** -- Optional boolean parameter for 2-CTA MMAs
        * **arch** (*str*) -- The architecture of the GPU.

cutlass.cute.arch.relinquish\_tmem\_alloc\_permit( : *is\_two\_cta=None*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Relinquishes the right to allocate TMEM so that other CTAs potentially in a different grid can
    allocate.

cutlass.cute.arch.dealloc\_tmem( : *tmem\_ptr: cutlass.cute.typing.Pointer*, : *num\_columns: cutlass.cute.typing.Int*, : *is\_two\_cta=None*, : *\**, : *arch: str = 'sm\_100'*, : *loc=None*, : *ip=None*, ) → None
:   Deallocates TMEM using the provided pointer and number of columns.

    Parameters:
    :   * **tmem\_ptr** (*Pointer*) -- A pointer to the TMEM allocation to de-allocate
        * **num\_columns** (*Int*) -- The number of columns in the TMEM allocation
        * **is\_two\_cta** -- Optional boolean parameter for 2-CTA MMAs

cutlass.cute.arch.prmt(*src*, *src\_reg\_shifted*, *prmt\_indices*, *\**, *loc=None*, *ip=None*)

cutlass.cute.arch.cvt\_i8\_bf16\_intrinsic(*vec\_i8*, *length*, *\**, *loc=None*, *ip=None*)
:   Fast conversion from int8 to bfloat16. It converts a vector of int8 to a vector of bfloat16.

    Parameters:
    :   * **vec\_i8** (*1D vector* *of* *int8*) -- The input vector of int8.
        * **length** (*int*) -- The length of the input vector.

    Returns:
    :   The output 1D vector of bfloat16 with the same length as the input vector.

    Return type:
    :   1D vector of bfloat16

cutlass.cute.arch.cvt\_i4\_bf16\_intrinsic( : *vec\_i4*, : *length*, : *\**, : *with\_shuffle=False*, : *loc=None*, : *ip=None*, )
:   Fast conversion from int4 to bfloat16. It converts a vector of int4 to a vector of bfloat16.

    Parameters:
    :   * **vec\_i4** (*1D vector* *of* *int4*) -- The input vector of int4.
        * **length** (*int*) -- The length of the input vector.
        * **with\_shuffle** (*bool*) -- Whether the input vec\_i4 follows a specific shuffle pattern.
          If True, for consecutive 8 int4 values with indices of (0, 1, 2, 3, 4, 5, 6, 7),
          the input elements are shuffled to (0, 2, 1, 3, 4, 6, 5, 7). For tailing elements less than 8,
          the shuffle pattern is (0, 2, 1, 3) for 4 elements. No shuffle is needed for less than 4 elements.
          Shuffle could help to produce converted bf16 values in the natural order of (0, 1, 2 ,3 ,4 ,5 ,6 ,7)
          without extra prmt instructions and thus better performance.

    Returns:
    :   The output 1D vector of bfloat16 with the same length as the input vector.

    Return type:
    :   1D vector of bfloat16

cutlass.cute.arch.issue\_clc\_query( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *clc\_response\_ptr: cutlass.cute.typing.Pointer*, : *loc=None*, : *ip=None*, ) → None
:   The clusterlaunchcontrol.try\_cancel instruction requests atomically cancelling the launch
    of a cluster that has not started running yet. It asynchronously writes an opaque response
    to shared memory indicating whether the operation succeeded or failed. On success, the
    opaque response contains the ctaid of the first CTA of the canceled cluster.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier address in SMEM
        * **clc\_response\_ptr** (*Pointer*) -- A pointer to the cluster launch control response address in SMEM

cutlass.cute.arch.clc\_response( : *result\_addr: cutlass.cute.typing.Pointer*, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32, cutlass.cute.typing.Int32]
:   After loading response from clusterlaunchcontrol.try\_cancel instruction into 16-byte
    register, it can be further queried using clusterlaunchcontrol.query\_cancel instruction.
    If the cluster is canceled successfully, predicate p is set to true; otherwise, it is
    set to false. If the request succeeded, clusterlaunchcontrol.query\_cancel.get\_first\_ctaid
    extracts the CTA id of the first CTA in the canceled cluster. By default, the instruction
    returns a .v4 vector whose first three elements are the x, y and z coordinate of first CTA
    in canceled cluster.

    Parameters:
    :   **result\_addr** (*Pointer*) -- A pointer to the cluster launch control response address in SMEM
