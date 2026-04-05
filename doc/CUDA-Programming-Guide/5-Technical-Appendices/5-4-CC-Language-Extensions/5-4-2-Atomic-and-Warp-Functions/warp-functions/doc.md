
The following section describes the warp functions that allow threads within a warp to communicate with each other and perform computations.

Hint

It is suggested to use the `CUB` [Warp-Wide "Collective" Primitives](https://nvidia.github.io/cccl/cub/api_docs/warp_wide.html#warp-wide-collective-primitives) to perform warp operations whenever possible for efficiency, safety, and portability reasons.

### 5.4.6.1. Warp Active Mask

```
unsigned __activemask();
```

The function returns a 32-bit integer mask representing all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when `__activemask()` is called. [Inactive threads](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes) are represented by 0 bits in the returned mask. Threads that have exited the program are always marked as inactive.

Warning

`__activemask()` cannot be used to determine which warp lanes execute a given branch. This function is intended for opportunistic warp-level programming and only provides an instantaneous snapshot of the active threads within a warp.

```
// Check whether at least one thread's predicate evaluates to true
if (pred) {
    // Invalid: the value of 'at_least_one' is non-deterministic
    // and could vary between executions.
    at_least_one = __activemask() > 0;
}
```

Note that threads convergent at an `__activemask()` call are not guaranteed to remain convergent at subsequent instructions unless those instructions are warp synchronizing intrinsics (`__sync`).

For example, the compiler could reorder instructions, and the set of active threads might not be preserved:

```
unsigned mask      = __activemask();              // Assume mask == 0xFFFFFFFF (all bits set, all threads active)
int      predicate = threadIdx.x % 2 == 0;        // 1 for even threads, 0 for odd threads
int      result    = __any_sync(mask, predicate); // Active threads might not be preserved
```

### 5.4.6.2. Warp Vote Functions

```
int      __all_sync   (unsigned mask, int predicate);
int      __any_sync   (unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
```

The warp vote functions enable the threads of a given [warp](../01-introduction/programming-model.html#programming-model-warps-simt) to perform a reduction-and-broadcast operation. These functions take an integer `predicate` as input from each non-exited thread in the warp and compare those values with zero. The results of the comparisons are then combined (reduced) across the [active threads](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes) of the warp in one of the following ways, broadcasting a single return value to each participating thread:

`__all_sync(unsigned mask, predicate)`:
:   Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for all of them.

`__any_sync(unsigned mask, predicate)`:
:   Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for one or more of them.

`__ballot_sync(unsigned mask, predicate)`:
:   Evaluates `predicate` for all non-exited threads in `mask` and returns an integer whose Nth bit is set if `predicate` evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. Otherwise, the Nth bit is zero.

The functions are subject to the [Warp \_\_sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.3. Warp Match Functions

Hint

It is suggested to use the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/warp/warp_match_all.html) `cuda::device::warp_match_all()` function as a generalized and safer alternative to `__match_all_sync` function.

```
unsigned __match_any_sync(unsigned mask, T value);
unsigned __match_all_sync(unsigned mask, T value, int *pred);
```

The warp match functions perform a broadcast-and-compare operation of a variable between non-exited threads within a [warp](../01-introduction/programming-model.html#programming-model-warps-simt).

`__match_any_sync`
:   Returns the mask of non-exited threads that have the same bitwise `value` in `mask`.

`__match_all_sync`
:   Returns `mask` if all non-exited threads in `mask` have the same bitwise `value`; otherwise 0 is returned. Predicate `pred` is set to `true` if all non-exited threads in `mask` have the same bitwise `value`; otherwise the predicate is set to false.

`T` can be `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.

The functions are subject to the [Warp \_\_sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.4. Warp Reduce Functions

Hint

It is suggested to use the `CUB` [Warp-Wide "Collective" Primitives](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html#_CPPv4I0_iEN3cub10WarpReduceE) to perform a Warp Reduction whenever possible for efficiency, safety, and portability reasons.

Supported by devices of compute capability 8.x or higher.

```
T        __reduce_add_sync(unsigned mask, T value);
T        __reduce_min_sync(unsigned mask, T value);
T        __reduce_max_sync(unsigned mask, T value);

unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync (unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);
```

The `__reduce_<op>_sync` intrinsics perform a reduction operation on the data provided in `value` after synchronizing all non-exited threads named in `mask`.

`__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`
:   Returns the result of applying an arithmetic add, min, or max reduction operation on the values provided in `value` by each non-exited thread named in `mask`. `T` can be an `unsigned` or `signed` integer.

`__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`
:   Returns the result of applying a bitwise AND, OR, or XOR reduction operation on the values provided in `value` by each non-exited thread named in `mask`.

The functions are subject to the [Warp \_\_sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.5. Warp Shuffle Functions

Hint

It is suggested to use the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/warp/warp_shuffle.html#libcudacxx-extended-api-warp-warp-shuffle) `cuda::device::warp_shuffle()` functions as a generalized and safer alternative to `__shfl_sync()` and `__shfl_<op>_sync()` intrinsics.

```
T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);
```

Warp shuffle functions exchange a value between non-exited threads within a [warp](../01-introduction/programming-model.html#programming-model-warps-simt) without the use of shared memory.

`__shfl_sync()`: Direct copy from indexed lane.
:   The intrinsic function returns the value of `value` held by the thread whose ID is given by `srcLane`.

    * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.
    * If `srcLane` is outside the range `[0, width - 1]`, the result corresponds to the value held by the `srcLane % width`, which is within the same subsection.

---

`__shfl_up_sync()`: Copy from a lane with a lower ID than the caller's.
:   The intrinsic function calculates a source lane ID by subtracting `delta` from the caller's lane ID. The value of `value` held by the resulting lane ID is returned: in effect, `value` is shifted up the warp by `delta` lanes.

    * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.
    * The source lane index will not wrap around the value of `width`, so the lower `delta` lanes will remain unchanged.

---

`__shfl_down_sync()`: Copy from a lane with a higher ID than the caller's.
:   The intrinsic function calculates a source lane ID by adding `delta` to the caller's lane ID. The value of `value` held by the resulting lane ID is returned: this has the effect of shifting `value` down the warp by `delta` lanes.

    * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.
    * As for `__shfl_up_sync()`, the ID number of the source lane will not wrap around the value of width and so the upper `delta` lanes will effectively remain unchanged.

---

`__shfl_xor_sync()`: Copy from a lane based on bitwise XOR of own lane ID.
:   The intrinsic function calculates a source lane ID by performing a bitwise XOR of the caller's lane ID and `laneMask`: the value of `value` held by the resulting lane ID is returned. This mode implements a butterfly addressing pattern, which is used in tree reduction and broadcast.

    * If `width` is less than `warpSize`, then each group of `width` consecutive threads are able to access elements from earlier groups. However, if they attempt to access elements from later groups of threads their own value of `value` will be returned.

---

`T` can be:

* `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.
* `__half` and `__half2` with the `cuda_fp16.h` header included.
* `__nv_bfloat16` and `__nv_bfloat162` with the `cuda_bf16.h` header included.

Threads may only read data from another thread that is actively participating in the intrinsics. If the target thread is [inactive](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes), the retrieved value is undefined.

`width` must be a power of two in the range `[1, warpSize]`, namely 1, 2, 4, 8, 16, or 32. Other values will produce undefined results.

The functions are subject to the [Warp \_\_sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Examples of valid warp shuffle usage:

```
int laneId = threadIdx.x % warpSize;
int data   = ...

// all warp threads get 'data' from lane 0
int result1 = __shfl_sync(0xFFFFFFFF, data, 0);

if (laneId < 4) {
    // lanes 0, 1, 2, 3 get 'data' from lane 1
    int result2 = __shfl_sync(0xb1111, data, 1);
}

// lanes [0 - 15] get 'data' from lane 0
// lanes [16 - 31] get 'data' from lane 16
int result3 = __shfl_sync(0xFFFFFFFF, value, warpSize / 2);

// each lane gets 'data' from the lane two positions above
// lanes 30, 31 get their original value
int result4 = __shfl_down_sync(0xFFFFFFFF, data, 2);
```

Examples of invalid warp shuffle usage:

```
int laneId = threadIdx.x % warpSize;
int value  = ...
 // undefined behavior: lane 0 does not participate in the call
int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;

if (laneId <= 4) {
    // undefined behavior: destination lanes 5, 6 are not active for lanes 3, 4
    result = __shfl_down_sync(0b11111, value, 2);
}

// undefined behavior: width is not a power of 2
__shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);
```

Warning

These intrinsics do not imply a memory barrier. They do not guarantee any memory ordering.

Example 1: Broadcast of a single value across a warp

CUDA C++

```
#include <cassert>
#include <cuda/warp>

__global__ void warp_broadcast_kernel(int input) {
    int laneId = threadIdx.x % 32;
    int value;
    if (laneId == 0) { // unused variable for all threads except lane 0
        value = input;
    }
    value = cuda::device::warp_shuffle_idx(value, 0); // Synchronize all threads in warp, and get "value" from lane 0
    assert(value == input);
}

int main() {
    warp_broadcast_kernel<<<1, 32>>>(1234);
    cudaDeviceSynchronize();
    return 0;
}
```

Intrinsics

```
#include <assert.h>

__global__ void warp_broadcast_kernel(int input) {
    int laneId = threadIdx.x % 32;
    int value;
    if (laneId == 0) { // unused variable for all threads except lane 0
        value = input;
    }
    value = __shfl_sync(0xFFFFFFFF, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
    assert(value == input);
}

int main() {
    warp_broadcast_kernel<<<1, 32>>>(1234);
    cudaDeviceSynchronize();
    return 0;
}
```

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/E3E3Y5e4e).

Example 2: Inclusive plus-scan across sub-partitions of 8 threads

Hint

It is suggested to use the [cub::WarpScan](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html) function for efficient and generalized warp scan functions.

CUDA C++

```
#include <cstdio>
#include <cub/cub.cuh>

__global__ void scan_sub_partition_with_8_threads_kernel() {
    using WarpScan    = cub::WarpScan<int, 8>;
    using TempStorage = typename WarpScan::TempStorage;
    __shared__ TempStorage temp_storage;

    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // starting value to accumulate
    int partial_sum;
    WarpScan(temp_storage).InclusiveSum(value, partial_sum);
    printf("Thread %d final value = %d\n", threadIdx.x, partial_sum);
}

int main() {
    scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Intrinsics

```
#include <stdio.h>

__global__ void scan_sub_partition_with_8_threads_kernel() {
    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // starting value to accumulate
    // Loop to accumulate scan within my partition.
    // Scan requires log2(8) == 3 steps for 8 threads
    for (int delta = 1; delta <= 4; delta *= 2) {
        int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // read from laneId - delta
        int source_lane = laneId % 8 - delta;
        if (source_lane >= 0) // lanes with 'source_lane < 0' have their value unchanged
            value += tmp;
    }
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/Tohd38edc).

Example 3: Reduction across a warp

Hint

It is suggested to use the [cub::WarpReduce](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html) function for efficient and generalized warp reduction functions.

CUDA C++

```
#include <cstdio>
#include <cub/cub.cuh>
#include <cuda/warp>

__global__ void warp_reduce_kernel() {
    using WarpReduce  = cub::WarpReduce<int>;
    using TempStorage = typename WarpReduce::TempStorage;
    __shared__ TempStorage temp_storage;

    int laneId     = threadIdx.x % 32;
    int value      = 31 - laneId; // starting value to accumulate
    auto aggregate = WarpReduce(temp_storage).Sum(value);
    aggregate      = cuda::device::warp_shuffle_idx(aggregate, 0);
    printf("Thread %d final value = %d\n", threadIdx.x, aggregate);
}

int main() {
    warp_reduce_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Intrinsics

```
#include <stdio.h>

__global__ void warp_reduce_kernel() {
    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // starting value to accumulate
    // Use XOR mode to perform butterfly reduction
    // A full-warp reduction requires log2(32) == 5 steps
    for (int i = 1; i <= 16; i *= 2)
        value += __shfl_xor_sync(0xFFFFFFFF, value, i);
    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warp_reduce_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/T94nfGMzG).

### 5.4.6.6. Warp `__sync` Intrinsic Constraints

All warp `__sync` intrinsics, such as:

* `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
* `__match_any_sync`, `__match_all_sync`
* `__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`, `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`
* `__syncwarp`

use the `mask` parameter to indicate which warp threads participate in the call. This parameter ensures proper convergence before the hardware executes the intrinsic.

Each bit in the `mask` corresponds to a thread's lane ID (`threadIdx.x % warpSize`). The intrinsic waits until all non-exited warp threads specified in the `mask` reach the call.

The following constraints must be met for correct execution:

* Each calling thread must have its corresponding bit set in the `mask`.
* Each non-calling thread must have its corresponding bit set to zero in the `mask`. Exited threads are ignored.
* All non-exited threads specified in the `mask` must execute the intrinsic with the same `mask` value.
* Warp threads may call the intrinsic concurrently with different `mask` values, provided the masks are disjoint. Such condition is valid even in divergent control flow.

The behavior of warp `__sync` functions is invalid, such as kernel hang, or undefined if:

* A calling thread is not specified in the `mask`.
* A non-exited thread specified in the `mask` fails to either eventually exit or call the intrinsic at the same program point with the same `mask` value.
* In conditional code, all conditions must evaluate identically across all non-exited threads specified in the `mask`.

Note

The intrinsics achieve the best efficiency when all warp threads participate in the call, namely when the `mask` is set to `0xFFFFFFFF`.

Examples of valid warp intrinsics usage:

```
__global__ void valid_examples() {
    if (threadIdx.x < 4) {        // threads 0, 1, 2, 3 are active
        __all_sync(0b1111, pred); // CORRECT, threads 0, 1, 2, 3 participate in the call
    }

    if (threadIdx.x == 0)
        return; // exit
    // CORRECT, all non-exited threads participate in the call
    __all_sync(0xFFFFFFFF, pred);
}
```

Disjoint `mask` examples:

```
__global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // CORRECT
        __syncwarp(mask);
        if (threadIdx.x == 0 || threadIdx.x == 16)
            output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
    }
}
```

```
__global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        if (threadIdx.x < 16) {
            unsigned mask = 0xFFFF; // CORRECT
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
        }
        else {
            unsigned mask = 0xFFFF0000; // CORRECT
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
        }
    }
}
```

Examples of invalid warp intrinsics usage:

```
if (threadIdx.x < 4) {           // threads 0, 1, 2, 3 are active
    __all_sync(0b0000011, pred); // WRONG, threads 2, 3 are active but not set in mask
    __all_sync(0b1111111, pred); // WRONG, threads 4, 5, 6 are not active but set in mask
}

// WRONG, participating threads have a different and overlapping mask
__all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);
```

