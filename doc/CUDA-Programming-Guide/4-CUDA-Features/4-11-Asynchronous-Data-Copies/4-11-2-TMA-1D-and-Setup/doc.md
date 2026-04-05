## 4.11.2. Using the Tensor Memory Accelerator (TMA)

Many applications need to move large amounts of data to and from global memory. Often, the data is laid out in global memory as a multi-dimensional array with non-sequential data access patterns. To reduce global memory accesses, sub-tiles of such arrays are copied to shared memory before use in computations. The loading and storing involves address-calculations that can be error-prone and repetitive. To offload these computations, compute capability 9.0 (Hopper) and later (see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)) have a *tensor memory accelerator* (TMA). The primary goal of the TMA is to provide an efficient data transfer mechanism from global memory to shared memory for multi-dimensional arrays.

**Naming**. Tensor memory accelerator (TMA) is a broad term used to refer to the features described in this section. For the purpose of forward-compatibility and to reduce discrepancies with the PTX ISA, the text in this section refers to TMA operations as either *bulk-asynchronous copies* or *bulk-tensor asynchronous copies*, depending on the specific type of copy used. The term “bulk” is used to contrast these operations with the asynchronous memory operations described in the previous section.

**Dimensions**. TMA supports copying both one-dimensional and multi-dimensional arrays (up to 5-dimensional). The programming model for bulk-asynchronous copies of one-dimensional contiguous arrays is different from the programming model for bulk-tensor asynchronous copies of multi-dimensional arrays. To perform a bulk-tensor asynchronous copy of a multi-dimensional array, the hardware requires a [tensor map](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap). This object describes the layout of the multi-dimensional array in global and shared memory. A tensor map is typically created on the host using the [cuTensorMapEncode API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY) and then transferred from host to device as a `const` kernel parameter annotated with `__grid_constant__` (see [\_\_grid\_constant\_\_ Parameters](../05-appendices/cpp-language-extensions.html#grid-constant)). The tensor map is transferred from host to device as a `const` kernel parameter annotated with `__grid_constant__`, and can be used on the device to copy a tile of data between shared and global memory. In contrast, performing a bulk-asynchronous copy of a contiguous one-dimensional array does not require a tensor map: it can be performed on-device with a pointer and size parameter.

**Source and destination**. The source and destination addresses of TMA operations can be in shared or global memory. The operations can read data from global to shared memory, write data from shared to global memory, and also copy from shared memory to [distributed shared memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory) of another block in the same cluster. In addition, when in a cluster, a bulk-asynchronous tensor operation can be specified as being *multicast*. In this case, data can be transferred from global memory to the shared memory of multiple blocks within the cluster. The multicast feature is optimized for target architecture `sm_90a` and may have [significantly reduced performance](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) on other targets. Hence, it is advised to be used with [compute architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) `sm_90a`.

**Asynchronicity**. Data transfers using TMA are [asynchronous](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async proxy operations (see [Async Thread and Async Proxy](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. *Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future*. There are several [completion mechanisms](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms) that bulk-asynchronous operations can use to signal that they have completed. When the operation reads from global to shared memory, any thread in the block can wait for the data to be readable in shared memory by waiting on a [shared memory barrier](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers). When the bulk-asynchronous operation writes data from shared memory to global or distributed shared memory, only the initiating thread can wait for the operation to have completed. This is accomplished using a *bulk async-group* based completion mechanism. A table describing the completion mechanisms can be found below and in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).

Table 19 Asynchronous copies with possible source and destination memory spaces and completion mechanisms using TMA. An empty cell indicates that a source-destination pair is not supported.

| Direction | | Asynchronous Copy (TMA, CC 9.0+) |
| --- | --- | --- |
| Source | Destination | Completion Mechanism |
| global | global |  |
| shared::cta | global | bulk async-group |
| global | shared::cta | shared memory barrier |
| global | shared::cluster | shared memory barrier (multicast) |
| shared::cta | shared::cluster | shared memory barrier |
| shared::cta | shared::cta |  |

### 4.11.2.1. Using TMA to transfer one-dimensional arrays

The following table summarizes the possible source and destination memory spaces and completion mechanisms for bulk-asynchronous TMA along with the API that exposes it.

Table 20 Asynchronous copies with possible source and destination memory spaces and completion mechanisms using bulk-asynchronous TMA. An empty cell indicates that a source-destination pair is not supported.

| Direction | | Bulk-Asynchronous Copy (TMA, CC9.0+) | |
| --- | --- | --- | --- |
| Source | Destination | Completion Mechanism | API |
| global | global |  |  |
| shared::cta | global | bulk async-group | [cuda::ptx::cp\_async\_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| global | shared::cta | shared memory barrier | [cuda::memcpy\_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html), [cuda::device::memcpy\_async\_tx](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async_tx.html), [cuda::ptx::cp\_async\_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| global | shared::cluster | shared memory barrier | [cuda::ptx::cp\_async\_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| shared::cta | shared::cluster | shared memory barrier | [cuda::ptx::cp\_async\_bulk](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html) |
| shared::cta | shared::cta |  |  |

Some functionality requires inline PTX that is currently made available through the `cuda::ptx` namespace in the [CUDA Standard C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) library. The availability of these wrappers can be checked with the following code:

```
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

Note that `cuda::memcpy_async` uses TMA if the source and destination addresses are 16-byte aligned and the size is a multiple of 16 bytes, otherwise it falls back to synchronous copies. On the other hand, `cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` always use TMA and will result in undefined behavior if the requirements are not met.

In the following, we demonstrate how to use bulk-asynchronous copies through an example. The example read-modify-writes a one-dimensional array. The kernel goes through the following steps:

1. Initialize a shared memory barrier as a completion mechanism for the bulk-asynchronous copy from global to shared memory.
2. Initiate the copy of a block of memory from global to shared memory.
3. Arrive and wait on the shared memory barrier for completion of the copy.
4. Increment the shared memory buffer values.
5. Use a proxy fence to ensure shared memory writes (generic proxy) become visible to the subsequent bulk-asynchronous copy (async proxy).
6. Initiate a bulk-asynchronous copy of the buffer in shared memory to global memory.
7. Wait for the bulk-asynchronous copy to have finished reading shared memory.

```
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operation should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory from a single thread.
  if (is_elected()) {
    // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).

    // Version 1: cuda::memcpy_async
    cuda::memcpy_async(
        smem_data, data + offset,
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar);

    // Version 2: cuda::device::memcpy_async_tx
    // cuda::device::memcpy_async_tx(
    //   smem_data, data + offset,
    //   cuda::aligned_size_t<16>(sizeof(smem_data)),
    //   bar);
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));

    // Version 3: cuda::ptx::cp_async_bulk
    // ptx::cp_async_bulk(
    //     ptx::space_shared, ptx::space_global,
    //     smem_data, data + offset,
    //     sizeof(smem_data),
    //     cuda::device::barrier_native_handle(bar));
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));
  }

  // 3a. All threads arrive on the barrier.
  barrier::arrival_token token = bar.arrive();

  // 3b. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory.
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory.
  if (is_elected()) {
    ptx::cp_async_bulk(
        ptx::space_global, ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

**Barrier initialization**. The barrier is initialized with the number of threads participating in the block. As a result, the barrier will flip only if
all threads have arrived on this barrier. Shared memory barriers are described
in more detail in [shared memory barriers](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers).

**TMA read**. The bulk-asynchronous copy instruction directs the
hardware to copy a large chunk of data into shared memory, and to update the
[transaction count](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)
of the shared memory barrier after completing the read. In general, issuing as
few bulk copies with as big a size as possible results in the best performance.
Because the copy can be performed asynchronously by the hardware, it is not
necessary to split the copy into smaller chunks.

The thread that initiates the bulk-asynchronous copy operation also tells the barrier how many transactions (tx) are expected to arrive.
In this case, the transactions are counted in bytes. This is automatically performed by `cuda::memcpy_async`, but not by
`cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` after which we need to explicitly call `cuda::ptx::mbarrier_expect_tx`.
If multiple threads update the transaction count, the expected transaction will be the sum
of the updates. The barrier will only flip once all threads have arrived **and**
all bytes have arrived. Once the barrier has flipped, the bytes are safe to read
from shared memory, both by the threads as well as by subsequent
bulk-asynchronous copies. More information about barrier transaction accounting
can be found in [Tracking Asynchronous Memory Operations](async-barriers.html#asynchronous-barriers-tracking).

**Barrier wait**. Waiting for the barrier to flip is done using tokens with `bar.wait()`. It can be more efficient to use explicit phase tracking of the barrier (see [Explicit Phase Tracking](async-barriers.html#asynchronous-barriers-explicit-phase)).

**SMEM write and sync**. The increment of the buffer values reads and writes to shared
memory. To make the writes visible to subsequent bulk-asynchronous copies, the
`cuda::ptx::fence_proxy_async` function is used. This orders the writes to
shared memory before subsequent reads from bulk-asynchronous copy operations,
which read through the async proxy. So each thread first orders the writes to
objects in shared memory in the async proxy via the
`cuda::ptx::fence_proxy_async`, and these operations by all threads are
ordered before the async operation performed in thread 0 using
`__syncthreads()`.

**TMA write and sync**. The write from shared to global memory is again
initiated by a single thread. The completion of the write is not tracked by a
shared memory barrier. Instead, a thread-local mechanism is used. Multiple
writes can be batched into a so-called *bulk async-group*. Afterwards, the
thread can wait for all operations in this group to have completed reading from
shared memory (as in the code above) or to have completed writing to global
memory, making the writes visible to the initiating thread. For more information,
refer to the PTX ISA documentation of [cp.async.bulk.wait\_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
Note that the bulk-asynchronous and non-bulk-asynchronous copy instructions have
different async-groups: there exist both `cp.async.wait_group` and
`cp.async.bulk.wait_group` instructions.

Note

It is recommended to initiate TMA operations by a single thread in the block.
While using `if (threadIdx.x == 0)` might seem sufficient, the compiler cannot
verify that indeed only one thread is initiating the copy and may insert a peeling
loop over all active threads, which results in warp serialization and reduced
performance. To prevent this, we define the `is_elected()` helper function that
uses `cuda::ptx::elect_sync` to select one thread from warp 0 – which is known to
the compiler – to execute the copy allowing it to generate more efficient code.
Alternatively, the same effect can be achieved with [cooperative\_groups::invoke\_one](cooperative-groups.html#cooperative-groups-invoke-one).

The bulk-asynchronous instructions have specific alignment requirements on their source and
destination addresses. More information can be found in the table below.

Table 21 Alignment requirements for one-dimensional bulk-asynchronous operations.

| Address / Size | Alignment |
| --- | --- |
| Global memory address | Must be 16 byte aligned. |
| Shared memory address | Must be 16 byte aligned. |
| Shared memory barrier address | Must be 8 byte aligned (this is guaranteed by `cuda::barrier`). |
| Size of transfer | Must be a multiple of 16 bytes. |

#### 4.11.2.1.1. Prefetching Data

In this example, we will demonstrate how to use TMA to prefetch data from global memory to shared memory. In an iterative copy and compute pattern, this allows hiding the latency of data transfers of future iterations with computation on the current iteration, potentially increasing bytes-in-flight.

CUDA C++ `cuda::device::memcpy_async_tx`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda/barrier> #include <cuda/ptx>  namespace ptx = cuda::ptx; namespace cg = cooperative_groups;  __device__ inline bool is_elected() {     unsigned int tid = threadIdx.x;     unsigned int warp_id = tid / 32;     unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.     return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0. }  template <int block_size, int num_stages> __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {     auto grid = cg::this_grid();     auto block = cg::this_thread_block();     const int tid = threadIdx.x;     assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size      // 1. Initialization Phase     __shared__ int shared[num_stages * block_size];     size_t shared_offset[num_stages];     for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();      auto block_batch = [&](size_t batch) -> int {         return block.group_index().x * block.size() + grid.size() * batch;     };      // Initialize shared memory barrier with the number of threads participating in the barrier.     // We will use explicit phase tracking for the barrier, which allows us to have only one      // thread arrive on the barrier to set the transaction count and other threads wait for      // a parity-based phase flip.     #pragma nv_diag_suppress static_var_with_dynamic_init     __shared__ cuda::barrier<cuda::thread_scope_block> bar[num_stages];     if (tid == 0) {         #pragma unroll num_stages         for (int i = 0; i < num_stages; i++) {             init(&bar[i], 1);         }     }     __syncthreads();      // Fill the pipeline with the first ``num_stages`` batches.     if (is_elected()) {         size_t num_bytes = block_size * sizeof(int);          #pragma unroll num_stages         for (int s = 0; s < num_stages; ++s) {             cuda::device::memcpy_async_tx(&shared[shared_offset[s]], &global_in[block_batch(s)], cuda::aligned_size_t<16>(num_bytes), bar[s]);             (void)cuda::device::barrier_arrive_tx(bar[s], 1, num_bytes);         }     }      // 2. Main Processing Loop.     // compute_batch: next batch to process.     // fetch_batch:   next batch to fetch from global memory.     int stage = 0;       // current stage in the shared memory buffer.     uint32_t parity = 0; // barrierparity     for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; ++compute_batch, ++fetch_batch) {         // (a) Wait on current batch.         while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, cuda::device::barrier_native_handle(bar[stage]), parity)) {}          // (b) Compute on the current batch.         compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);         __syncthreads();          // (c) Load next stage ``num_stages`` ahead of current compute batch.         if (is_elected() && fetch_batch < batch_size) {             size_t num_bytes = block_size * sizeof(int);             cuda::device::memcpy_async_tx(&shared[shared_offset[stage]], &global_in[block_batch(fetch_batch)], cuda::aligned_size_t<16>(num_bytes), bar[stage]);             (void)cuda::device::barrier_arrive_tx(bar[stage], 1, num_bytes);         }          // (d) Stage management.         stage++;         if (stage == num_stages) {             stage = 0;             parity ^= 1;         }     } } ``` |

This example implements *multi-stage data prefetching* using `cuda::device::memcpy_async_tx` for the TMA copies and employs shared memory barriers with explicit phase tracking for synchronization of the copies.

1. **Initialization Phase**: Sets up shared memory barriers (one per stage) and pre-loads the first `num_stages` batches into different shared memory sections.
2. **Main Processing Loop**:

   1. **Wait**: Uses `mbarrier_try_wait_parity()` to wait for the current batch to complete copying.
   2. **Compute**: Processes the current batch data.
   3. **Prefetch**: Schedules the next `memcpy_async_tx` operation for future data (staying `num_stages` ahead).
   4. **Stage Management**: Cycles through stages using a rotating buffer approach and tracks barrier parity.

### 4.11.2.2. Using TMA to transfer multi-dimensional arrays

In this section, we will focus on multi-dimensional TMA copies.
The primary difference between the one-dimensional and multi-dimensional case is
that a tensor map must be created on the host and passed to the CUDA kernel.

The following table summarizes the possible source and destination memory spaces and completion mechanisms for bulk-tensor asynchronous TMA along with the API that exposes it in device code.

Table 22 Asynchronous copies with possible source and destination memory spaces and completion mechanisms using bulk-tensor asynchronous TMA. An empty cell indicates that a source-destination pair is not supported.

| Direction | | Bulk-Tensor Asynchronous Copy (TMA, CC9.0+) | |
| --- | --- | --- | --- |
| Source | Destination | Completion Mechanism | API |
| global | global |  |  |
| shared::cta | global | bulk async-group | [cuda::ptx::cp\_async\_bulk\_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| global | shared::cta | shared memory barrier | [cuda::ptx::cp\_async\_bulk\_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| global | shared::cluster | shared memory barrier | [cuda::ptx::cp\_async\_bulk\_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| shared::cta | shared::cluster | shared memory barrier | [cuda::ptx::cp\_async\_bulk\_tensor](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html) |
| shared::cta | shared::cta |  |  |

All functionality requires inline PTX that is currently made available through the `cuda::ptx` namespace in the [CUDA Standard C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) library.

In the following, we describe how to create a tensor map using the CUDA driver API, how
to pass it to the device, and how to use it on the device.

**Driver API**. A tensor map is created using the [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
driver API. This API can be accessed by linking to the driver directly
(`-lcuda`) or by using the [cudaGetDriverEntryPointByVersion](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html)
API. Below, we show how to get a pointer to the `cuTensorMapEncodeTiled` API.
For more information, refer to [Driver Entry Point Access](driver-entry-point-access.html#driver-entry-point-access).

```
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
```

**Creation**. Creating a tensor map requires many parameters. Among
them are the base pointer to an array in global memory, the size of the array
(in number of elements), the stride from one row to the next (in bytes), the
size of the shared memory buffer (in number of elements). The code below creates
a tensor map to describe a two-dimensional row-major array of size `GMEM_HEIGHT
x GMEM_WIDTH`. Note the order of the parameters: the fastest moving dimension
comes first.

```
  CUtensorMap tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
```

**Host-to-device transfer**. There are three ways to make a tensor map accessible to
device code. The recommended approach is to pass the tensor map as a const `__grid_constant__`
parameter to a kernel. The other possibilities are copying the tensor map into device `__constant__`
memory using `cudaMemcpyToSymbol` or accessing it via global memory. When passing the tensor map as a parameter, some versions of the
GCC C++ compiler issue the warning “the ABI for passing parameters with 64-byte
alignment has changed in GCC 4.6”. This warning can be ignored.

```
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // Use tensor_map here.
}
int main() {
  CUtensorMap map;
  // [ ..Initialize map.. ]
  kernel<<<1, 1>>>(map);
}
```

As an alternative to the `__grid_constant__` kernel parameter, a global
`__constant__` variable can be used. An example is included
below.

```
#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // Use global_tensor_map here.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}
```

Finally, it is possible to copy the tensor map to global memory. Using a pointer to a
tensor map in global device memory requires a fence in each thread block before any thread
in the block uses the updated tensor map. Further uses of the tensor map by that thread block
do not need to be fenced unless the tensor map is modified again. Note that this mechanism
may be slower than the two mechanisms described above.

```
#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  // Since the tensor map was modified from the host using cudaMemcpy,
  // the scope should be .sys.
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
 );
 // Safe to use tensor_map after fence inside this thread.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(global_tensor_map);
}
```

**Use**. The kernel below loads a 2D tile of size `SMEM_HEIGHT x SMEM_WIDTH`
from a larger 2D array. The top-left corner of the tile is indicated by the
indices `x` and `y`. The tile is loaded into shared memory, modified, and
written back to global memory.

```
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (is_elected()) {
    // Initiate bulk tensor copy.
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
      &smem_buffer, &tensor_map, tensor_coords,
      cuda::device::barrier_native_handle(bar));
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (is_elected()) {
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_global, ptx::space_shared,
      &tensor_map, tensor_coords, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

**Negative indices and out of bounds**. When part of the tile that is being
*read* from global to shared memory is out of bounds, the shared memory that
corresponds to the out of bounds area is zero-filled. The top-left corner
indices of the tile may also be negative. When *writing* from shared to global
memory, parts of the tile may be out of bounds, but the top left corner cannot
have any negative indices.

**Size and stride**. The size of a tensor is the number of elements along one
dimension. All sizes must be greater than one. The stride is the number of bytes
between elements of the same dimension. For instance, a 4 x 4 matrix of
integers has sizes 4 and 4. Since it has 4 bytes per element, the strides are 4
and 16 bytes. Due to alignment requirements, a 4 x 3 row-major matrix of
integers must have strides of 4 and 16 bytes as well. Each row is padded with 4
extra bytes to ensure that the start of the next row is aligned to 16 bytes.
More information about alignment requirements can be found in the table below.

Table 23 Alignment requirements for multi-dimensional bulk tensor asynchronous copy operations.

| Address / Size | Alignment |
| --- | --- |
| Global memory address | Must be 16 byte aligned. |
| Global memory sizes | Must be greater than or equal to one. Does not have to be a multiple of 16 bytes. |
| Global memory strides | Must be multiples of 16 bytes. |
| Shared memory address | Must be 128 byte aligned. |
| Shared memory barrier address | Must be 8 byte aligned (this is guaranteed by `cuda::barrier`). |
| Size of transfer | Must be a multiple of 16 bytes. |

