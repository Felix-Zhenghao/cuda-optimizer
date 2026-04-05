# 4.11. Asynchronous Data Copies

Building on [Section 3.2.5](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies), this section provides detailed guidance and examples for asynchronous data movement within the GPU memory hierarchy. It covers LDGSTS for element-wise copies, the Tensor Memory Accelerator (TMA) for bulk (one-dimensional and multi-dimensional) transfers, and STAS for register to distributed shared memory copies, and shows how these mechanisms integrate with [asynchronous barriers](async-barriers.html#asynchronous-barriers) and [pipelines](pipelines.html#pipelines).

## 4.11.1. Using LDGSTS

Many CUDA applications require frequent data movement between global and shared memory. Often, this involves copying smaller data elements or performing irregular memory access patterns. The primary goal of LDGSTS (CC 8.0+, see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)) is to provide an efficient asynchronous data transfer mechanism from global memory to shared memory for smaller, element-wise data transfers while enabling better utilization of compute resources through overlapped execution.

**Dimensions**. LDGSTS supports copying 4, 8, or 16 bytes. Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode, in which case data is also cached in the L1, while copying 16-bytes enables the L1 BYPASS mode, in which case the L1 is not polluted.

**Source and destination**. The only direction supported for asynchronous copy operations with LDGSTS is from global to shared memory. The pointers need to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied. Best performance is achieved when the alignment of both shared memory and global memory is 128 bytes.

**Asynchronicity**. Data transfers using LDGSTS are [asynchronous](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async thread operations (see [Async Thread and Async Proxy](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. *Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future*.

LDGSTS must provide a signal when the operation is complete. LDGSTS can use [shared memory barriers](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) or [pipelines](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines) as mechanisms to provide completion signals. By default, each thread only waits for its own LDGSTS copies. Thus, if you use LDGSTS to prefetch some data that will be shared with other threads, a `__syncthreads()` is necessary after synchronizing with the LDGSTS completion mechanism.

Table 18 Asynchronous copies with possible source and destination memory spaces and completion mechanisms using LDGSTS. An empty cell indicates that a source-destination pair is not supported.

| Direction | | Asynchronous Copy (LDGSTS, CC 8.0+) | |
| --- | --- | --- | --- |
| Source | Destination | Completion Mechanism | API |
| global | global |  |  |
| shared::cta | global |  |  |
| global | shared::cta | shared memory barrier, pipeline | [cuda::memcpy\_async](https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html), [cooperative\_groups::memcpy\_async](../05-appendices/device-callable-apis.html#cg-api-async-memcpy), [\_\_pipeline\_memcpy\_async](../05-appendices/device-callable-apis.html#pipeline-primitives-interface) |
| global | shared::cluster |  |  |
| shared::cluster | shared::cta |  |  |
| shared::cta | shared::cta |  |  |

In the following sections, we will demonstrate how to use LDGSTS through examples and explain the differences between the different APIs.

### 4.11.1.1. Batching Loads in Conditional Code

In this stencil example, the first warp of the thread block is responsible for collectively loading all the required data from the center as well as the left and right halos. With synchronous copies, due to the conditional nature of the code, the compiler may choose to generate a sequence of load-from-global (LDG) store-to-shared (STS) instructions instead of 3 LDGs followed by 3 STSs, which would be the optimal way to load the data to hide the global memory latency.

```
__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    // Left halo (8 elements) - center (32 elements) - right halo (8 elements)
    __shared__ float buffer[8 + 32 + 8];
    const int tid = threadIdx.x;

    if (tid < 8) {
        buffer[tid] = left[tid]; // Left halo
    } else if (tid >= 32 - 8) {
        buffer[tid + 16] = right[tid]; // Right halo
    }
    if (tid < 32) {
      buffer[tid + 8] = center[tid]; // Center
    }
    __syncthreads();

    // Compute stencil
}
```

To ensure that the data is loaded in the optimal way, we can replace the synchronous memory copies with asynchronous copies that load data directly from global memory to shared memory. This not only reduces register usage by copying the data directly to shared memory, but also ensures all loads from global memory are in-flight.

CUDA C++ `cuda::memcpy_async`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda/barrier>  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     auto block = cooperative_groups::this_thread_block();     auto thread = cooperative_groups::this_thread();     using barrier_t = cuda::barrier<cuda::thread_scope_block>;     __shared__ barrier_t barrier;     __shared__ float buffer[8 + 32 + 8];          // Initialize synchronization object.     if (block.thread_rank() == 0) {         init(&barrier, block.size());     }     __syncthreads();      // Version 1: Issue the copies in individual threads.     if (tid < 8) {         cuda::memcpy_async(buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Left halo         // or cuda::memcpy_async(thread, buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     } else if (tid >= 32 - 8) {         cuda::memcpy_async(buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Right halo         // or cuda::memcpy_async(thread, buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     }     if (tid < 32) {         cuda::memcpy_async(buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); // Center         // or cuda::memcpy_async(thread, buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier);     }          // Version 2: Cooperatively issue the copies across all threads.     cuda::memcpy_async(block, buffer, left, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Left halo     cuda::memcpy_async(block, buffer + 8, center, cuda::aligned_size_t<4>(32 * sizeof(float)), barrier); // Center     cuda::memcpy_async(block, buffer + 40, right, cuda::aligned_size_t<4>(8 * sizeof(float)), barrier); // Right halo          // Wait for all copies to complete.     barrier.arrive_and_wait();     __syncthreads();      // Compute stencil       } ``` |

CUDA C++ `cooperative_groups::memcpy_async`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cooperative_groups/memcpy_async.h>  namespace cg = cooperative_groups;  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     cg::thread_block block = cg::this_thread_block();     // Left halo (8 elements) - center (32 elements) - right halo (8 elements).      __shared__ float buffer[8 + 32 + 8];      // Cooperatively issue the copies across all threads.     cg::memcpy_async(block, buffer, left, 8 * sizeof(float)); // Left halo     cg::memcpy_async(block, buffer + 8, center, 32 * sizeof(float)); // Center     cg::memcpy_async(block, buffer + 40, right, 8 * sizeof(float)); // Right halo     cg::wait(block); // Waits for all copies to complete.     __syncthreads();      // Compute stencil. } ``` |

CUDA C primitives

|  |
| --- |
| ``` #include <cuda_pipeline.h>  __global__ void stencil_kernel(const float *left, const float *center, const float *right) {     // Left halo (8 elements) - center (32 elements) - right halo (8 elements).     __shared__ float buffer[8 + 32 + 8];     const int tid = threadIdx.x;      if (tid < 8) {         __pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float)); // Left halo     } else if (tid >= 32 - 8) {         __pipeline_memcpy_async(buffer + tid + 16, right + tid, sizeof(float)); // Right halo     }     if (tid < 32) {         __pipeline_memcpy_async(buffer + tid + 8, center + tid, sizeof(float)); // Center     }     __pipeline_commit();     __pipeline_wait_prior(0);     __syncthreads();      // Compute stencil. } ``` |

The `cuda::memcpy_async` overload for `cuda::barrier` enables synchronizing asynchronous data transfers using an [asynchronous barrier](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers). This overload executes the copy operation as-if performed by another thread bound to the barrier by incrementing the expected count of the current phase on creation, and decrementing it on completion of the copy operation, such that the phase of the `barrier` will only advance when all threads participating in the barrier have arrived, and all `memcpy_async` bound to the current phase of the barrier have completed. We use a block-wide `barrier`, where all threads in the block participate, and merge the arrival and wait on the barrier with `arrive_and_wait`, since we do not perform any work between the phases.

Note that we can either use thread-level copies (version 1) or collective copies (version 2) to achieve the same result. In version 2, the API will automatically handle how the copies are done under the hood. In both versions, we use `cuda::aligned_size_t<4>()` to inform the compiler that the data is aligned to 4 bytes and the size of the data to copy is a multiple of 4 to enable use of LDGSTS. Note that for interoperability with `cuda::barrier`, `cuda::memcpy_async` from the `cuda/barrier` header is used here.

The [cooperative\_groups::memcpy\_async](../05-appendices/device-callable-apis.html#cg-api-async-memcpy) implementation coordinates the memory transfers collectively across all threads in the block, but synchronizes completion with `cg::wait(block)` instead of explicit barrier operations.

The implementation based on the low-level primitives uses `__pipeline_memcpy_async()` to initiate element-wise memory transfers, `__pipeline_commit()` to commit the batch of copies, and `__pipeline_wait_prior(0)` to wait for all operations in the pipeline to complete. This provides the most direct control at the expense of more verbose code compared to the higher-level APIs. It also ensures LDGSTS will be used under the hood, which is not guaranteed with the higher-level APIs.

Note

The `cooperative_groups::memcpy_async` API is less efficient than the other APIs in this example because it automatically commits each copy operation immediately upon launch, preventing the optimization of batching multiple copies before a single commit operation that the other APIs enable.

### 4.11.1.2. Prefetching Data

In this example, we will demonstrate how to use asynchronous data copies to prefetch data from global memory to shared memory. In an iterative copy and compute pattern, this allows hiding the latency of data transfers of future iterations with computation on the current iteration, potentially increasing bytes-in-flight.

CUDA C++ `cuda::memcpy_async`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda/pipeline>  template <size_t num_stages = 2 /* Pipeline with num_stages stages */> __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {     auto grid = cooperative_groups::this_grid();     auto block = cooperative_groups::this_thread_block();     auto thread = cooperative_groups::this_thread();     assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size      extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes     size_t shared_offset[num_stages];     for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();      auto block_batch = [&](size_t batch) -> int {         return block.group_index().x * block.size() + grid.size() * batch;     };      // Fill the pipeline with the first ``num_stages`` batches.     for (int s = 0; s < num_stages; ++s) {         pipeline.producer_acquire();         cuda::memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s) + tid, cuda::aligned_size_t<4>(sizeof(int)), pipeline);         pipeline.producer_commit();     }      int stage = 0;      // compute_batch: next batch to process     // fetch_batch:   next batch to fetch from global memory     for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; ++compute_batch, ++fetch_batch) {         // Wait for the first requested stage to complete.         constexpr size_t pending_batches = num_stages - 1;         cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);         __syncthreads(); // Not required if each thread works on the data it copied.          // Compute on the current batch         compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);                  // Release the current stage.         pipeline.consumer_release();         __syncthreads(); // Not required if each thread works on the data it copied.          // Load future stage ``num_stages`` ahead of current compute batch.         pipeline.producer_acquire();         if (fetch_batch < batch_size) {             cuda::memcpy_async(shared + shared_offset[stage] + tid, global_in + block_batch(fetch_batch) + tid, cuda::aligned_size_t<4>(sizeof(int)), pipeline);         }         pipeline.producer_commit();         stage = (stage + 1) % num_stages;     } } ``` |

CUDA C++ `cooperative_groups::memcpy_async`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cooperative_groups/memcpy_async.h>  namespace cg = cooperative_groups;  template <size_t num_stages = 2 /* Pipeline with num_stages stages */> __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {     auto grid = cooperative_groups::this_grid();     auto block = cooperative_groups::this_thread_block();     assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size      extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes     size_t shared_offset[num_stages];     for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();      auto block_batch = [&](size_t batch) -> int {         return block.group_index().x * block.size() + grid.size() * batch;     };      // Fill the pipeline with the first ``num_stages`` batches.     for (int s = 0; s < num_stages; ++s) {         size_t block_batch_idx = block_batch(s);         cg::memcpy_async(block, shared + shared_offset[s], global_in + block_batch_idx, cuda::aligned_size_t<4>(sizeof(int));     }      int stage = 0;      // compute_batch: next batch to process     // fetch_batch:   next batch to fetch from global memory     for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; ++compute_batch, ++fetch_batch) {         // Wait for the first requested stage to complete.         size_t pending_batches = (fetch_batch < batch_size - num_stages) ? num_stages - 1 : batch_size - fetch_batch - 1;         cg::wait_prior(pending_batches);         __syncthreads(); // Not required if each thread works on the data it copied.          // Compute on the current batch.         compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);                  __syncthreads(); // Not required if each thread works on the data it copied.          // Load future stage ``num_stages`` ahead of current compute batch.         size_t fetch_batch_idx = block_batch(fetch_batch);         if (fetch_batch < batch_size) {             cg::memcpy_async(block, shared + shared_offset[stage], global_in + block_batch(fetch_batch), cuda::aligned_size_t<4>(sizeof(int)) * block.size());         }         stage = (stage + 1) % num_stages;     } } ``` |

CUDA C primitives

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda_awbarrier_primitives.h>  template <size_t num_stages = 2 /* Pipeline with num_stages stages */> __global__ void prefetch_kernel(int* global_out, int const* global_in, size_t size, size_t batch_size) {     auto grid = cooperative_groups::this_grid();     auto block = cooperative_groups::this_thread_block();     assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size      extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes     size_t shared_offset[num_stages];     for (int s = 0; s < num_stages; ++s) shared_offset[s] = s * block.size();      auto block_batch = [&](size_t batch) -> int {         return block.group_index().x * block.size() + grid.size() * batch;     };      // Fill the pipeline with the first ``num_stages`` batches.     for (int s = 0; s < num_stages; ++s) {         __pipeline_memcpy_async(shared + shared_offset[s] + tid, global_in + block_batch(s)+ tid, cuda::aligned_size_t<4>(sizeof(int)));         __pipeline_commit();     }      // compute_batch: next batch to process     // fetch_batch:   next batch to fetch from global memory     for (size_t compute_batch = 0, fetch_batch = num_stages; compute_batch < batch_size; ++compute_batch, ++fetch_batch) {         // Wait for the first requested stage to complete.         constexpr size_t pending_batches = num_stages - 1;         __pipeline_wait_prior<pending_batches>();         __syncthreads(); // Not required if each thread works on the data it copied.          // Compute on the current batch.         compute(global_out + block_batch(compute_batch) + tid, shared + shared_offset[stage] + tid);                  __syncthreads(); // Not required if each thread works on the data it copied.          // Load future stage ``num_stages`` ahead of current compute batch.         if (fetch_batch < batch_size) {             __pipeline_memcpy_async(shared + shared_offset[stage] + tid, global_in + block_batch(fetch_batch) + tid, cuda::aligned_size_t<4>(sizeof(int)));         }         __pipeline_commit();         stage = (stage + 1) % num_stages;     } } ``` |

The `cuda::memcpy_async` implementation demonstrates a multi-stage data prefetching using `cuda::pipeline` (see [Pipelines](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines)) with `cuda::memcpy_async`. It:

* Initializes a pipeline that is local to the thread.
* Kick-starts the pipeline by scheduling `num_stages` `memcpy_async` operations.
* Loops over all the batches: it blocks all threads on the completion of the current batch, then performs the computation on the current batch and finally schedules the next `memcpy_async` if there is one.

The `cooperative_groups::memcpy_async` implementation demonstrates multi-stage data prefetching using `cooperative_groups::memcpy_async`. The main difference with the previous implementation is that we do not use a pipeline object, but instead rely on `cooperative_groups::memcpy_async` to schedule the memory transfers in stages under the hood.

The CUDA C primitives implementation demonstrates multi-stage data prefetching using the low-level primitives in a quite similar manner to the first.

An important detail to enable efficient code generation in this example is to keep `num_stages` batches in the pipeline, even if there are no more batches to fetch. This is done by committing to the pipeline even if there are no more batches to fetch (`pipeline.producer_commit()` or `__pipeline_commit()`). Note that this is not possible with the cooperative groups API as we have no access to the internal pipeline.

### 4.11.1.3. Producer-Consumer Pattern Through Warp Specialization

In this example, we will demonstrate how to implement a producer-consumer pattern where a single warp is specialized as the producer performing asynchronous data copies from global to shared memory, while the remaining warps consume the data from shared memory and perform computations. To enable concurrency between the producer and the consumer threads, we use double-buffering in shared memory. While consumer warps process data in one buffer, the producer warp asynchronously fetches the next batch of data into the other buffer.

CUDA C++ `cuda::memcpy_async`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda/pipeline>  #pragma nv_diag_suppress static_var_with_dynamic_init  using pipeline = cuda::pipeline<cuda::thread_scope_block>;  __device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *in, int N) {   if (batch < num_batches)   {     pipe.producer_acquire();     /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */     cuda::memcpy_async(buffer + stage * buffer_len + threadIdx.x, in + batch * buffer_len + threadIdx.x, cuda::aligned_size_t<4>(sizeof(float)), pipe);     pipe.producer_commit();   } }  __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *out, int N) {   pipe.consumer_wait();   /* consume buffer(stage) and update out(batch) */   pipe.consumer_release(); }  __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len) {   auto block = cooperative_groups::this_thread_block();   constexpr int warpSize = 32;    /* Shared memory buffer declared below is of size 2 * buffer_len      so that we can alternatively work between two buffers.      buffer_0 = buffer and buffer_1 = buffer + buffer_len */   __shared__ extern float buffer[];    const int num_batches = N / buffer_len;    // Create a partitioned pipeline with 2 stages where the first warp is the producer and the other warps are consumers.   constexpr auto scope = cuda::thread_scope_block;   constexpr int num_stages = 2;   cuda::std::size_t producer_count = warpSize;   __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;   pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);    // Producer fills the pipeline   if (block.thread_rank() < producer_count)     for (int s = 0; s < num_stages; ++s)       produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);    // Process the batches   int stage = 0;   for (size_t b = 0; b < num_batches; ++b)   {     if (block.thread_rank() < producer_count)     {       // Producers prefetch the next batch       produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);     }     else     {       // Consumers consume the oldest batch       consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);     }     stage = (stage + 1) % num_stages;   } } ``` |

CUDA C primitives

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda_awbarrier_primitives.h>  __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N) {   for (int i = 0; i < N / buffer_len; ++i)   {     __mbarrier_token_t token = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */     while(!__mbarrier_try_wait(&ready[i % 2], token, 1000)) {}     /* produce, i.e., fill in, buffer_(i%2)  */     __pipeline_memcpy_async(buffer + i * buffer_len + threadIdx.x, in + i * buffer_len + threadIdx.x, cuda::aligned_size_t<4>(sizeof(float)));     __pipeline_arrive_on(filled[i % 2]);     __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */   } }  __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N) {   __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */   __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */   for (int i = 0; i < N / buffer_len; ++i)   {     __mbarrier_token_t token = __mbarrier_arrive(&filled[i % 2]);     while(!__mbarrier_try_wait(&filled[i % 2], token, 1000)) {}     /* consume buffer_(i%2) */     __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */   } }  __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len) {    /* Shared memory buffer declared below is of size 2 * buffer_len      so that we can alternatively work between two buffers.      buffer_0 = buffer and buffer_1 = buffer + buffer_len */   __shared__ extern float buffer[];    /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,      while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */   __shared__ __mbarrier_t bar[4];    // Initialize the barriers   auto block = cooperative_groups::this_thread_block();   if (block.thread_rank() < 4)     __mbarrier_init(bar + block.thread_rank(), block.size());   __syncthreads();    if (block.thread_rank() < warpSize)     produce(bar, bar + 2, buffer, buffer_len, in, N);   else     consume(bar, bar + 2, buffer, buffer_len, out, N); } ``` |

The `cuda::memcpy_async` implementation demonstrates the API with the highest level of abstraction with `cuda::memcpy_async` and a `cuda::pipeline` with 2 stages. It uses a partitioned pipeline (see [Pipelines](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines)) where the first warp serves as a producer and the remaining warps as consumers. Producers initially fill both pipeline stages. Then, in the main processing loop, while consumers process the current batch, producers fetch data for future batches, maintaining a steady flow of work.

The CUDA C primitives implementation based on primitives combines `__pipeline_memcpy_async()` with [shared memory barriers](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) as the completion mechanism to coordinate the asynchronous memory transfers. The `__pipeline_arrive_on()` function associates the memory copy with the barrier. It increments the barrier arrival count by one and when all asynchronous operations sequenced before it have completed, the arrival count is automatically decremented by one and hence the net effect on the arrival count is zero. For this reason, we also need to explicitly wait on the barrier with `__mbarrier_arrive()`.

