## 4.1.2. Unified Memory on Devices with only CUDA Managed Memory Support

For devices with compute capability 6.x or higher but without pageable memory access, see table [Overview of Unified Memory Paradigms](../02-basics/understanding-memory.html#table-unified-memory-levels), CUDA managed memory is fully supported and coherent but the GPU cannot access system-allocated memory. The programming model and performance tuning of unified memory is largely similar to the model as described in the section, [Unified Memory on Devices with Full CUDA Unified Memory Support](#um-pageable-systems),
with the notable exception that system allocators cannot be used to allocate memory. Thus, the following list of sub-sections do not apply:

* [Unified Memory: In-Depth Examples](#um-system-allocator)
* [CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency](#um-hw-coherency)
* [Atomic Accesses and Synchronization Primitives](#um-atomics)
* [Access Counter Migration](#um-access-counters)
* [Avoid Frequent Writes to GPU-Resident Memory from the CPU](#um-traffic-hd)
* [Exploiting Asynchronous Access to System Memory](#um-async-access)

## 4.1.3. Unified Memory on Windows, WSL, and Tegra

Note

This section is only looking at devices with compute capability lower than 6.0 or Windows platforms, devices with `concurrentManagedAccess` property set to 0.

Devices with compute capability lower than 6.0 or Windows platforms, devices with `concurrentManagedAccess` property set to 0, see [Overview of Unified Memory Paradigms](../02-basics/understanding-memory.html#table-unified-memory-levels), support CUDA managed memory with the following limitations:

* **Data Migration and Coherency**: Fine-grained movement of the managed data to GPU on-demand is not supported. Whenever a GPU kernel is launched all managed memory generally has to be transferred to GPU memory to avoid faulting on memory access. Page faulting is only supported from the CPU side.
* **GPU Memory Oversubscription**: They cannot allocate more managed memory than the physical size of GPU memory.
* **Coherency and Concurrency**: Simultaneous access to managed memory is not possible, because coherence could not be guaranteed if the CPU accessed a unified memory allocation while a GPU kernel is active because of the missing GPU page faulting mechanism.

### 4.1.3.1. Multi-GPU

On systems with devices of compute capabilities lower than 6.0 or Windows platforms managed allocations are automatically visible to all GPUs in a system via the peer-to-peer capabilities of the GPUs. Managed memory allocations behave similar to unmanaged memory allocated using `cudaMalloc()`: the current active device is the home for the physical allocation but other GPUs in the system will access the memory at reduced bandwidth over the PCIe bus.

On Linux the managed memory is allocated in GPU memory as long as all GPUs that are actively being used by a program have the peer-to-peer support. If at any time the application starts using a GPU that doesn’t have peer-to-peer support with any of the other GPUs that have managed allocations on them, then the driver will migrate all managed allocations to system memory. In this case, all GPUs experience PCIe bandwidth restrictions.

On Windows, if peer mappings are not available (for example, between GPUs of different architectures), then the system will automatically fall back to using mapped memory, regardless of whether both GPUs are actually used by a program. If only one GPU is actually going to be used, it is necessary to set the `CUDA_VISIBLE_DEVICES` environment variable before launching the program. This constrains which GPUs are visible and allows managed memory to be allocated in GPU memory.

Alternatively, on Windows users can also set `CUDA_MANAGED_FORCE_DEVICE_ALLOC` to a non-zero value to force the driver to always use device memory for physical storage. When this environment variable is set to a non-zero value, all devices used in that process that support managed memory have to be peer-to-peer compatible with each other.
The error `::cudaErrorInvalidDevice` will be returned if a device that supports managed memory is used and it is not peer-to-peer compatible with any of the other managed memory supporting devices that were previously used in that process, even if `::cudaDeviceReset` has been called on those devices. These environment variables are described in [CUDA Environment Variables](../05-appendices/environment-variables.html#cuda-environment-variables).

### 4.1.3.2. Coherency and Concurrency

To ensure coherency the unified memory programming model puts constraints on data accesses while both the CPU and GPU are executing concurrently. In effect, the GPU has exclusive access to all managed data and the CPU is not permitted to access it,
while any kernel operation is executing, regardless of whether the specific kernel is actively using the data.
Concurrent CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault because the page is considered inaccessible to the CPU.

For example the following code runs successfully on devices of compute capability 6.x due to the GPU page faulting capability which lifts all restrictions on simultaneous access
but fails on on pre-6.x architectures and Windows platforms because the GPU program kernel is still active when the CPU touches `y`:

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    y = 20;            // Error on GPUs not supporting concurrent access

    cudaDeviceSynchronize();
    return  0;
}
```

The program must explicitly synchronize with the GPU before accessing `y` (regardless of whether the GPU kernel actually touches `y` (or any managed data at all):

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    y = 20;            //  Success on GPUs not supporting concurrent access
    return  0;
}
```

Note that any function call that logically guarantees the GPU completes its work is valid to ensure logically that the GPU work is completed, see [Explicit Synchronization](../03-advanced/advanced-host-programming.html#advanced-host-explicit-synchronization).

Note that if memory is dynamically allocated with `cudaMallocManaged()` or `cuMemAllocManaged()` while the GPU is active, the behavior of the memory is unspecified until additional work is launched or the GPU is synchronized.
Attempting to access the memory on the CPU during this time may or may not cause a segmentation fault. This does not apply to memory allocated using the flag `cudaMemAttachHost` or `CU_MEM_ATTACH_HOST`.

### 4.1.3.3. Stream Associated Unified Memory

The CUDA programming model provides streams as a mechanism for programs to indicate dependence and independence among kernel launches. Kernels launched into the same stream are guaranteed to execute consecutively,
while kernels launched into different streams are permitted to execute concurrently. See section [CUDA Streams](../02-basics/asynchronous-execution.html#cuda-streams).

#### 4.1.3.3.1. Stream Callbacks

It is legal for the CPU to access managed data from within a stream callback, provided no other stream that could potentially be accessing managed data is active on the GPU. In addition, a callback that is not followed by any device work can be used for synchronization: for example, by signaling a condition variable from inside the callback; otherwise, CPU access is valid only for the duration of the callback(s).
There are several important points of note:

1. It is always permitted for the CPU to access non-managed mapped memory data while the GPU is active.
2. The GPU is considered active when it is running any kernel, even if that kernel does not make use of managed data. If a kernel might use data, then access is forbidden
3. There are no constraints on concurrent inter-GPU access of managed memory, other than those that apply to multi-GPU access of non-managed memory.
4. There are no constraints on concurrent GPU kernels accessing managed data.

Note how the last point allows for races between GPU kernels, as is currently the case for non-managed GPU memory. In the perspective of the GPU, managed memory functions are identical to non-managed memory.
The following code example illustrates these points:

```
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}
```

#### 4.1.3.3.2. Managed memory associated to streams allows for finer-grained control

Unified memory builds upon the stream-independence model by allowing a CUDA program to explicitly associate managed allocations with a CUDA stream.
In this way, the programmer indicates the use of data by kernels based on whether they are launched into a specified stream or not. This enables opportunities for concurrency based on program-specific data access patterns.
The function to control this behavior is:

```
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                     void *ptr,
                                     size_t length=0,
                                     unsigned int flags=0);
```

The `cudaStreamAttachMemAsync()` function associates length bytes of memory starting from ptr with the specified stream. This allows CPU access to that memory region so long as all operations in stream have completed, regardless of whether other streams are active.
In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.
Most importantly, if an allocation is not associated with a specific stream, it is visible to all running kernels regardless of their stream.
This is the default visibility for a `cudaMallocManaged()` allocation or a `__managed__` variable; hence, the simple-case rule that the CPU may not touch the data while any kernel is running.

Note

By associating an allocation with a specific stream, the program makes a guarantee that only kernels launched into that stream will touch that data. No error checking is performed by the unified memory system.

Note

In addition to allowing greater concurrency, the use of `cudaStreamAttachMemAsync()` can enable data transfer optimizations within the unified memory system that may affect latencies and other overhead.

The following example shows how to explicitly associate `y` with host accessibility, thus enabling access at all times from the CPU. (Note the absence of `cudaDeviceSynchronize()` after the kernel call.) Accesses to `y` by the GPU running kernel will now produce undefined results.

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();          // Wait for Host attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>(); // Note: Launches into stream1.
    y = 20;                           // Success – a kernel is running but “y”
                                      // has been associated with no stream.
    return  0;
}
```

#### 4.1.3.3.3. A more elaborate example on multithreaded host programs

The primary use for `cudaStreamAttachMemAsync()` is to enable independent task parallelism using CPU threads. Typically in such a program, a CPU thread creates its own stream for all work that it generates because using CUDA’s NULL stream would cause dependencies between threads.
The default global visibility of managed data to any GPU stream can make it difficult to avoid interactions between CPU threads in a multi-threaded program. Function `cudaStreamAttachMemAsync()` is therefore used to associate a thread’s managed allocations with that thread’s own stream, and the association is typically not changed for the life of the thread.
Such a program would simply add a single call to `cudaStreamAttachMemAsync()` to use unified memory for its data accesses:

```
// This function performs some task, in its own , in its own private stream and can be run in parallel
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}
```

In this example, the allocation-stream association is established just once, and then data is used repeatedly by both the host and device. The result is much simpler code than occurs with explicitly copying data between host and device, although the result is the same.

The function `cudaMallocManaged()` specifies the cudaMemAttachHost flag, which creates an allocation that is initially invisible to device-side execution. (The default allocation would be visible to all GPU kernels on all streams.)
This ensures that there is no accidental interaction with another thread’s execution in the interval between the data allocation and when the data is acquired for a specific stream.

Without this flag, a new allocation would be considered in-use on the GPU if a kernel launched by another thread happens to be running.
This might impact the thread’s ability to access the newly allocated data from the CPU before it is able to explicitly attach it to a private stream. To enable safe independence between threads, therefore, allocations should be made specifying this flag.

An alternative would be to place a process-wide barrier across all threads after the allocation has been attached to the stream. This would ensure that all threads complete their data/stream associations before any kernels are launched, avoiding the hazard.
A second barrier would be needed before the stream is destroyed because stream destruction causes allocations to revert to their default visibility. The `cudaMemAttachHost` flag exists both to simplify this process, and because it is not always possible to insert global barriers where required.

#### 4.1.3.3.4. Data Movement of Stream Associated Unified Memory

Memcpy()/Memset() with stream associated unified memory behaves different on devices where `concurrentManagedAccess` is not set, the following rules apply:

If `cudaMemcpyHostTo*` is specified and the source data is unified memory, then it will be accessed from the host if it is coherently accessible from the host in the copy stream [(1)](#um-legacy-memcpy-cit1); otherwise it will be accessed from the device. Similar rules apply to the destination when `cudaMemcpy*ToHost` is specified and the destination is unified memory.

If `cudaMemcpyDeviceTo*` is specified and the source data is unified memory, then it will be accessed from the device. The source must be coherently accessible from the device in the copy stream [(2)](#um-legacy-memcpy-cit2); otherwise, an error is returned. Similar rules apply to the destination when `cudaMemcpy*ToDevice` is specified and the destination is unified memory.

If `cudaMemcpyDefault` is specified, then unified memory will be accessed from the host either if it cannot be coherently accessed from the device in the copy stream [(2)](#um-legacy-memcpy-cit2) or if the preferred location for the data is `cudaCpuDeviceId` and it can be coherently accessed from the host in the copy stream [(1)](#um-legacy-memcpy-cit1); otherwise, it will be accessed from the device.

When using `cudaMemset*()` with unified memory, the data must be coherently accessible from the device in the stream being used for the `cudaMemset()` operation [(2)](#um-legacy-memcpy-cit2); otherwise, an error is returned.

When data is accessed from the device either by `cudaMemcpy*` or `cudaMemset*`, the stream of operation is considered to be active on the GPU. During this time, any CPU access of data that is associated with that stream or data that has global visibility, will result in a segmentation fault if the GPU has a zero value for the device attribute `concurrentManagedAccess`. The program must synchronize appropriately to ensure the operation has completed before accessing any associated data from the CPU.

> 1. Coherently accessible from the host in a given stream means that the memory neither has global visibility nor is it associated with the given stream.

> 2. Coherently accessible from the device in a given stream means that the memory either has global visibility or is associated with the given stream.

## 4.1.4. Performance Hints

Performance hints allow programmers to provide CUDA with more information about unified memory usage.
CUDA uses performance hints to managed memory more efficiently and improve application performance.
Performance hints never impact the correctness of an application.
Performance hints only affect performance.

Note

Applications should only use unified memory performance hints if they improve performance.

Performance hints may be used on any unified memory allocation, including CUDA managed memory.
On systems with full CUDA unified memory support, performance hints can be applied to all system-allocated memory.

### 4.1.4.1. Data Prefetching

The `cudaMemPrefetchAsync` API is an asynchronous stream-ordered API that may migrate data to reside closer to the specified processor.
The data may be accessed while it is being prefetched.
The migration does not begin until all prior operations in the stream have completed,
and completes before any subsequent operation in the stream.

```
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream=0);
```

A memory region containing `[devPtr, devPtr + count)` may be migrated to
the destination device `location.id` if `location.type` is `cudaMemLocationTypeDevice`, or CPU if `location.type` is `cudaMemLocationTypeHost`,
when the prefetch task is executed in the given `stream`. For details on `flags`, see the current
[CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

Consider the simple code example below:

System Allocator

```
void test_prefetch_sam(const cudaStream_t& s) {
  // initialize data on CPU
  char *data = (char*)malloc(dataSizeBytes);
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // use data on GPU
  const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);
  free(data);
}
```

Managed

```
void test_prefetch_managed(const cudaStream_t& s) {
  // initialize data on CPU
  char *data;
  cudaMallocManaged(&data, dataSizeBytes);
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // use data on GPU
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
```

### 4.1.4.2. Data Usage Hints

When multiple processors simultaneously access the same data,
`cudaMemAdvise` may be used to hint how the data at
`[devPtr, devPtr + count)` will be accessed:

```
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

The example shows how to use `cudaMemAdvise`:

```
  init_data(data, dataSizeBytes);
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // encourage data to move to GPU before use
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // use data on GPU
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

  // encourage data to move back to CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  cudaStreamSynchronize(s);

  // use data on CPU
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
// test-prefetch-managed-end

static const int maxDevices = 1;
static const int maxOuterLoopIter = 3;
static const int maxInnerLoopIter = 4;

// test-advise-managed-begin
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
```

Where `advice` may take the following values:

* `cudaMemAdviseSetReadMostly`:
  :   This implies that the data is mostly going to be read from and only occasionally written to.
      In general, it allows trading off read bandwidth for write bandwidth on this region.

* `cudaMemAdviseSetPreferredLocation`:
  :   This hint sets the preferred location for the data to be the specified device’s physical memory.
      This hint encourages the system to keep the data at the preferred location, but does not guarantee it.
      Passing in a value of `cudaMemLocationTypeHost` for location.type sets the preferred location as CPU memory.
      Other hints, like `cudaMemPrefetchAsync`, may override this hint and allow the memory to migrate away from its preferred location.

* `cudaMemAdviseSetAccessedBy`:
  :   In some systems, it may be beneficial for performance to establish a
      mapping into memory before accessing the data from a given processor.
      This hint tells the system that the data will be frequently accessed by `location.id`
      when `location.type` is `cudaMemLocationTypeDevice`,
      enabling the system to assume that creating these mappings pays off.
      This hint does not imply where the data should reside,
      but it can be combined with `cudaMemAdviseSetPreferredLocation` to specify that.
      On hardware-coherent systems, this hint switches on access counter migration, see [Access Counter Migration](#um-access-counters).

Each advice can be also unset by using one of the following values:
`cudaMemAdviseUnsetReadMostly`, `cudaMemAdviseUnsetPreferredLocation` and
`cudaMemAdviseUnsetAccessedBy`.

The example shows how to use `cudaMemAdvise`:

System Allocator

```
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // Allocate memory using malloc or cudaMallocManaged
  dataPtr = (char*)malloc(dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  free(dataPtr);
}
```

Managed

```
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);

  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written by the CPU each outer loop iteration
    init_data(dataPtr, dataSize);

    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  cudaFree(dataPtr);
}
```

### 4.1.4.3. Memory Discarding

The `cudaMemDiscardBatchAsync` API allows applications to inform the CUDA runtime that the contents
of specified memory ranges are no longer useful. The Unified Memory driver performs automatic memory
transfers due to fault-based migration or memory evictions to support device memory oversubscription.
These automatic memory transfers can sometimes be redundant, which severely decreases performance.
Marking an address range as ‘discard’ will inform the Unified Memory driver that the application has
consumed the contents in the range and there is no need to migrate this data on prefetches or page evictions
in order to make room for other allocations. Reading a discarded page without a subsequent write access
or prefetch will yield an indeterminate value. Whereas any new writes after the discard operation is guaranteed
to be seen by a subsequent read access. Concurrent accesses or prefetches to address ranges being discarded
will result in undefined behavior.

```
cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                    size_t *sizes,
                                    size_t count,
                                    unsigned long long flags,
                                    cudaStream_t stream);
```

The function performs a batch of memory discards on address ranges specified in `dptrs` and `sizes` arrays.
Both arrays must be of the same length as specified by `count`. Each memory range must refer to
managed memory allocated via `cudaMallocManaged` or declared via `__managed__` variables.

The `cudaMemDiscardAndPrefetchBatchAsync` API combines both discard and prefetch operations. Calling
`cudaMemDiscardAndPrefetchBatchAsync` is semantically equivalent to calling `cudaMemDiscardBatchAsync`
followed by `cudaMemPrefetchBatchAsync`, but is more optimal. This is useful when the application needs
the memory to be on the target location but does not need the contents of the memory.

```
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                               size_t *sizes,
                                               size_t count,
                                               struct cudaMemLocation *prefetchLocs,
                                               size_t *prefetchLocIdxs,
                                               size_t numPrefetchLocs,
                                               unsigned long long flags,
                                               cudaStream_t stream);
```

The `prefetchLocs` array specifies the destinations for prefetching, while `prefetchLocIdxs`
indicates which operations each prefetch location applies to. For example, if a batch has 10 operations
and the first 6 should be prefetched to one location while the remaining 4 to another, then
`numPrefetchLocs` would be 2, `prefetchLocIdxs` would be {0, 6}, and `prefetchLocs` would
contain the two destination locations.

**Important considerations:**

* Reading from a discarded range without a subsequent write or prefetch will return an indeterminate value
* The discard operation can be undone by writing to the range or prefetching it via `cudaMemPrefetchAsync`
* Any reads, writes, or prefetches that occur simultaneously with the discard operation result in undefined behavior
* All devices must have a non-zero value for `cudaDevAttrConcurrentManagedAccess`

### 4.1.4.4. Querying Data Usage Attributes on Managed Memory

A program can query memory range attributes assigned through `cudaMemAdvise`
or `cudaMemPrefetchAsync` on CUDA managed memory by using the following API:

```
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

This function queries an attribute of the memory range starting at `devPtr` with a size of `count` bytes.
The memory range must refer to managed memory allocated via `cudaMallocManaged` or
declared via `__managed__` variables. It is possible to query the following attributes:

* `cudaMemRangeAttributeReadMostly`: returns 1 if the entire memory range has the `cudaMemAdviseSetReadMostly` attribute set, or 0 otherwise.
* `cudaMemRangeAttributePreferredLocation`: the result returned will be a GPU device id or `cudaCpuDeviceId` if the entire memory range has the corresponding processor as preferred location, otherwise `cudaInvalidDeviceId` will be returned.
  An application can use this query API to make decision about staging data through CPU or GPU depending on the preferred location attribute of the managed pointer.
  Note that the actual location of the memory range at the time of the query may be different from the preferred location.
* `cudaMemRangeAttributeAccessedBy`: will return the list of devices that have that advise set for that memory range.
* `cudaMemRangeAttributeLastPrefetchLocation`: will return the last location to which the memory range was prefetched explicitly using `cudaMemPrefetchAsync`. Note that this simply returns the last location that the application requested to prefetch the memory range to. It gives no indication as to whether the prefetch operation to that location has completed or even begun.
* `cudaMemRangeAttributePreferredLocationType`: it returns the location type of the preferred location with the following values:

  + `cudaMemLocationTypeDevice`: if all pages in the memory range have the same GPU as their preferred location,
  + `cudaMemLocationTypeHost`: if all pages in the memory range have the CPU as their preferred location,
  + `cudaMemLocationTypeHostNuma`: if all the pages in the memory range have the same host NUMA node ID as their preferred location,
  + `cudaMemLocationTypeInvalid`: if either all the pages don’t have the same preferred location or some of the pages don’t have a preferred location at all.
* `cudaMemRangeAttributePreferredLocationId`: returns the device ordinal if the `cudaMemRangeAttributePreferredLocationType` query for the same address range returns `cudaMemLocationTypeDevice`. If the preferred location type is a host NUMA node, it returns the host NUMA node ID. Otherwise, the id should be ignored.
* `cudaMemRangeAttributeLastPrefetchLocationType`: returns the last location type to which all pages in the memory range were prefetched explicitly via `cudaMemPrefetchAsync`. The following values are returned:

  + `cudaMemLocationTypeDevice`: if all pages in the memory range were prefetched to the same GPU,
  + `cudaMemLocationTypeHost`: if all pages in the memory range were prefetched to the CPU,
  + `cudaMemLocationTypeHostNuma`: if all the pages in the memory range were prefetched to the same host NUMA node ID,
  + `cudaMemLocationTypeInvalid`: if either all the pages were not prefetched to the same location or some of the pages were never prefetched at all.
* `cudaMemRangeAttributeLastPrefetchLocationId`: if the `cudaMemRangeAttributeLastPrefetchLocationType` query for the same address range returns `cudaMemLocationTypeDevice`, it will be a valid device ordinal or if it returns `cudaMemLocationTypeHostNuma`, it will be a valid host NUMA node ID. Otherwise, the id should be ignored.

Additionally, multiple attributes can be queried by using corresponding `cudaMemRangeGetAttributes` function.

### 4.1.4.5. GPU Memory Oversubscription

Unified memory enables applications to *oversubscribe* the memory of any individual processor:
in other words they can allocate and share arrays larger than
the memory capacity of any individual processor in the system,
enabling among others out-of-core processing of datasets that do not fit within
a single GPU, without adding significant complexity to the programming model.

Additionally, multiple attributes can be queried by using corresponding `cudaMemRangeGetAttributes` function.
