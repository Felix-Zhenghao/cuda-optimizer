### 4.16.3.2. Share and Import

**Sharing Memory Handle**
Once the handle is exported, it must be shared with the receiving process or
processes using an inter-process communication protocol. The developer is free to
use any method for sharing the handle. The specific IPC method used depends on the
application’s design and environment. Common methods include OS-specific
inter-process sockets and distributed message passing. Using OS-specific IPC
offers high-performance transfer, but is limited to processes on the
same machine and not portable. Fabric-specific IPC is simpler and more
portable. However, fabric-specific IPC requires system-level support. The chosen
method must securely and reliably transfer the handle data to the target
process so it can be used to import the memory and establish a valid mapping.
The flexibility in choosing the IPC method allows the VMM API to be integrated
into a wide range of system architectures, from single-node applications to
distributed, multi-node setups. In the following code snippets, we’ll provide
examples for sharing and receiving handles using both socket programming and
MPI.

Send: OS-Specific IPC (Linux)

```
int ipcSendShareableHandle(int socket, int fd, pid_t process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);

    struct cmsghdr *cmptr;
    ssize_t readResult;
    struct sockaddr_un cliaddr;
    socklen_t len = sizeof(cliaddr);

    // Construct client address to send this SHareable handle to
    memset(&cliaddr, 0, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[20];
    sprintf(temp, "%s%u", "/tmp/", process);
    strcpy(cliaddr.sun_path, temp);
    len = sizeof(cliaddr);

    // Send corresponding shareable handle to the client
    int sendfd = fd;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

Send: OS-Specific IPC (WIN)

```
int ipcSendShareableHandle(HANDLE *handle, HANDLE &shareableHandle, PROCESS_INFORMATION process) {
    HANDLE hProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process.dwProcessId);
    HANDLE hDup = INVALID_HANDLE_VALUE;
    DuplicateHandle(GetCurrentProcess(), shareableHandle, hProcess, &hDup, 0, FALSE, DUPLICATE_SAME_ACCESS);
    DWORD cbWritten;
    WriteFile(handle->hMailslot[i], &hDup, (DWORD)sizeof(hDup), &cbWritten, (LPOVERLAPPED)NULL);
    CloseHandle(hProcess);
    return 0;
}
```

Send: Fabric IPC

```
MPI_Send(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

Receive: OS-Specific IPC (Linux)

```
int ipcRecvShareableHandle(int socket, int* fd) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    struct cmsghdr cm;

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        // This will not work on QNX as QNX CMSG_SPACE calls __cmsg_alignbytes
        // And __cmsg_alignbytes is a runtime function instead of compile-time macros
        // char control[CMSG_SPACE(sizeof(int))]
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);
    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    char dummy_buffer[1];
    ssize_t sendResult;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    if ((n = recvmsg(socket, &msg, 0)) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        free(control_un.control);
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
        if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        free(control_un.control);
        return -1;
        }

        memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
        *fd = receivedfd;
    } else {
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

Receive: OS-Specific IPC (WIN)

```
int ipcRecvShareableHandle(HANDLE &handle, HANDLE *shareableHandle) {
    DWORD cbRead;
    ReadFile(handle, shareableHandle, (DWORD)sizeof(*shareableHandles), &cbRead, NULL);
    return 0;
}
```

Receive: Fabric IPC

```
MPI_Recv(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

**Importing Memory Handle**
Again, the user can import handles for OS-specific
IPC or fabric-specific IPC. OS-specific IPC handles can only be used on a
single-node. Fabric-specific handles can be used for single or multi node.

OS-Specific Handle (Linux)

```
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
cuMemImportFromShareableHandle(handle, (void*) &fd, handleType);
```

Fabric Handle

```
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
cuMemImportFromShareableHandle(handle, (void*) &fh, handleType);
```

### 4.16.3.3. Reserve and Map

**Reserving a Virtual Address Range**

Since notions of address and memory are distinct in VMM,
applications must carve out an address range that can hold the
memory allocations made by `cuMemCreate`. The address range reserved must be
at least as large as the sum of the sizes of all the physical memory
allocations the user plans to place in them.

Applications can reserve a virtual address range by passing appropriate
parameters to `cuMemAddressReserve`. The address range obtained will not
have any device or host physical memory associated with it. The reserved
virtual address range can be mapped to memory chunks belonging to any device
in the system, thus providing the application a continuous VA range backed and
mapped by memory belonging to different devices. Applications are expected to
return the virtual address range back to CUDA using `cuMemAddressFree`.
Users must ensure that the entire VA range is unmapped before calling
`cuMemAddressFree`. These functions are conceptually similar to `mmap` and `munmap`
on Linux or `VirtualAlloc` AND `VirtualFree` on Windows. The following
code snippet illustrates the usage for the function:

```
CUdeviceptr ptr;
// `ptr` holds the returned start of virtual address range reserved.
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment
```

**Mapping Memory**

The allocated physical memory and the carved out virtual address space from
the previous two sections represent the memory and address distinction
introduced by the VMM APIs. For the allocated memory to
be useable, the user must map the memory to the address space. The
address range obtained from `cuMemAddressReserve` and the physical
allocation obtained from `cuMemCreate` or `cuMemImportFromShareableHandle`
must be associated with each other by using `cuMemMap`.

Users can associate allocations from multiple devices to reside in contiguous
virtual address ranges as long as they have carved out enough address space.
To decouple the physical allocation and the address range, users must
unmap the address of the mapping with `cuMemUnmap`. Users can map and
unmap memory to the same address range as many times as they want, so long as
they ensure that they don’t attempt to create mappings on VA range
reservations that are already mapped. The following code snippet illustrates
the usage for the function:

```
CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```

### 4.16.3.4. Access Rights

CUDA’s virtual memory management APIs enable applications to explicitly protect
their VA ranges with access control mechanisms. Mapping the allocation to a
region of the address range using `cuMemMap` does not make the address
accessible, and would result in a program crash if accessed by a CUDA kernel.
Users must specifically select access control using the `cuMemSetAccess`
function on source and accessing devices. This allows or restricts access for
specific devices to a mapped address range. The following code snippet
illustrates the usage for the function:

```
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
```

The access control mechanism exposed with VMM allows
users to be explicit about which allocations they want to share with other
peer devices on the system. As specified earlier, `cudaEnablePeerAccess`
forces all prior and future allocations made with `cudaMalloc` to be mapped to the
target peer device. This can be convenient in many cases as user doesn’t have
to worry about tracking the mapping state of every allocation to every device
in the system. But
this approach [has performance implications](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/).
With access control at allocation granularity, VMM
allows peer mappings with minimal overhead.

The `vectorAddMMAP` [sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAddMMAP)
can be used as an example for using the Virtual
Memory Management APIs.

### 4.16.3.5. Releasing the Memory

To release the allocated memory and address space, both the source and target
processes should use cuMemUnmap, cuMemRelease, and cuMemAddressFree functions
in that order. The cuMemUnmap function un-maps a previously mapped memory
region from an address range, effectively detaching the physical memory from
the reserved virtual address space. Next, cuMemRelease deallocates the
physical memory that was previously created, returning it to the system.
Finally, cuMemAddressFree frees a virtual address range that was previously
reserved, making it available for future use. This specific order ensures a
clean and complete deallocation of both the physical memory and the virtual
address space.

```
cuMemUnmap(ptr, size);
cuMemRelease(handle);
cuMemAddressFree(ptr, size);
```

Note

In the OS-specific case, the exported handle must be closed using fclose.
This step is not applicable to the fabric-based case.

## 4.16.4. Multicast Memory Sharing

The [Multicast Object Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST/)
provide a way for the application to create multicast objects and, in combination with the [Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
described above, allow applications to leverage NVLink SHARP on supported NVLink connected GPUs connected with NVSwitch. NVLink SHARP
allows CUDA applications to leverage in-fabric computing to accelerate operations like broadcast and reductions between GPUs connected with
NVSwitch. For this to work, multiple NVLink connected GPUs form a multicast team and each GPU from the team backs up a multicast object with
physical memory. So a multicast team of N GPUs has N physical replicas of a multicast object, each local to one participating GPU.
The [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)
using mappings of multicast objects work with all replicas of the multicast object.

To work with multicast objects, an application needs to

* Query multicast support
* Create a multicast handle with `cuMulticastCreate`.
* Share the multicast handle with all processes that control a GPU which should participate in a multicast team. This works with `cuMemExportToShareableHandle` as described above.
* Add all GPUs that should participate in the multicast team with `cuMulticastAddDevice`.
* For each participating GPU, bind physical memory allocated with `cuMemCreate` as described above to the multicast handle. All devices need to be added to the multicast team before binding memory on any device.
* Reserve an address range, map the multicast handle and set access rights as described above for regular unicast mappings. Unicast and multicast mappings to the same physical memory are possible. See the [Virtual Aliasing Support](#virtual-aliasing-support) section above on how to ensure consistency between multiple mappings to the same physical memory.
* Use the [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/) with the multicast mappings.

The `multi_node_p2p` example in the [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models/) GitHub
repository contains a complete example using fabric memory including multicast objects to leverage NVLink SHARP. Please note that this example is
for developers of libraries like NCCL or NVSHMEM. It shows how higher-level programming models like NVSHMEM work internally within a (multi-node)
NVLink domain. Application developers generally should use the higher-level MPI, NCCL, or NVSHMEM interfaces instead of this API.

### 4.16.4.1. Allocating Multicast Objects

Multicast objects can be created with `cuMulticastCreate`:

```
CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // or on single node CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;

    // Create Multicast Object this has no devices and no physical memory associated yet
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);

    return mcHandle;
}
```

### 4.16.4.2. Add Devices to Multicast Objects

Devices can be added to a multicast team with `cuMulticastAddDevice`:

```
cuMulticastAddDevice(&mcHandle, device);
```

This step needs to be completed on all processes controlling devices that participate in a multicast team before memory on any device is
bound to the multicast object.

### 4.16.4.3. Bind Memory to Multicast Objects

After a multicast object has been created and all participating devices have been added to the multicast object it needs to be backed with
physical memory allocated with `cuMemCreate` for each device:

```
cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);
```

### 4.16.4.4. Use Multicast Mappings

To use multicast mappings in CUDA C++, it is necessary to use the [multimem PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)
with inline PTX:

```
__global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                               float* partial_l2_norm_mc,
                                               unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
                                               const unsigned int expected_count) {
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900

    // atomic reduction to all replicas
    // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
    cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t, cuda::ptx::op_add_t, arrival_counter_mc, n);

    // Need a fence between Multicast (mc) and Unicast (uc) access to the same memory `arrival_counter_uc` and `arrival_counter_mc`:
    // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
    // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
    // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
    cuda::ptx::fence_proxy_alias();

    // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
    // Note: all ranks need to reach another barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
    // arrival of a rank for the next iteration if some other rank is slow.
    cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

#else
    #error "ERROR: multimem instructions require compute capability 9.0 or larger."
#endif

    *l2_norm = std::sqrt(l2_norm_sum);
}
```

## 4.16.5. Advanced Configuration

### 4.16.5.1. Memory Type

VMM also provides a mechanism for applications to allocate special types of memory that
certain devices may support. With `cuMemCreate`,
applications can specify memory type requirements using the
`CUmemAllocationProp::allocFlags` to opt-in to specific memory features.
Applications must ensure that the requested memory type is supported by
the device.

### 4.16.5.2. Compressible Memory

Compressible memory can be used to accelerate accesses to data with
unstructured sparsity and other compressible data patterns. Compression can
save DRAM bandwidth, L2 read bandwidth, and L2 capacity depending on the data.
Applications that want to allocate compressible memory on
devices that support compute data compression can do so by setting
`CUmemAllocationProp::allocFlags::compressionType` to
`CU_MEM_ALLOCATION_COMP_GENERIC`. Users must query if device supports
Compute Data Compression by using
`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`. The following code
snippet illustrates querying compressible memory support
`cuDeviceGetAttribute`.

```
int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
```

On devices that support compute data compression, users must opt in at
allocation time as shown below:

```
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

For a variety of reasons such as limited hardware resources, the allocation may not
have compression attributes. To verify that the flags worked, the user query the properties
of the allocated memory using `cuMemGetAllocationPropertiesFromHandle`.

```
CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}
```

### 4.16.5.3. Virtual Aliasing Support

The virtual memory management APIs provide a way to create multiple virtual
memory mappings or “proxies” to the same allocation using multiple calls to
`cuMemMap` with different virtual addresses. This is called virtual aliasing.
Unless otherwise noted in the PTX ISA, writes to one proxy of the allocation
are considered inconsistent and incoherent with any other proxy of the same
memory until the writing device operation (grid launch, memcpy, memset, and so
on) completes. Grids present on the GPU prior to a writing device operation
but reading after the writing device operation completes are also considered
to have inconsistent and incoherent proxies.

For example, the following snippet is considered undefined, assuming device
pointers A and B are virtual aliases of the same memory allocation:

```
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // Undefined behavior!  *B can take on either
// the previous value or some value in-between.
}
```

The following is defined behavior, assuming these two kernels are ordered
monotonically (by streams or events).

```
__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
// to complete before launching
}

cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
// operation boundaries
foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                            // cudaMemcpy (which both
                                            // read) wait for foo1 (which writes)
                                            // to complete before proceeding
```

If accessing same allocation through different “proxies” is required in the
same kernel, a `fence.proxy.alias` can be used between the two accesses. The
above example can thus be made legal with inline PTX assembly:

```
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  cuda::ptx::fence_proxy_alias();
  printf("%d\n", *B);    // *B == *A == 0x1
}
```

### 4.16.5.4. OS-Specific Handle Details for IPC

With `cuMemCreate`, users have can indicate at
allocation time that they have earmarked a particular allocation for inter-process communication
or graphics interop purposes. Applications can do this
by setting `CUmemAllocationProp::requestedHandleTypes` to a
platform-specific field. On Windows, when
`CUmemAllocationProp::requestedHandleTypes` is set to
`CU_MEM_HANDLE_TYPE_WIN32` applications must also specify an
LPSECURITYATTRIBUTES attribute in
`CUmemAllocationProp::win32HandleMetaData`. This security attribute defines
the scope of which exported allocations may be transferred to other processes.

Users must ensure they query for support of the requested handle type before
attempting to export memory allocated with `cuMemCreate`. The following
code snippet illustrates query for handle type support in a platform-specific
way.

```
int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
```

Users should set the `CUmemAllocationProp::requestedHandleTypes` appropriately as shown below:

```
#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif
```
