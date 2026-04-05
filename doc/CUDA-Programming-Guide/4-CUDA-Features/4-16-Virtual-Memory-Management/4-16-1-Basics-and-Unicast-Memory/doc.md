# 4.16. Virtual Memory Management

In the CUDA programming model, memory allocation calls (such as
`cudaMalloc()`) return a memory address that in GPU memory.
The address can be used with any CUDA API or inside a device
kernel.
Developers can enable peer device access to that memory
allocations by using `cudaEnablePeerAccess`. By doing so,
kernels on different devices can access the same data. However,
all past and future user allocations are also mapped to the target peer device.
This can lead to users unintentionally paying a runtime cost for
mapping all `cudaMalloc` allocations to peer devices. In most
situations, applications communicate by sharing only a few allocations with
another device. It is usually not necessary to map all allocations to all
devices. In addition, extending this approach to multi-node settings becomes
inherently difficult.

CUDA provides a *virtual memory management*
(VMM) API to give developers explicit, low-level control over this process.

Virtual memory allocation, a complex process managed by the operating system
and the Memory Management Unit (MMU), works in two key stages. First, the OS
reserves a contiguous range of virtual addresses for a program without
assigning any physical memory. Then, when the program attempts to use that
memory for the first time, the OS commits the virtual addresses, assigning
physical storage to the virtual pages as needed.

CUDA’s VMM API brings a similar concept to GPU memory management by allowing
developers to explicitly reserve a virtual address range and then later map it
to physical GPU memory. With VMM, applications can specifically choose
certain allocations to be accessible by other devices.

The VMM API lets complex applications to manage
memory more efficiently across multiple GPUs (and CPU cores). By enabling
manual control over memory reservation, mapping, and access permissions, the
VMM API enables advanced techniques like fine-grained data sharing, zero-copy
transfers, and custom memory allocators. The CUDA VMM API expose fine grained
control to the user for managing the GPU memory in applications.

Developers can benefit from the VMM API in several key ways:

* Fine-grained control over virtual and physical memory management, allowing
  allocation and mapping of non-contiguous physical memory chunks to
  contiguous virtual address spaces. This helps reduce GPU memory
  fragmentation and improve memory utilization, especially for large
  workloads like deep neural network training.
* Efficient memory allocation and deallocation by separating the reservation
  of virtual address space from the physical memory allocation. Developers
  can reserve large virtual memory regions and map physical memory on demand
  without costly memory copies or reallocations, leading to performance
  improvements in dynamic data structures and variable-sized memory
  allocations.
* The ability to grow GPU memory allocations dynamically without needing to
  copy and reallocate all data, similar to how `realloc` or `std::vector` works
  in CPU memory management. This supports more flexible and efficient GPU
  memory use patterns.
* Enhancements to developer productivity and application performance by
  providing low-level APIs that allow building sophisticated memory
  allocators and cache management systems, such as dynamically managing
  key-value caches in large language models, improving throughput and
  latency.
* The CUDA VMM API is highly valuable in distributed multi-GPU settings as
  it enables efficient memory sharing and access across multiple GPUs. By
  decoupling virtual addresses from physical memory, the API allows
  developers to create a unified virtual address space where data can be
  dynamically mapped to different GPUs. This optimizes memory usage and
  reduces data transfer overhead. For instance, NVIDIA’s libraries like NCCL,
  and NVShmem actively uses VMM.

In summary, the CUDA VMM API gives developers advanced tools for fine-tuned,
efficient, flexible, and scalable GPU memory management beyond traditional
malloc-like abstractions, which is important for high-performance and
large-memory applications

Note

The suite of APIs described in this section require a system that
supports UVA. See The [Virtual Memory Management APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html).

## 4.16.1. Preliminaries

### 4.16.1.1. Definitions

**Fabric Memory:**
Fabric memory refers to memory that is accessible over a high-speed
interconnect fabric such as NVIDIA’s NVLink and NVSwitch. This fabric provides
a memory coherence and high-bandwidth communication layer between multiple
GPUs or nodes, enabling them to share memory efficiently as if the memory is
attached to a unified fabric rather than isolated on individual devices.

CUDA 12.4 and later have a VMM allocation handle type
`CU_MEM_HANDLE_TYPE_FABRIC`. On supported platforms and provided the NVIDIA
IMEX daemon is running, this allocation handle type enables sharing allocations
not only intra-node with any communication mechanism, e.g. MPI, but also inter-node.
This allows GPUs in a multi-node NVLink system to map the memory of all
other GPUs part of the same NVLink fabric even if they are in different nodes.

**Memory Handles:**
In VMM, handles are opaque identifiers that represent physical memory
allocations. These handles are central to managing memory in the low-level
CUDA VMM API. They enable flexible control over physical memory objects that
can be mapped into virtual address spaces.
A handle uniquely identifies a physical memory allocation.
Handles serve as an abstract reference to memory resources without exposing
direct pointers. Handles allow operations like exporting and importing memory
across processes or devices, facilitating memory sharing and virtualization.

**IMEX Channels:**
The name IMEX stands for *internode memory exchange* and is part of NVIDIA’s
solution for GPU-to-GPU communication across different nodes.
IMEX channels are a GPU driver feature that provides user-based memory
isolation in multi-user or multi-node environments within an IMEX domain.
IMEX channels serve as a security and isolation mechanism.

IMEX channels are directly related to the fabric handle and has to be enabled
in multi-node GPU communication. When a GPU allocates
memory and wants to make it accessible to a GPU on a different node, it first
needs to export a handle to that memory. The IMEX channel is used during
this export process to generate a secure fabric handle that can only be
imported by a remote process with the correct channel access.

**Unicast Memory Access:**
Unicast memory access in the context of VMM API refers to the
controlled, direct mapping and access of physical memory to a unique virtual
address range by a specific device or process. Instead of broadcasting access
to multiple devices, unicast memory access means that a particular GPU device
is granted explicit read/write permissions to a reserved virtual address range
that maps to a physical memory allocation.

**Multicast Memory Access:**
Multicast memory access in the context of the VMM API refers to the capability
for a single physical memory allocation or region to be mapped
simultaneously to multiple devices’ virtual address spaces using a multicast
mechanism. This allows data to be efficiently shared in a one-to-many fashion
across multiple GPUs, reducing redundant data transfers and
improving communication efficiency.
NVIDIA’s CUDA VMM API supports creating a multicast object that binds together
physical memory allocations from multiple devices.

### 4.16.1.2. Query for Support

Applications should query for feature support before attempting to use
them, as their availability can vary depending on the GPU architecture, driver
version, and specific software libraries being used. The following sections
detail how to programmatically check for the necessary support.

**VMM Support**
Before attempting to use VMM APIs, applications must
ensure that the devices they want to use support CUDA virtual memory
management. The following code sample shows querying for VMM support:

```
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management
}
```

**Fabric Memory Support:**
Before attempting to use fabric memory, applications must ensure that the
devices they want to use support fabric memory. The following code
sample shows querying for fabric memory support:

```
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` supports Fabric Memory
}
```

Aside from using `CU_MEM_HANDLE_TYPE_FABRIC` as handle type and not
requiring OS native mechanisms for inter-process communication to exchange
sharable handles, there is no difference in using fabric memory compared to
other allocation handle types.

**IMEX Channels Support**
Within an IMEX domain, IMEX channels enable secure memory sharing in
multi-user environments. The NVIDIA driver implements this by creating a
character device, `nvidia-caps-imex-channels`. To use fabric handle-based
sharing, users should verify two things:

* First, applications must verify that this device exists under
  /proc/devices:

```
# cat /proc/devices | grep nvidia
195 nvidia
195 nvidiactl
234 nvidia-caps-imex-channels
509 nvidia-nvswitch

The nvidia-caps-imex-channels device should have a major number (e.g., 234).
```

* Second, for two CUDA processes (an exporter and an importer) to share memory,
  they must both have access to the same IMEX channel file. These files, such
  as /dev/nvidia-caps-imex-channels/channel0, are nodes that represent
  individual IMEX channels. System administrators must create these files, for
  example, using the mknod() command.

```
# mknod /dev/nvidia-caps-imex-channels/channelN c <major_number> 0

This command creates channelN using the major number obtained from
/proc/devices.
```

Note

By default, the driver can create channel0
if the NVreg\_CreateImexChannel0 module parameter is specified.

**Multicast Object Support:**
Before attempting to use multicast objects, applications must ensure that the
devices they want to use support them. The following code sample
shows querying for multicast object support:

```
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` supports Multicast Objects
}
```

## 4.16.2. API Overview

The VMM API provides developers
with granular control over virtual memory management. VMM, being a very low-level API,
requires use of the [CUDA Driver API](../03-advanced/driver-api.html#driver-api) directly.
This versatile API can be used in both single-node and multi-node
environments.

To use VMM effectively,
developers must have a solid grasp of a few key concepts in memory management:
- Knowledge of the operating system’s virtual memory fundamentals, including how it handles pages and address spaces
- An understanding of memory hierarchy and hardware characteristics is necessary
- Familiarity with inter-process communication (IPC) methods, such as sockets or message passing,
- A basic knowledge of security for memory access rights

![VMM Usage Overview Diagram](img/Figure-52-VMM-Usage-Overview-This-diagram-outlines-the-series-of-steps-required-for-VMM-utilization-The-process-begins-b.png)

*Figure 52 VMM Usage Overview.
This diagram outlines the series of steps required for VMM utilization.
The process begins by evaluating the environmental setup. Based on this
assessment, the user must make a critical initial decision: whether to
utilize fabric memory handles or OS-specific handles.
A distinct series of subsequent steps must be taken based on the initial
handle choice. However, the final memory management operations—specifically
mapping, reserving, and setting access rights of the allocated memory—are
identical to the type of handle that was selected.*

The VMM API workflow involves a sequence of steps for memory management, with
a key focus on sharing memory between different devices or processes.
Initially, a developer must allocate physical memory on the source device. To
facilitate sharing, the VMM API utilizes handles to convey necessary
information to the target device or process. The user must export a handle for
sharing, which can be either an OS-specific handle or a fabric-specific
handle. OS-specific handles are limited to inter-process communication on a
single node, while fabric-specific handles offer greater versatility and can
be used in both single-node and multi-node environments. It’s important to
note that using fabric-specific handles requires the enablement of IMEX
channels.

Once the handle is exported, it must be shared with the receiving process or
processes using an inter-process communication protocol, with the choice
of method left to the developer. The receiving process then uses the VMM API
to import the handle. After the handle has been successfully exported, shared,
and imported, both the source and target processes must reserve virtual
address space where the allocated physical memory will be mapped. The final
step is to set the memory access rights for each device, ensuring proper
permissions are established. This entire process, including both handle
approaches, is further detailed in the accompanying figure.

## 4.16.3. Unicast Memory Sharing

Sharing GPU memory can happen on one machine with multiple GPUs or across a
network of machines. The process follows these steps:

* Allocate and Export: A CUDA program on one GPU allocates memory and gets a
  sharable handle for it.
* Share and Import: The handle is then sent to other programs on the node
  using IPC, MPI, or NCCL etc. In the receiving GPUs, the CUDA driver imports
  the handle, creates the necessary memory objects
* Reserve and Map: The driver creates a mapping from the program’s Virtual
  Address (VA) to the GPU’s Physical Address (PA) to its network Fabric
  Address (FA).
* Access Rights: Setting access rights for the allocation.
* Releasing the Memory: Freeing all allocations when program ends its execution.

![Unicast Memory Sharing Example](img/Figure-53-Unicast-Memory-Sharing-Example.png)

*Figure 53 Unicast Memory Sharing Example*

### 4.16.3.1. Allocate and Export

**Allocating Physical Memory**
The first step in memory allocation using virtual memory management APIs is to
create a physical memory chunk that will provide a backing for the allocation.
In order to allocate physical memory, applications must use the
`cuMemCreate` API. The allocation created by this function does not have any
device or host mappings. The function argument
`CUmemGenericAllocationHandle` describes the properties of the memory to
allocate such as the location of the allocation, if the allocation is going to
be shared to another process (or graphics APIs), or the physical
attributes of the memory to be allocated. Users must ensure the requested
allocation’s size is aligned to appropriate granularity. Information
regarding an allocation’s granularity requirements can be queried using
`cuMemGetAllocationGranularity`.

OS-Specific Handle (Linux)

```
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

Fabric Handle

```
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}
```

Note

The memory allocated by `cuMemCreate` is referenced by the
`CUmemGenericAllocationHandle` it returns. Note that this reference is
not a pointer and its memory is not accessible yet.

Note

Properties of the allocation handle can be queried using
`cuMemGetAllocationPropertiesFromHandle`.

**Exporting Memory Handle**
The CUDA virtual memory management API expose a new mechanism for interprocess
communication using handles to exchange necessary information about the
allocation and physical address space. One can export handles for OS-specific
IPC or fabric-specific IPC. OS-specific IPC handles can only be used on a
single-node setup. Fabric-specific handles can be used on a single or multi
node setups.

OS-Specific Handle (Linux)

```
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
int fd;
cuMemExportToShareableHandle(&fd, handle, handleType, 0);
```

Fabric Handle

```
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
CUmemFabricHandle fh;
cuMemExportToShareableHandle(&fh, handle, handleType, 0);
```

Note

OS-specific handles require all processes to be part of the same OS.

Note

Fabric-specific handles require IMEX channels to be enabled by sysadmin.

The [memMapIpcDrv](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv/)
sample can be used as an example for using IPC with VMM allocations.

