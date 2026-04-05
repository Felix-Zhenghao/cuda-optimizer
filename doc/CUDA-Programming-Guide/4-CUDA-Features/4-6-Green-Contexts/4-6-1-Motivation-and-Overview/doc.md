# 4.6. Green Contexts

A green context (GC) is a lightweight context associated, from its creation, with a set of specific GPU resources.
Users can partition GPU resources, currently streaming multiprocessors (SMs) and work queues (WQs), during green context creation, so that GPU work targeting a green context can only use its provisioned SMs and work queues.
Doing so can be beneficial in reducing, or better controlling, interference due to use of common resources.
An application can have multiple green contexts.

Using green contexts does not require any GPU code (kernel) changes, just small host-side changes (e.g., green context creation and stream creation for this green context).
The green context functionality can be useful in various scenarios. For example, it can help ensure some SMs are always available for a latency-sensitive kernel to start executing, assuming no other
constraints, or provide a quick way to test the effect of using fewer SMs without any kernel modifications.

Green context support first became available via the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS).
Starting from CUDA 13.1, contexts are exposed in the CUDA runtime via the execution context (EC) abstraction.
Currently, an execution context can correspond to either the primary context (the context runtime API users have always implicitly interacted with) or a green context.
This section will use the terms *execution context* and *green context* interchangeably when referring to a green context.

With the runtime exposure of green contexts, using the CUDA runtime API directly is strongly recommended. This section will also solely use the CUDA runtime API.

The remaining of this section is organized as follows:
[Section 4.6.1](#green-contexts-motivation) provides a motivating example, [Section 4.6.2](#green-contexts-ease-of-use) highlights ease of use, and [Section 4.6.3](#green-contexts-device-resource-and-desc) presents the device resource and resource descriptor structs.
[Section 4.6.4](#green-contexts-creation-example) explains how to create a green context, [Section 4.6.5](#green-contexts-launching-work) how to launch work that targets it,
and [Section 4.6.6](#green-contexts-apis) highlights some additional green context APIs. Finally, [Section 4.6.7](#green-contexts-example) wraps up with an example.

## 4.6.1. Motivation / When to Use

When launching a CUDA kernel, the user has no direct control over the number of SMs that kernel will execute on. One can only indirectly influence this
by changing the kernel’s launch geometry or anything that can affect the kernel’s maximum number of active thread blocks per SM.
Additionally, when multiple kernels execute in parallel on the GPU (kernels running on different CUDA streams or as part of a CUDA graph), they may
also contend for the same SM resources.

There are, however, use cases where the user needs to ensure there are always GPU resources available for latency-sensitive work to start, and thus complete, as soon as possible.
Green contexts provide a way towards that by partitioning SM resources, so a given green context can only use specific SMs (the ones provisioned during its creation).

[Figure 42](#id2) illustrates such an example. Assume an application where two independent kernels A and B run on two different non-blocking CUDA streams.
Kernel A is launched first and starts executing occupying all available SM resources. When, later in time, latency-sensitive kernel B is launched, no SM resources are available.
As a result, kernel B can only start executing once kernel A ramps down, i.e., once thread blocks from kernel A finish executing.
The first graph illustrates this scenario where critical work B gets delayed. The y-axis shows the percentage of SMs occupied and x-axis depicts time.

![Green Contexts Motivation](img/Figure-42-Motivation-GCsâ-static-resource-partitioning-enables-latency-sensitive-work-B-to-start-and-complete-sooner.png)

*Figure 42 Motivation: GCs’ static resource partitioning enables latency-sensitive work B to start and complete sooner*

Using green contexts, one could partition the GPU’s SMs, so that green context A, targeted by kernel A, has access to some SMs of the GPU, while green context B, targeted by kernel B, has access to the remaining SMs.
In this setting, kernel A can only use the SMs provisioned for green context A, irrespective of its launch configuration. As a result, when critical kernel B gets launched, it is guaranteed that there will be available SMs for it to start executing immediately, barring any other resource constraints. As the second graph in [Figure 42](#id2) illustrates, even though the duration of kernel A may increase, latency-sensitive work B will no longer be delayed due to unavailable SMs. The figure shows that green context A is provisioned with an SM count equivalent to 80% SMs of the GPU for illustration purposes.

This behavior can be achieved without any code modifications to kernels A and B. One simply needs to ensure they are launched on CUDA streams belonging to the appropriate green contexts. The number of SMs each green context will have access to should be decided by the user during green context creation on a per case basis.

**Work Queues**:

Streaming multiprocessors are one resource type that can be provisioned for a green context. Another resource type is work queues.
Think of a workqueue as a black-box resource abstraction, which can also influence GPU work execution concurrency, along with other factors.
If independent GPU work tasks (e.g., kernels submitted on different CUDA streams) map to the same workqueue, a false dependence between these tasks may be introduced,
which can lead to their serialized execution.
The user can influence the upper limit of work queues on the GPU via the `CUDA_DEVICE_MAX_CONNECTIONS` environment variable (see [Section 5.2](../05-appendices/environment-variables.html#cuda-environment-variables), [Section 3.1](../03-advanced/advanced-host-programming.html#advanced-apis-and-features)).

Building on top of the previous example, assume work B maps to the same workqueue as work A.
In that case, even if SM resources are available (green contexts case), work B may still need to wait for work A to complete in its entirety.
Similar to SMs, the user has no direct control over the specific work queues that may be used under the hood.
But green contexts allow the user to express the maximum concurrency they would expect in terms of expected number of concurrent stream-ordered workloads.
The driver can then use this value as a hint to try to prevent work from different execution contexts from using the same workqueue(s), thus preventing unwanted interference across execution contexts.

Attention

Even when different SM resources and work queues are provisioned per green context, concurrent execution of independent GPU work is not guaranteed.
It is best to think of all the techniques described under the [Green Contexts](#green-contexts) section as removing factors which can prevent concurrent execution (i.e., reducing potential interference).

**Green Contexts versus MIG or MPS**

For completeness, this section briefly compares green contexts with two other resource partitioning mechanisms:
[MIG (Multi-Instance GPU)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) and
[MPS (Multi-Process Service)](https://docs.nvidia.com/deploy/mps/index.html).

MIG statically partitions a MIG-supported GPU into multiple MIG instances (“smaller GPUs”).
This partitioning has to happen before the launch of an application, and different applications can use different MIG instances.
Using MIG can be beneficial for users whose applications consistently underutilize the available GPU resources; an issue
more pronounced as GPUs get bigger. With MIG, users can run these different applications on different MIG instances, thus improving GPU utilization.
MIG can be attractive for cloud service providers (CSPs) not only for the increased GPU utilization for such applications, but also
for the quality of service (QoS) and isolation it can provide across clients running on different MIG instances. Please refer to the MIG documentation linked above for more details.

But using MIG cannot address the problematic scenario described earlier, where critical work B is delayed because all SM resources are occupied by other GPU work from the same application.
This issue can still exist for an application running on a single MIG instance.
To address it, one can use green contexts alongside MIG. In that case, the SM resources available for partitioning would be the resources of the given MIG instance.

MPS primarily targets different processes (e.g., MPI programs), allowing them to run on the GPU at the same time without time-slicing.
It requires an MPS daemon to be running before the application is launched.
By default, MPS clients will contend for all available SM resources of the GPU or the MIG instance they are running on.
In this multiple client processes setting, MPS can support dynamic partitioning of SM resources, using the active thread percentage option, which places an upper limit on the percentage of SMs an MPS client process can use.
Unlike green contexts, the active thread percentage partitioning happens with MPS at the process level, and the percentage is typically specified by an environment variable before the application is launched.
The MPS active thread percentage signifies that a given client application cannot use more than x% of a GPU’s SMs, let that be N SMs. However, these SMs can be any N SMs of the GPU, which can also vary over time.
On the other hand, a green context provisioned with N SMs during its creation can only use these specific N SMs.

Starting with CUDA 13.1, MPS also supports static partitioning, if it is explicitly enabled when starting the MPS control daemon. With static partitioning, the user has to specify the static partition an MPS client process can use, when the application is launched. Dynamic sharing with active thread percentage is no longer applicable in that case.
A key difference between MPS in static partitioning mode and green contexts is that MPS targets different processes, while green contexts is applicable within a single process too.
Also, contrary to green contexts, MPS with static partitioning does not allow oversubscription of SM resources.

With MPS, programmatic partitioning of SM resources is also possible for a CUDA context created via the `cuCtxCreate` driver API, with execution affinity.
This programmatic partitioning allows different client CUDA contexts from one or more processes to each use up to a specified number of SMs.
As with the active thread percentage partitioning, these SMs can be any SMs of the GPU and can vary over time, unlike the green contexts case.
This option is possible even under the presence of static MPS partitioning.
Please note that creating a green context is much more lightweight in comparison to an MPS context, as many underlying structures are owned by the primary context and thus shared.

## 4.6.2. Green Contexts: Ease of use

To highlight how easy it is to use green contexts, assume you have the following code snippet that creates two CUDA streams and then calls a function that launches kernels via `<<<>>>` on these CUDA streams.
As discussed earlier, other than changing the kernels’ launch geometries, one cannot influence how many SMs these kernels can use.

```
int gpu_device_index = 0; // GPU ordinal
CUDA_CHECK(cudaSetDevice(gpu_device_index));

cudaStream_t strm1, strm2;
CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

// No control over how many SMs kernel(s) running on each stream can use
code_that_launches_kernels_on_streams(strm1, strm2); // what is abstracted in this function + the kernels is the vast majority of your code

// cleanup code not shown
```

Starting with CUDA 13.1, one can control the number of SMs a given kernel can have access to, using green contexts.
The code snippet below shows how easy it is to do that. With a few extra lines and without any kernel modifications,
you can control the SMs resources kernel(s) launched on these different streams can use.

```
int gpu_device_index = 0; // GPU ordinal
CUDA_CHECK(cudaSetDevice(gpu_device_index));

/* ------------------ Code required to create green contexts --------------------------- */

// Get all available GPU SM resources
cudaDevResource initial_GPU_SM_resources {};
CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));

// Split SM resources. This example creates one group with 16 SMs and one with 8. Assuming your GPU has >= 24 SMs
cudaDevSmResource result[2] {{}, {}};
cudaDevSmResourceGroupParams group_params[2] =  {
        {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
        {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
CUDA_CHECK(cudaDevSmResourceSplit(&result[0], 2, &initial_GPU_SM_resources, nullptr, 0, &group_params[0]));

// Generate resource descriptors for each resource
cudaDevResourceDesc_t resource_desc1 {};
cudaDevResourceDesc_t resource_desc2 {};
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc1, &result[0], 1));
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc2, &result[1], 1));

// Create green contexts
cudaExecutionContext_t my_green_ctx1 {};
cudaExecutionContext_t my_green_ctx2 {};
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx1, resource_desc1, gpu_device_index, 0));
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx2, resource_desc2, gpu_device_index, 0));

/* ------------------ Modified code --------------------------- */

// You just need to use a different CUDA API to create the streams
cudaStream_t strm1, strm2;
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm1, my_green_ctx1, cudaStreamDefault, 0));
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm2, my_green_ctx2, cudaStreamDefault, 0));

/* ------------------ Unchanged code --------------------------- */

// No need to modify any code in this function or in your kernel(s).
// Reminder: what is abstracted in this function + kernels is the vast majority of your code
// Now kernel(s) running on stream strm1 will use at most 16 SMs and kernel(s) on strm2 at most 8 SMs.
code_that_launches_kernels_on_streams(strm1, strm2);

// cleanup code not shown
```

Various execution context APIs, some of which were shown in the previous example, take an explicit `cudaExecutionContext_t` handle and thus ignore the context that is current to the calling thread.
Until now, CUDA runtime users who did not use the driver API would by default only interact with the primary context that is implicitly set as current to a thread via `cudaSetDevice()`.
This shift to explicit context-based programming provides easier to understand semantics and can have additional benefits compared to the previous implicit context-based programming that relied on thread-local state (TLS).

The following sections will explain all the steps shown in the previous code snippet in detail.

## 4.6.3. Green Contexts: Device Resource and Resource Descriptor

At the heart of a green context is a device resource (`cudaDevResource`) tied to a specific GPU device.
Resources can be combined and encapsulated into a descriptor (`cudaDevResourceDesc_t`).
A green context only has access to the resources encapsulated into the descriptor used for its creation.

Currently the `cudaDevResource` data structure is defined as:

```
struct {
     enum cudaDevResourceType type;
     union {
         struct cudaDevSmResource sm;
         struct cudaDevWorkqueueConfigResource wqConfig;
         struct cudaDevWorkqueueResource wq;
     };
 };
```

The supported valid resource types are `cudaDevResourceTypeSm`, `cudaDevResourceTypeWorkqueueConfig` and `cudaDevResourceTypeWorkqueue`, while `cudaDevResourceTypeInvalid` identifies an invalid resource type.

A valid device resource can be associated with:

* a specific set of streaming multiprocessors (SMs) (resource type `cudaDevResourceTypeSm`),
* a specific workqueue configuration (resource type `cudaDevResourceTypeWorkqueueConfig`) or
* a pre-existing workqueue resource (resource type `cudaDevResourceTypeWorkqueue`).

One can query if a given execution context or CUDA stream is associated with a `cudaDevResource` resource of a given type,
using the `cudaExecutionCtxGetDevResource` and `cudaStreamGetDevResource` APIs respectively.
Being associated with different types of device resources (e.g., SMs and work queues) is also possible for an execution context,
while a stream can only be associated with an SM-type resource.

A given GPU device has, by default, all three device resource types: an SM-type resource encompassing all the SMs of the GPU, a workqueue configuration resource encompassing all available work queues
and its corresponding workqueue resource. These resources can be retrieved via the `cudaDeviceGetDevResource` API.

**Overview of relevant device resource structs**

The different resource type structs have fields that are set either explicitly by the user or by a relevant CUDA API call.
It is recommended to zero-initialize all device resource structs.

* An SM-type device resource (`cudaDevSmResource`) has the following relevant fields:

  + `unsigned int smCount`: number of SMs available in this resource
  + `unsigned int minSmPartitionSize`: minimum SM count required to partition this resource
  + `unsigned int smCoscheduledAlignment`: number of SMs in the resource guaranteed to be co-scheduled on the same GPU processing cluster, which is relevant for thread block clusters. `smCount` is a multiple of this value when `flags` is zero.
  + `unsigned int flags`: supported flags are 0 (default) and `cudaDevSmResourceGroupBackfill` (see `cudaDevSmResourceGroup` flags).

  The above fields will be set via either the appropriate split API (`cudaDevSmResourceSplitByCount` or `cudaDevSmResourceSplit`) used to create this SM-type resource or will be populated by the `cudaDeviceGetDevResource` API which retrieves the SM resources of a given GPU device. These fields should never be set directly by the user. See next section for more details.
* A workqueue configuration device resource (`cudaDevWorkqueueConfigResource`) has the following relevant fields:

  + `int device`: the device on which the workqueue resources are available
  + `unsigned int wqConcurrencyLimit`: the number of stream-ordered workloads expected to avoid false dependencies
  + `enum cudaDevWorkqueueConfigScope sharingScope`: the sharing scope for the workqueue resources. Supported values are: `cudaDevWorkqueueConfigScopeDeviceCtx` (default) and `cudaDevWorkqueueConfigScopeGreenCtxBalanced`. With the default option, all workqueue resources are shared across all contexts, while with the balanced option the driver tries to use non-overlapping workqueue resources across green contexts wherever possible, using the user-specified `wqConcurrencyLimit` as a hint.

  These fields need to be set by the user. There is no CUDA API similar to the split APIs that generates a workqueue configuration resource, with the exception of the workqueue configuration resource populated by the `cudaDeviceGetDevResource` API. That API can retrieve the workqueue configuration resources of a given GPU device.
* Finally, a pre-existing workqueue resource (`cudaDevResourceTypeWorkqueue`) has no fields that can be set by the user. As with the other resource types, `cudaDevGetDevResource` can retrieve the pre-existing workqueue resource of a given GPU device.

## 4.6.4. Green Context Creation Example

There are four main steps involved in green context creation:

* Step 1: Start with an initial set of resources, e.g., by fetching the available resources of the GPU
* Step 2: Partition the SM resources into one or more partitions (using one of the available split APIs).
* Step 3: Create a resource descriptor combining, if needed, different resources
* Step 4: Create a green context from the descriptor, provisioning its resources

After the green context has been created, you can create CUDA streams belonging to that green context.
GPU work subsequently launched on such a stream, such as a kernel launched via `<<< >>>`, will only have access to this green context’s provisioned resources.
Libraries can also easily leverage green contexts, as long as the user passes a stream belonging to a green context to them.
See [Green Contexts - Launching work](#green-contexts-launching-work) for more details.

### 4.6.4.1. Step 1: Get available GPU resources

The first step in green context creation is to get the available device resources and populate the `cudaDevResource` struct(s).
There are currently three possible starting points: a device, an execution context or a CUDA stream.

The relevant CUDA runtime API function signatures are listed below:

* For a **device**: `cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type)`
* For an **execution context**: `cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type)`
* For a **stream**: `cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type)`

All valid `cudaDevResourceType` types are permitted for each of these APIs, with the exception of `cudaStreamGetDevResource` which only supports an SM-type resource.

Usually, the starting point will be a GPU device. The code snippet below shows how to get the available SM resources of a given GPU device.
After a successful `cudaDeviceGetDevResource` call, the user can review the number of SMs available in this resource.

```
int current_device = 0; // assume device ordinal of 0
CUDA_CHECK(cudaSetDevice(current_device));

cudaDevResource initial_SM_resources = {};
CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                   &initial_SM_resources /* device resource to populate */,
                                   cudaDevResourceTypeSm /* resource type*/));

std::cout << "Initial SM resources: " << initial_SM_resources.sm.smCount << " SMs" << std::endl; // number of available SMs

// Special fields relevant for partitioning (see Step 3 below)
std::cout << "Min. SM partition size: " <<  initial_SM_resources.sm.minSmPartitionSize << " SMs" << std::endl;
std::cout << "SM co-scheduled alignment: " <<  initial_SM_resources.sm.smCoscheduledAlignment << " SMs" << std::endl;
```

One can also get the available workqueue config. resources, as shown in the code snippet below.

```
int current_device = 0; // assume device ordinal of 0
CUDA_CHECK(cudaSetDevice(current_device));

cudaDevResource initial_WQ_config_resources = {};
CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU device */,
                                   &initial_WQ_config_resources /* device resource to populate */,
                                   cudaDevResourceTypeWorkqueueConfig /* resource type*/));

std::cout << "Initial WQ config. resources: " << std::endl;
std::cout << "  - WQ concurrency limit: " << initial_WQ_config_resources.wqConfig.wqConcurrencyLimit << std::endl;
std::cout << "  - WQ sharing scope: " << initial_WQ_config_resources.wqConfig.sharingScope << std::endl;
```

After a successful `cudaDeviceGetDevResource` call, the user can review the `wqConcurrencyLimit` for this resource.
When the starting point is a GPU device, the `wqConcurrencyLimit` will match the value of `CUDA_DEVICE_MAX_CONNECTIONS` environment variable or its default value.

