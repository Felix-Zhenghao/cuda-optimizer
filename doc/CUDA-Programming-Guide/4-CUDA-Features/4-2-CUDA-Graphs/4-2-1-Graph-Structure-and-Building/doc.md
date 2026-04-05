# 4.2. CUDA Graphs

CUDA Graphs present another model for work submission in CUDA. A graph is a series of operations such as kernel launches, data movement, etc., connected by dependencies, which is defined separately from its execution. This allows a graph to be defined once and then launched repeatedly. Separating out the definition of a graph from its execution enables a number of optimizations: first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams.

To see the optimizations possible with graphs, consider what happens in a stream: when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU. These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time. By creating a CUDA graph that encompasses a workflow that will be launched many times, these overhead costs can be paid once for the entire graph during instantiation, and the graph itself can then be launched repeatedly with very little overhead.

## 4.2.1. Graph Structure

An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.

An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.

### 4.2.1.1. Node Types

A graph node can be one of:

* kernel
* CPU function call
* memory copy
* memset
* empty node
* waiting on a [CUDA Event](../02-basics/asynchronous-execution.html#cuda-events)
* recording a [CUDA Event](../02-basics/asynchronous-execution.html#cuda-events)
* signalling an [external semaphore](graphics-interop.html#external-resource-interoperability)
* waiting on an [external semaphore](graphics-interop.html#external-resource-interoperability)
* [conditional node](#cuda-graphs-conditional-graph-nodes)
* [memory node](#cuda-graphs-graph-memory-nodes)
* child graph: To execute a separate nested graph, as shown in the following figure.

![Child Graph Example](img/Figure-21-Child-Graph-Example.png)

*Figure 21 Child Graph Example*

### 4.2.1.2. Edge Data

CUDA 12.3 introduced edge data on CUDA Graphs. At this time, the only use for non-default edge data is enabling
[Programmatic Dependent Launch](programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization).

Generally speaking, edge data modifies a dependency specified by an edge and consists of three parts:
an outgoing port, an incoming port, and a type. An outgoing port specifies when an associated edge is triggered. An incoming port
specifies what portion of a node is dependent on an associated edge. A type modifies the relation between the endpoints.

Port values are specific to node type and direction, and edge types may be restricted to specific node types. In all cases,
zero-initialized edge data represents default behavior. Outgoing port 0 waits on an entire task, incoming port 0 blocks an
entire task, and edge type 0 is associated with a full dependency with memory synchronizing behavior.

Edge data is optionally specified in various graph APIs via a parallel array to the associated nodes. If it is omitted as an
input parameter, zero-initialized data is used. If it is omitted as an output (query) parameter, the API accepts this if the
edge data being ignored is all zero-initialized, and returns `cudaErrorLossyQuery` if the call would discard information.

Edge data is also available in some stream capture APIs: `cudaStreamBeginCaptureToGraph()`, `cudaStreamGetCaptureInfo()`,
and `cudaStreamUpdateCaptureDependencies()`. In these cases, there is not yet a downstream node. The data is associated with
a dangling edge (half edge) which will either be connected to a future captured node or discarded at termination of stream capture.
Note that some edge types do not wait on full completion of the upstream node. These edges are ignored when considering if a
stream capture has been fully rejoined to the origin stream, and cannot be discarded at the end of capture. See [Stream Capture](#cuda-graphs-creating-a-graph-using-stream-capture).

No node types define additional incoming ports, and only kernel nodes define additional outgoing ports. There is
one non-default dependency type, `cudaGraphDependencyTypeProgrammatic`, which is used to enable [Programmatic Dependent Launch](programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) between two kernel nodes.

## 4.2.2. Building and Running Graphs

Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution.

* During the **definition** or **creation** phase, a program creates a description of the operations in the graph along with the dependencies between them.
* **Instantiation** takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an *executable graph.*
* An **executable** graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.

### 4.2.2.1. Graph Creation

Graphs can be created via two mechanisms: using the explicit Graph API and via stream capture.

#### 4.2.2.1.1. Graph APIs

The following is an example (omitting declarations and other boilerplate code) of creating the below graph. Note the use of `cudaGraphCreate()` to create the graph and `cudaGraphAddNode()` to add the kernel nodes and their dependencies. [The CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html) lists all the functions available for adding nodes and dependencies.

![Creating a Graph Using Graph APIs Example](img/Figure-22-Creating-a-Graph-Using-Graph-APIs-Example.png)

*Figure 22 Creating a Graph Using Graph APIs Example*

```
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// Create the nodes and their dependencies
cudaGraphNode_t nodes[4];
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
kParams.kernel.func         = (void *)kernelName;
kParams.kernel.gridDim.x    = kParams.kernel.gridDim.y  = kParams.kernel.gridDim.z  = 1;
kParams.kernel.blockDim.x   = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;

cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 2, &kParams);
```

The example above shows four kernel nodes with dependencies between them to illustrate the creation of a very simple graph. In a typical user application there would also need to be nodes added for memory operations, such as `cudaGraphAddMemcpyNode()` and the like. For full reference of all graph API functions to add nodes, see [The CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html) .

#### 4.2.2.1.2. Stream Capture

Stream capture provides a mechanism to create a graph from existing stream-based APIs. A section of code which launches work into streams, including existing code, can be bracketed with calls to `cudaStreamBeginCapture()` and `cudaStreamEndCapture()`. See below.

```
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```

A call to `cudaStreamBeginCapture()` places a stream in capture mode. When a stream is being captured, work launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built up. This graph is then returned by calling `cudaStreamEndCapture()`, which also ends capture mode for the stream. A graph which is actively being constructed by stream capture is referred to as a *capture graph.*

Stream capture can be used on any CUDA stream except `cudaStreamLegacy` (the “NULL stream”). Note that it *can* be used on `cudaStreamPerThread`. If a program is using the legacy stream, it may be possible to redefine stream 0 to be the per-thread stream with no functional change. See [Blocking and non-blocking streams and the default stream](../02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream).

Whether a stream is being captured can be queried with `cudaStreamIsCapturing()`.

Work can be captured to an existing graph using `cudaStreamBeginCaptureToGraph()`. Instead of capturing to an internal graph, work is captured to a graph provided by the user.

##### 4.2.2.1.2.1. Cross-stream Dependencies and Events

Stream capture can handle cross-stream dependencies expressed with `cudaEventRecord()` and `cudaStreamWaitEvent()`, provided the event being waited upon was recorded into the same capture graph.

When an event is recorded in a stream that is in capture mode, it results in a *captured event.* A captured event represents a set of nodes in a capture graph.

When a captured event is waited on by a stream, it places the stream in capture mode if it is not already, and the next item in the stream will have additional dependencies on the nodes in the captured event. The two streams are then being captured to the same capture graph.

When cross-stream dependencies are present in stream capture, `cudaStreamEndCapture()` must still be called in the same stream where `cudaStreamBeginCapture()` was called; this is the *origin stream*. Any other streams which are being captured to the same capture graph, due to event-based dependencies, must also be joined back to the origin stream. This is illustrated below. All streams being captured to the same capture graph are taken out of capture mode upon `cudaStreamEndCapture()`. Failure to rejoin to the origin stream will result in failure of the overall capture operation.

```
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

// stream1 and stream2 no longer in capture mode
```

The graph returned by the above code is shown in [Figure 22](#cuda-graphs-creating-a-graph-using-api-fig-creating-using-graph-apis).

Note

When a stream is taken out of capture mode, the next non-captured item in the stream (if any) will still have a dependency on the most recent prior non-captured item, despite intermediate items having been removed.

##### 4.2.2.1.2.2. Prohibited and Unhandled Operations

It is invalid to synchronize or query the execution status of a stream which is being captured or a captured event, because they do not represent items scheduled for execution. It is also invalid to query the execution status of or synchronize a broader handle which encompasses an active stream capture, such as a device or context handle when any associated stream is in capture mode.

When any stream in the same context is being captured, and it was not created with `cudaStreamNonBlocking`, any attempted use of the legacy stream is invalid. This is because the legacy stream handle at all times encompasses these other streams; enqueueing to the legacy stream would create a dependency on the streams being captured, and querying it or synchronizing it would query or synchronize the streams being captured.

It is therefore also invalid to call synchronous APIs in this case. One example of a synchronous APIs is `cudaMemcpy()` which enqueues work to the legacy stream and synchronizes on it before returning.

Note

As a general rule, when a dependency relation would connect something that is captured with something that was not captured and instead enqueued for execution, CUDA prefers to return an error rather than ignore the dependency. An exception is made for placing a stream into or out of capture mode; this severs a dependency relation between items added to the stream immediately before and after the mode transition.

It is invalid to merge two separate capture graphs by waiting on a captured event from a stream which is being captured and is associated with a different capture graph than the event. It is invalid to wait on a non-captured event from a stream which is being captured without specifying the `cudaEventWaitExternal` flag.

A small number of APIs that enqueue asynchronous operations into streams are not currently supported in graphs and will return an error if called with a stream which is being captured, such as `cudaStreamAttachMemAsync()`.

##### 4.2.2.1.2.3. Invalidation

When an invalid operation is attempted during stream capture, any associated capture graphs are *invalidated*. When a capture graph is invalidated, further use of any streams which are being captured or captured events associated with the graph is invalid and will return an error, until stream capture is ended with `cudaStreamEndCapture()`. This call will take the associated streams out of capture mode, but will also return an error value and a NULL graph.

##### 4.2.2.1.2.4. Capture Introspection

Active stream capture operations can be inspected using `cudaStreamGetCaptureInfo()`. This allows the user to obtain the status of the capture, a unique(per-process) ID for the capture, the underlying graph object, and dependencies/edge data for the next node to be captured in the stream. This dependency information can be used to obtain a handle to the node(s) which were last captured in the stream.

#### 4.2.2.1.3. Putting It All Together

The example in [Figure 22](#cuda-graphs-creating-a-graph-using-api-fig-creating-using-graph-apis) is a simplistic example intended to show a small graph conceptually. In an application that utilizes CUDA graphs, there is more complexity to using either the graph API or stream capture. The following code snippet shows a side by side comparison of the Graph API and Stream Capture to create a CUDA graph to execute a simple two stage reduction algorithm.

[Figure 23](#cuda-graphs-visualize-a-graph-using-graphviz) is an illustration of this CUDA graph and was generated using the `cudaGraphDebugDotPrint` function applied to the code below, with small adjustments for readability, and then rendered using [Graphviz](https://graphviz.org/).

![CUDA graph example using two stage reduction kernel](img/Figure-23-CUDA-graph-example-using-two-stage-reduction-kernel.png)

*Figure 23 CUDA graph example using two stage reduction kernel*

Graph API

```
void cudaGraphsManual(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t                 streamForGraph;
   cudaGraph_t                  graph;
   std::vector<cudaGraphNode_t> nodeDependencies;
   cudaGraphNode_t              memcpyNode, kernelNode, memsetNode;
   double                       result_h = 0.0;

   cudaStreamCreate(&streamForGraph));

   cudaKernelNodeParams kernelNodeParams = {0};
   cudaMemcpy3DParms    memcpyParams     = {0};
   cudaMemsetParams     memsetParams     = {0};

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
   memcpyParams.kind     = cudaMemcpyHostToDevice;

   memsetParams.dst         = (void *)outputVec_d;
   memsetParams.value       = 0;
   memsetParams.pitch       = 0;
   memsetParams.elementSize = sizeof(float); // elementSize can be max 4 bytes
   memsetParams.width       = numOfBlocks * 2;
   memsetParams.height      = 1;

   cudaGraphCreate(&graph, 0);
   cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);
   nodeDependencies.push_back(memcpyNode);

   void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

   kernelNodeParams.func           = (void *)reduce;
   kernelNodeParams.gridDim        = dim3(numOfBlocks, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   kernelNodeParams.kernelParams   = (void **)kernelArgs;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memsetParams, 0, sizeof(memsetParams));
   memsetParams.dst         = result_d;
   memsetParams.value       = 0;
   memsetParams.elementSize = sizeof(float);
   memsetParams.width       = 2;
   memsetParams.height      = 1;
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);

   memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
   kernelNodeParams.func           = (void *)reduceFinal;
   kernelNodeParams.gridDim        = dim3(1, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   void *kernelArgs2[3]            = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};
   kernelNodeParams.kernelParams   = kernelArgs2;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memcpyParams, 0, sizeof(memcpyParams));

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
   memcpyParams.kind     = cudaMemcpyDeviceToHost;

   cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
   nodeDependencies.clear();
   nodeDependencies.push_back(memcpyNode);

   cudaGraphNode_t    hostNode;
   cudaHostNodeParams hostParams = {0};
   hostParams.fn                 = myHostNodeCallback;
   callBackData_t hostFnData;
   hostFnData.data     = &result_h;
   hostFnData.fn_name  = "cudaGraphsManual";
   hostParams.userData = &hostFnData;

   cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(), nodeDependencies.size(), &hostParams);
}
```

Stream Capture

```
void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t stream1, stream2, stream3, streamForGraph;
   cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
   cudaGraph_t  graph;
   double       result_h = 0.0;

   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   cudaStreamCreate(&stream3);
   cudaStreamCreate(&streamForGraph);

   cudaEventCreate(&forkStreamEvent);
   cudaEventCreate(&memsetEvent1);
   cudaEventCreate(&memsetEvent2);

   cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

   cudaEventRecord(forkStreamEvent, stream1);
   cudaStreamWaitEvent(stream2, forkStreamEvent, 0);
   cudaStreamWaitEvent(stream3, forkStreamEvent, 0);

   cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

   cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2);

   cudaEventRecord(memsetEvent1, stream2);

   cudaMemsetAsync(result_d, 0, sizeof(double), stream3);
   cudaEventRecord(memsetEvent2, stream3);

   cudaStreamWaitEvent(stream1, memsetEvent1, 0);

   reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

   cudaStreamWaitEvent(stream1, memsetEvent2, 0);

   reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
   cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

   callBackData_t hostFnData = {0};
   hostFnData.data           = &result_h;
   hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
   cudaHostFn_t fn           = myHostNodeCallback;
   cudaLaunchHostFunc(stream1, fn, &hostFnData);
   cudaStreamEndCapture(stream1, &graph);
}
```

### 4.2.2.2. Graph Instantiation

Once a graph has been created, either by the use of the graph API or stream capture, the graph must be instantiated to create an executable graph, which can then be launched. Assuming the `cudaGraph_t graph` has been created successfully, the following code will instantiate the graph and create the executable graph `cudaGraphExec_t graphExec`:

```
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```

### 4.2.2.3. Graph Execution

After a graph has been created and instantiated to create an executable graph, it can be launched. Assuming the `cudaGraphExec_t graphExec` has been created successfully, the following code snippet will launch the graph into the specified stream:

```
cudaGraphLaunch(graphExec, stream);
```

Pulling it all together and using the stream capture example from [Section 4.2.2.1.2](#cuda-graphs-creating-a-graph-using-stream-capture), the following code snippet will create a graph, instantiate it, and launch it:

```
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

