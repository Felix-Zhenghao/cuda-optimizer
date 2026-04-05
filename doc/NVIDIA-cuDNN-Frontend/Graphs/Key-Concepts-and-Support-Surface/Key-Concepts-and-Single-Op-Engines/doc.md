# Graphs

The cuDNN library provides a declarative programming model for describing computation as a graph of operations. This *graph API* was introduced in cuDNN 8.0 to provide a more flexible API, especially with the growing importance of operation fusion.

The user starts by building a graph of operations. At a high level, the user is describing a dataflow graph of operations on tensors, which typically represents a partition of the user's full network graph, which they would like to offload to a GPU kernel (or small set of kernels). Given a *finalized graph*, the user then selects and configures an engine that can execute that graph. There are several methods for selecting and configuring engines, which have tradeoffs with respect to ease-of-use, runtime overhead, and engine performance.

## Key Concepts

As mentioned previously, the key concepts in the graph API are:

> * [Operations and Operation Graphs](#op-op-graphs)
> * [Engines and Engine Configurations](#engine-engine-config)
> * [Heuristics](#heuristics)

These concepts are covered in the subsections below. Later we'll go through an example to tie them all together.

### Operations and Operation Graphs

An operation graph is a dataflow graph of operations on tensors. It is meant to be a mathematical specification and is decoupled from the underlying *engines* that can implement it, as there may be more than one engine available for a given graph.

I/O tensors connect the operations implicitly, for example, an operation A may produce a tensor X, which is then consumed by operation B, implying that operation B depends on operation A.

### Engines and Engine Configurations

For a given operation graph, there are some number of engines that are candidates for implementing that graph. The typical way to query for a list of candidate engines is through a heuristics query, covered below.

An engine has knobs for configuring the engine.

### Other Runtime Concepts

### Heuristics

A *heuristic* is a way to get a list of engine configurations that are intended to be sorted from the most performant to least performant for the given operation graph. There are three modes:

> * Heuristics Mode A - intended to be fast and be able to handle most operation graph patterns. It returns a list of engine configs ranked by the expected performance.
> * Heuristics Mode B - intended to be more generally accurate than mode A, but with the tradeoff of higher CPU latency to return the list of engine configs. The underlying implementation may fall back to the mode A heuristic in cases where we know mode A can do better.
> * Fallback Heuristics Mode - intended to be fast and provide functional fallbacks without expectation of optimal performance.

The recommended workflow is to query either mode A or B and check for support. The first engine config with support is expected to have the best performance.

You can "auto-tune", that is, iterate over the list and time for each engine config and choose the best one for a particular problem on a particular device. Both the C++ and Python APIs have utility functions for doing this.

If all the engine configs are not supported, then use the mode fallback to find the functional fallbacks.

Expert users may also want to filter engine configs based on properties of the engine, such as numerical notes, behavior notes, or adjustable knobs. Numerical notes inform the user about the numerical properties of the engine such as whether it does datatype down conversion at the input or during output reduction. The behavior notes can signal something about the underlying implementation like whether or not it uses runtime compilation. The adjustable knobs allow fine grained control of the engine's behavior and performance.

## Supported Graph Patterns

The cuDNN Graph API supports a set of graph patterns. These patterns are supported by a large number of engines, each with their own support surfaces. These engines are grouped into four different classes, as reflected by the following four subsections: pre-compiled single operation engines, generic runtime fusion engines, specialized runtime fusion engines, and specialized pre-compiled fusion engines. The specialized engines, whether they use runtime compilation or pre-compilation, are targeted to a set of important use cases, and thus have a fairly limited set of patterns they currently support. Over time, we expect to support more of those use cases with the generic runtime fusion engines, whenever practical.

Since these engines have some overlap in the patterns they support, a given pattern may result in zero, one, or more engines.

### Pre-Compiled Single Operation Engines

One basic class of engines includes pre-compiled engines that support an operation graph with just one operation; specifically: `ConvolutionFwd`, `ConvolutionBwdFilter`, `ConvolutionBwdData`, or `ConvolutionBwBias`.

#### ConvolutionBwdData

`ConvolutionBwdData` computes the convolution data gradient of the tensor `dy`. In addition, it uses scaling factors alpha and beta to blend this result with the previous output. This graph operation is similar to [cudnnConvolutionBackwardData()](../../../backend/v9.20.0/api/cudnn-cnn-library.html#cudnnconvolutionbackwarddata "(in NVIDIA cuDNN Backend)").

![ConvolutionBwdData Engine](img/ConvolutionBwdData-Engine.png)

#### ConvolutionBwdFilter

`ConvolutionBwdFilter` computes the convolution filter gradient of the tensor `dy`. In addition, it uses scaling factors alpha and beta to blend this result with the previous output. This graph operation is similar to [cudnnConvolutionBackwardFilter()](../../../backend/v9.20.0/api/cudnn-cnn-library.html#cudnnconvolutionbackwardfilter "(in NVIDIA cuDNN Backend)").

![ConvolutionBwdFilter Engine](img/ConvolutionBwdFilter-Engine.png)

#### ConvolutionFwd

`ConvolutionFwd` computes the convolution of X with filter data W. In addition, it uses scaling factors alpha and beta to blend this result with the previous output. This graph operation is similar to [cudnnConvolutionForward()](../../../backend/v9.20.0/api/cudnn-cnn-library.html#cudnnconvolutionforward "(in NVIDIA cuDNN Backend)").

![ConvolutionBwdFilter Engine](img/ConvolutionBwdFilter-Engine.png)

### Generic Runtime Fusion Engines

The engines documented in the previous section support single-op patterns. Of course, for fusion to be interesting, the graph needs to support multiple operations. And ideally, we want the supported patterns to be flexible to cover a diverse set of use cases. To accomplish this generality, cuDNN has runtime fusion engines that generate the kernel (or kernels) at runtime based on the graph pattern. This section outlines the patterns supported by these runtime fusion engines (that is, engines with `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION` behavioral note).

We can think of the support surface as covering the following generic patterns:

1. `Matmul` fusions: \(g\_{2}\left( C=Matmul\left( A=g\_{1A} \left( inputs \right), B=g\_{1B} \right(inputs)), inputs \right)\)
2. `ConvolutionFwd` fusions: \(g\_{2}\left( Y=ConvolutionFwd\left( X=g\_{1} \left( inputs \right), W\right), inputs \right)\)
3. `ConvolutionBwdFilter` fusions: \(g\_{2}\left( dw=ConvolutionBwdFiler\left( dy, X=g\_{1} \right(inputs)), inputs \right)\)
4. `ConvolutionBwdData` fusions: \(g\_{2}\left( dx=ConvolutionBwdData\left( dy=g\_{1} \left( inputs \right), W \right), inputs \right)\)
5. `Pointwise` fusions: \(g\_{2}\left( inputs \right)\)

![Graphical Representation of the Generic Patterns Supported by the Runtime Fusion Engines](img/Graphical-Representation-of-the-Generic-Patterns-Supported-by-the-Runtime-Fusion-Engines.png)

Note

* g 1 (including g 1A and g 1B) indicates fusion operations applied to the inputs of the `matmul` and `convolution` operation.
* g 2 indicates fusion operations apply to the output of the `matmul` and `convolution` operation.
* g 2 can have more than one output.
* The fusion patterns in g 1 will be referred to as the mainloop fusion, and the fusion patterns in g 2 will be referred to as the epilogue fusion.
* The arrow going into g 2 can go into any of g 2 nodes and does not necessarily need to feed into a root node.
* The abbreviated notations for operations are used in the diagrams and throughout the text for visualization purposes.
