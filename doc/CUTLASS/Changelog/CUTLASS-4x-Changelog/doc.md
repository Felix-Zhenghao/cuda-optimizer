# Changelog

# CUTLASS 4.x

## [4.4.2](https://github.com/NVIDIA/cutlass/releases/tag/v4.4.2) (2026-03-13)

### CuTe DSL

* New features

  + CuTe DSL now supports Python 3.14 for both x86\_64 and aarch64
  + Runtime Pointer/Tensor/FakeTensor now supports **cache\_key**, providing a stable, hashable representation that simplifies and improves compiled function caching.
* Bug fixing and improvements

  + Fixed Hopper FMHA causal attention performance regression on CUDA toolkit 13.1 by
    optimizing mbarrier synchronization to avoid unnecessary convergence barriers.
  + Fix kernel loading race condition when multiple GPU are present in the same process in JAX.

### CUTLASS C++

* Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler.

## [4.4.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.4.1) (2026-02-27)

### CuTe DSL

* Bug fixing and improvements

  + Fixed a segfault issue with tvm-ffi on aarch64

## [4.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.4.0) (2026-02-14)

### CuTe DSL

* New features

  + CuTe DSL now supports CUDA toolkit 13.1!

    - Set up with cutlass/python/CuTeDSL/setup.sh --cu13
    - Refer to https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick\_start.html for more details
  + GB300 is now supported in CuTe DSL with CTK 13.1

    - Refer to [SM103 batched 3xFP4 blockscaled GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py) for example kernel
  + cute.experimental: introduce a higher-level, composable layer on top of existing CuTe DSL APIs (not a separate abstraction), which can be mixed with existing Cute DSL building blocks.

    - Fragment-free programming model: copy/dot APIs take memrefs directly instead of descriptors/fragments.
    - Automatic TMA descriptor generation and update insertion.
    - Automatic vectorization and predication for SIMT copies.
    - New pipeline abstraction with convenience wrappers
    - New Partition ops to simplify partitioning logic.
    - Device-side TMA descriptor allocation, initialization, and management
    - These examples can be found here https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/experimental
  + Ahead of Time (AoT) compilation is now available!

    - Refer to files under https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/export for example usage
  + JAX support - you can now use CuTeDSL along with JAX

    - Refer to files under https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/jax for example usage
  + Introduced versioning support in DSL:

    - cutlass.**version** for a string representation of DSL version
    - cutlass.CUDA\_VERSION for a version class to tell the CUDA version used for DSL
  + Added CopyDsmemStoreOp to store data to distributed shared memory with explicit synchronization.
  + Grouped GEMM example now supports device-only problem shapes.
  + We allow grid carve-out without problem shapes being available on host.
  + Tma+LdMatrix features for loading+unpacking narrow-width types (refer to mixed\_input\_fmha\_decode.py for example usage).
  + It is possible now to have customized epilogue fusion for persistent dense GEMM through a Python Epilogue Fusion Configuration (EFC) function, somewhat similar to CUTLASS C++ EVT. It also provides a PyTorch evaluator to compare the results.
* More examples of authorizing peak-performance kernels

  + [SM103 batched 3xFP4 blockscaled GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py)
  + Mixed input FMHA decode example with support for int4 KV (int8 KV supported in 4.3)
  + New acc\_scale grouped mixed input gemm kernel variant is introduced to deliver better performance for decoding cases.
  + All mixed\_input\_gemm examples are moved into a separate folder `mixed_input_gemm`. Common utility functions are also extracted into mixed\_input\_host\_utils.py under the same folder.
* Bug fixing and improvements

  + Fixed an issue that both branches of if are executed
  + Fixed `cute.printf` with f-string
  + Fixed an indexing issue of scalar tensor
  + Fixed small K reference check error for cta\_tile\_n = 256 case with overlapping accumulator optimization in [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py).
* API changes

  + Deprecate get\_num\_tmem\_alloc\_cols from blackwell\_helpers.py. Use the one from tmem\_allocator.py instead.
  + Deprecate SM100\_TMEM\_CAPACITY\_COLUMNS and SM100\_TMEM\_MIN\_ALLOC\_COLUMNS.
  + LdMatrix16x16x8bOp and StMatrix16x8x8bOp now require explicit transpose=True when calling **init**, to avoid ambiguity in data transposition.
  + LdMatrix16x16x8bOp copy traits updated to be faithful to PTX without permutations. Permuted variant is renamed to LdMatrix16x8x8bOp.
  + Grouped GEMM example takes the argument --host\_problem\_shape\_available. If the argument is provided, grid is carved out based upon the host problem shapes, otherwise, we launch maximum possible SMs.
  + hardware\_info.get\_max\_active\_cluster support pass in specific stream to query. Useful for green context based SM partition.
  + group\_bulk\_copy\_modes in async bulk copy example is now deprecated, use group\_modes directly instead.
  + Deprecate nvvm wrapper from using nvvm enum, use str instead.
  + cute.arch.calc\_packed\_f32x2\_op default enable ftz to default disable ftz
  + In CuTe DSL with CTK 13.1, following APIs in cutlass.cute.arch now require string literal instead of enum as argument:

    - fence\_proxy
    - fence\_view\_async\_tmem\_op
    - calc\_packed\_f32x2\_op
    - warp\_redux\_sync
    - atomic\_add
    - atomic\_and
    - atomic\_or
    - atomic\_xor
    - atomic\_max
    - atomic\_min
    - atomic\_exch
    - atomic\_cas
    - store
    - load
* Use 'Advanced control file' for mixed input gemm examples for better performance.

  + Advanced control file is an experimental feature of CUDA compiler. The controls file contains internal compiler settings tuned for specific kernels with a specific version of CUDA toolkit to get better GPU kernel code. More details and documentation on how to create these controls files will be provided in future CUDA toolkit release. Note: The advanced compiler control file is not expected to work for kernels that it was not tuned for. There is no compatibility guarantee, and the controls file will not work for CUDA toolkit with a different version.

### CUTLASS C++

* Add [example 93](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/) for Blackwell low latency generation phase GQA kernel.

  + Flash Decoding with cluster reduction.
  + Kernel design details please check [Readme](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa/readme.md).
* Add Blackwell SM100 State Space Decomposition (SSD) kernel in [example 112](https://github.com/NVIDIA/cutlass/tree/main/examples/112_blackwell_ssd).
* Add Hopper SM90 State Space Decomposition (SSD) kernel in [example 111](https://github.com/NVIDIA/cutlass/tree/main/examples/111_hopper_ssd).
* Add [example 94](https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise/) for Ada FP8xFP8 -> BF16 GEMM with blockwise dequantization of input matrices in the MMA loop with FP32 accumulation.

  + Generate additional device/kernel/threadblock files in CUTLASS include directory that add functionality to carry the scaling tensors + use them in MMA loop.
  + Add gemm\_blockwise to include files in [default\_mma\_core\_sm80](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/threadblock/default_mma_core_sm80.h)
* Add Hopper e2m1 to fp32 optimized conversion and e2m1 \* TF32 tensor core GEMM.

  + Set MmaType to tfloat32\_t for FP32 mode.
  + TF32 provides FP32 inputs with reduced precision (19-bit vs 32-bit)
  + Set TileShapeK=64 for TF32 (K must be multiple of 8)
  + Shuffle optimization enabled via `compute_memory_reordering_atom<tfloat32_t>()`
  + E2M1 -> FP32 -> TF32 TC path for mixed-precision GEMM
  + Enable [example 55](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) with TF32 support
* Add support for arbitrary application-provided strides for block-scale tensors.

  + Users and applications now must pass valid block-scale strides in all cases, even when the tensor is packed.
* Support 4x blockscaled public ptx for CUDA 13.1.
* Allow non-static `TmaGbasis` in `AuxTmaParams`.

  + Some cases in attention kernel may require non-static `tma_gbasis`.
  + Relax the restriction on `TmaGbasis` parameter of `AuxTmaParams` and users are allowed to manually construct a dynamic gbasis.
* Fix some kernel issues:

  + Fix MSVC pre process issue.
  + Fix a self assign issue in GEMV kernel.
  + Fix a TMA descriptor bug where the CUDA driver is not properly setting the OOB address gen mode correctly.
  + Fix memory fence for clc scheduler in Blackwell SM120 pingpong kernel.
  + Fix missing SMEM alignment in Blackwell SM120 scale factors.
  + Fix a PDL issue for grouped gemm.
  + Fix divide-by-zero issue in canimplement for sm100 implicit gemm kernels.
  + Fix cluster swizzle for Grouped GEMMs.

    - Move host-side swizzling heuristics to device.
    - Apply swizzle per group based on problem shape and max swizzle size.
    - Improve examples and unit tests.
* Fix some profiler issues:

  + Fix a core dump issue for nvfp4 grouped GEMM kernel.
  + Fix inconsistent GEMM verification logic.
  + Rework grouped gemm verification logic for different types.
  + Fix api break change in using nvMatmulHeuristics.
* Fix some failed links under `media/docs`.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 13.1.

## [4.3.5](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.5) (2026-01-09)

### CuTe DSL

* Bug fixing and improvements

  + Fixed the unexpected CPU overhead issue introduced by 4.3.4
* Update copyright to 2026.

### CUTLASS C++

* Update copyright to 2026.
* Use CUDA Driver Get Version Runtime APIs Rather than Driver APIs.

## [4.3.4](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.4) (2025-12-22)

### CuTe DSL

* New features

  + Added PDL support along with example [Kernel launch with Programmatic Dependent Launch](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/programmatic_dependent_launch.py)
* Bug fixing and improvements

  + Fixed a frame refcnt issue with cuda graph
  + Enhancement for tvm-ffi AoT case for earlier module unload
  + Fixed order issue in `make_smem_layout_a` in utils/hopper\_helpers.py

### CUTLASS C++

* Work around a driver TMA descriptor related bug which will cause occasional errors on Blackwell when the tensor's backing memory allocation is less than 128KB and it is not a dense non-overlapping tensor.

## [4.3.3](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.3) (2025-12-12)

* New features

  + Supported namedtuple and kwargs for JIT function arguments in tvm-ffi
  + Supported variadic tuples for JIT function argument in tvm-ffi
* Bug fixing and improvements

  + Fixed an issue when JIT function argument with union type annotation for tvm-ffi
  + Clearer error message for the case of runtime error cudaErrorInsufficientDriver

## [4.3.2](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.2) (2025-12-05)

* New features

  + New env var `CUTE_DSL_CACHE_DIR` to specify the path for dumping caches
* Bug fixing and improvements

  + Fixed an issue of CUDA JitExecutor when unloading kernels
  + Fixed an issue of allocating max smem when there's statically allocated smem

## [4.3.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.1) (2025-11-26)

### CuTe DSL

* New features

  + Added Blackwell SM103 support
  + Multiple dependent DSOs in the wheel have been merged into one single DSO
* Bug fixing and improvements

  + Fixed device reset issue with tvm-ffi
  + Fixed tvm-ffi export compiled function

### CUTLASS C++

* Support blockscaled variant of ragged contiguous grouped gemm with the new simplified MoE API in [example 92](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm/).

  + The new example works for all microscaling types.

## [4.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.3.0) (2025-11-21)

### CuTe DSL

* New features:

  + Supported Apache [TVM-FFI](https://tvm.apache.org/ffi/index.html) for further reduced host runtime overhead for JIT functions, better PyTorch and ML frameworks interopability
  + Added fake tensor and stream to decouple compile jit function with "from\_dlpack" flow. Now we no longer require users to have real tensor when compile jit function.
  + Added FastDivmodDivisor with Python operator overloads, new APIs, Cute dialect integration, and optimized static tile scheduler performance for faster index mapping.
  + Added l2 cache evict priority for tma related ops. Users could do fine-grain l2 cache control.
* Debuggability improvements:

  + Supported source location tracking for DSL APIs (Allow tools like `nsight` profiling to correlate perf metrics with Python source code)
  + Supported dumping PTX and CUBIN code: [Hello World Example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb)
* More examples and notebooks to get started with CuTe DSL:

  + Improved performance of [elementwise example](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/elementwise_apply.py):

    - Generalize code to handle list of input tensors
    - Generalize TV layout computation to handle different data types
  + Improved [Blackwell SM100 persistent dense GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py):

    - To demonstrate usage of new Pipeline APIs `PipelineProducer` and `PipelineConsumer` to simplify code without explicit pipeline state management (Exiting APIs are still maintained)
    - Separated epilogue code for non-TMA and TMA implementation
  + [Tutorial for Blackwell GEMM: Basic Blackwell SM100 GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/tutorial_gemm)

    - [Baseline Blackwell GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) achieves 84% SOL performance with MNK 8K
    - More examples are coming for demo of optimization: `Baseline + X`
  + [Tutorial for Async Pipeline API](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/async_pipeline.ipynb)
  + Reworked [elementwise add notebook](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb) with more details and detailed explanation about TV layout

    - Updated implementation to handle general data type and multiple inputs
    - Updated explanation for TV layout in simpler language
    - Added visualization of TV Layout with 3rd party utils
  + [Benchmark and autotune demonstration](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/benchmark_autotune.ipynb)
* More examples of authorizing peak-performance kernels:

  + [Blackwell SM100 mixed-input GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/mixed_input_gemm.py)
  + [Blackwell SM100 persistent blockwise dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py)
  + [Blackwell SM100 persistent blockwise contiguous grouped dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/blockwise_gemm/contiguous_grouped_gemm.py)
  + [Blackwell SM100 persistent blockwise masked grouped dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/blockwise_gemm/masked_grouped_gemm.py)
  + [Blackwell SM100 fmha bwd](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/fmha_bwd.py)
  + [Blackwell SM100 mla](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/mla.py)
  + [Hopper SM90 persistent dense GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py)
  + [Blackwell GeForce batched dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py)
  + [Ampere HSTU Attention](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/hstu_attention.py)
* API updates:

  + Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details
* Bug fixings and improvements

  + Add mma\_tiler\_n=64 and mma\_tiler\_n=192 support in [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py).
  + Fixed `TensorSSA.reduce` to support static value as initial value
  + Updated docstring for following APIs to be more concise and easier to understand:

    - `make_layout_tv`
    - `is_static`
    - `PipelineAsync`
    - `SmemAllocator`
  + Fixed documentation for `pipeline`, `utils` and `cute.math`
  + Added overlapping accumulator optimization for block tile N = 256 case for better epilogue latency hiding in [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py).
  + Fixed TensorSSA.**getitem** indexing to match CuTe's indexing convention
  + Fixed an issue with cutlass.max and cutlass.min
  + Fixed an issue with mark\_compact\_shape\_dynamic

### CUTLASS C++

* Further enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).

  + Add softmax skip correction.
  + Fix a shared memory allocation bug where it needs to opt in maximum dynamics shared memory explicitly once it exceeds 48KB.
  + Fix a dead hang issue caused by early return warp.
* Add support through cmdline argument lists for `batch`, `no_verif`, `cluster_shape` and `cluster_shape_fallback` in [example 89](https://github.com/NVIDIA/cutlass/tree/main/examples/89_sm103_fp4_ultra_gemm/).
* Add Ragged Contiguous Grouped gemm kernel in [example 92](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm/).

  + This kernel uses a TMA 3D load to load the weights matrix and use the tensormap update method to load activations.
* Add 256x128 tile size support for Hopper SM90 deepgemm in [example 67](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/).

  + Performance is optimized to align with Deepseek implementation.
* Simplification of API for MoE gemms.

  + Instead of requiring users to call several cute utilities to set up the stride, API `moe_stride_utils` is introduced to help setup strides in the kernel.
  + Instead of requiring users to set vectors like `problem_shapes_device` and `problem_shapes_hosts`, a new problem shape struct called `MoEProblemShape` is introduced which takes in max\_m, max\_n, max\_k and counts vector as input and deduce problem shapes internally whenever required.
* Enable GEMM\_K = 0 in grouped gemm.
* Optimize group gemm kernels by enabling async TMA desc update.
* Support Blackwell SM100 convolution stream-K kernel.

  + Unit tests: [fprop\_streamK](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/fprop/sm100_conv3d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16_streamk.cu), [dgrad\_streamK](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/dgrad/sm100_conv3d_dgrad_implicit_gemm_f16_f16_f16_tensorop_f16_streamk.cu), [wgrad\_streamK](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/wgrad/sm100_conv2d_wgrad_implicit_gemm_f16_f16_f16_tensorop_f16_streamk.cu).
* Add Blackwell SM100 sparse gemm compressor unit tests.

  + Unit tests: [compressor\_fp16](https://github.com/NVIDIA/cutlass/tree/main/test/unit/transform/device/sm100_sparse_gemm_compressor_f16.cu).
  + Add sub-bytes and runtime data type support in compressor unit test testbed.
* Add profiler support for:

  + Blackwell SM100 and SM120 blockscaled sparse kernels.
  + New MoE grouped gemm API.
  + Blackwell SM100 cpasync kernel.
* Fix some kernel issues:

  + Fix a race check issue of Blackwell SM103 kernels by adding missing elect one for prefetch barrier initialization.
  + Allow user to directly specify the number of stages for Hopper sm90 mixed input gemm.
  + Remove warnings caused by cuda vector type alignment setting in CUDA 13.
  + Remove problematic `cutlass::int8_t` and replace it with `int8_t`.
  + Fix a few bugs in distributed gemm API and examples.
  + Fix handling negative zero in sparse compressor.
  + Add missing `wait_on_dependent_grids` for PDL use case.
* Fix some profiler issues:

  + Add some missing reference kernels.
  + Support VoidC reference kernels.
  + Add calculation of scale factor A and B in function `bytes_with_problem_shape` of block scaled profiler.
  + Fix an issue when epilogue tile N is not divided by default subtile N.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 13.0U1.

## [4.2.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.1) (2025-09-22)

### CuTe DSL

* Bug fixings and improvements

  + Fixed an issue when running DSL codes with cuda-python 13.0
  + Fixed an issue when running inductor with DSL codes
  + Fixed an issue with unexpected logging when running DSL codes in FlashInfer
  + Fixed the issue reported in https://github.com/NVIDIA/cutlass/issues/2647
  + Fixed an issue when conditional define of variables outside of dynamic control flow

### CUTLASS C++

* Bypass EVT for nosmem blockwise kernels on Blackwell.
* Rename cutlass/python/cutlass directory to cutlass/python/cutlass\_cppgen.

## [4.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.0) (2025-09-15)

### CuTe DSL

* More Python versions are now supported for both x86-64 and aarch64, including

  + Python 3.10, 3.11, 3.12, and 3.13
* Added new example and updated notebook to get started with CuTe DSL

  + [Call kernels with dlpack bypassed](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/call_bypass_dlpack.py)
  + Updates on [TensorSSA demonstration](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)

    - Added a section for introducing the broadcast
* API updates

  + Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details
* Bug fixings and improvements

  + Fixed `cute.print_tensor` for coordinate tensor
  + Fixed `cute.print` for tuple of layouts
  + Fixed frozen object is not properly updated after fully assigned in dynamic control flow
  + Fixed assign tuple/list element in a dynamic control flow may cause compilation failure
  + Improved error message when CUDA context is not initialized
  + Improved docstring of congruent and weakly\_congruent

### CUTLASS C++

* Support for Blackwell SM103 kernels for B300 GPUs.

  + Collective mainloop codes: [Blockscaled datatypes with support for dense GEMM mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm103_blockscaled_mma_warpspecialized.hpp)
  + New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp) and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  + Kernel codes: [Blockscaled datatypes with support for dense GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM103 architecture:

  + [Blockscaled ultra fp4 dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/89_sm103_fp4_ultra_gemm/).
  + [Blockscaled ultra fp4 dense grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/90_sm103_fp4_ultra_grouped_gemm).
* Set of unit tests that demonstrate the usage of Blackwell SM103 blockscaled GEMM

  + Unit test files with prefix name of `sm103_` under [GEMM device unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/).
* Support for Blackwell SM121 kernels for DGX Spark GPUs.

  + Share the major codes with Blackwell SM120 kernels.
* Add support for heuristics-based kernel filtering and autotuning using `nvidia-matmul-heuristics` to find the best kernels for a given scenario.

  + Details please refer to [heuristics doc](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/heuristics.md).
* Further enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).

  + Add fused reduction kernel support for cutlass MLA.
  + Add softmax skip correction.
  + Support for GQA in FMHA backward kernel.
  + Fix an issue where `get_unmasked_trip_count` may return a negative value.
  + Fix an issue where mbarriers are initialized with a zero arrival count.
  + Fix a corner case issue where the sequence length of q is not a multiple of tile\_q.
  + Remove tma padding for forward kernel inputs.
* Add Blackwell SM100 kernels for MoEs (focusing on Low-Latency inference performance): [example 92](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm/). It uses TMA (for weights) and CPASYNC (for tokens) to load input matrices and allow only one problem dimension to vary across groups/experts, unlike general Grouped GEMMs. Note: further API simplifications and kernel improvements are upcoming. Any feedback on API is welcome.
* Further enhance blockwise and groupwise GEMMs on Hopper and Blackwell

  + On Blackwell SM120, a blockwise gemm kernel is added: [example 87](https://github.com/NVIDIA/cutlass/tree/main/examples/87_blackwell_geforce_gemm_blockwise/).
  + On Hopper, add K major scale factor support for SM90 blockwise kernels.
  + On Hopper, relax the restriction that the k dimension of the problem size has to be the multiple of the k dimension of the tile size.
  + On Hopper, grouped version supports the case when k = 0.
* Support for Blackwell SM100 fp4 gemv kernels.

  + Kernel codes: [Gemv kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemv_blockscaled.h).
  + Example codes: [example 91](https://github.com/NVIDIA/cutlass/tree/main/examples/91_fp4_gemv/)
* Support for Blackwell SM100 legacy mixed input GEMM kernels.

  + Collective mainloop codes: [Mixed input mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp).
  + Kernel codes: [Mixed input kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp).
  + Example codes: [example 86](https://github.com/NVIDIA/cutlass/tree/main/examples/86_blackwell_mixed_dtype_gemm/).
* Support for Blackwell SM100 cpasync kernel.

  + Collective mainloop codes: [cpasync mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp).
  + Kernel codes: [cpasync kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp).
* Support Blackwell SM120 mixed input blockscaled grouped GEMM.
* Instantiating more Blackwell kernels in profiler.

  + Blackwell SM100 and SM103 kernels support `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` to instantiate all possible combinations.
  + To use this feature, `CUTLASS_LIBRARY_KERNELS` must be non-empty. Profiler will combine `CUTLASS_LIBRARY_KERNELS` and `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` to instantiate specific kernels.
  + Details please check [Profiler Doc](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/profiler.md).
* Fix some profiler issues:

  + Modify default cluster callback values to none 0 to avoid profiler failure when these values are not set in command line.
  + Fix some no output and timeout issues.
  + Fix Pingpong Blockwise Hopper library generation.
* From CUDA 13.0, the Blackwell SM101 for Thor GPUs is renamed to SM110.

  + For CUDA toolkit version < 13.0, SM101 is still used for Thor GPUs.
  + For CUDA toolkit version >= 13.0, SM110 is used for Thor GPUs and SM101 is no longer valid.
* Rename legacy Python API package from `cutlass` to `cutlass_cppgen` and add Blackwell EVT support to legacy Python interface.

  + Restructuring the C++ Blackwell SM100 Collective Epilogue Builder to work with the Python interface's `EpilogueDescriptors`.
  + Added Blackwell SM100 EVT Emitter on the Python side and routed most emission through Hopper SM90 Emitter.
  + Added some support for running SM100 kernels via the Python interface.
* CuTe changes:

  + Fix inaccurate GridDim calculation under [CuTe tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/blackwell/).
  + Add [movmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix) support.
  + Fix smallest MMA-N allowed for Blackwell fp8 and fp16 gemm kernels.
  + Support fp16 accmulator for sm89 fp8 mma.
  + Shorten `nullspace` implementation.
  + Isolate and comment on `cosize` risky changes.
  + Important documentation correction: `E<0,1> == 1@0@1`.
* Fix some kernel issues:

  + Fix Hopper SM90 group gemm kernel to only use the commit group and wait group instead of also waiting on mbarriers.
  + Fix a tiny bug when K is large for Blackwell SM103 fp4 grouped GEMM kernel.
* Add following unit tests:

  + [fp16 accmulator for sm89 fp8 mma](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/ampere/cooperative_gemm.cu)
  + [movmatrix test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/turing/movm.cu)
  + [fp8 narrow mma n](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32_narrow_mma_n.cu) and [fp16 narrow mma n](_downloads/1436896da89c8028bb0ca7335297f00a/f8_f8_void_bf16_narrow_mma_n.cu)
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 13.0U1.

## [4.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.1.0) (2025-07-16)

### CuTe DSL

* Add aarch64 support, you can now pip install `nvidia-cutlass-dsl` on GB200 systems!
* More examples demonstrating how to use CuTe DSL to write peak-performance kernels

  + [Blackwell Mamba2 SSD](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd.py)
  + [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)
* API updates

  + Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details

### CUTLASS C++

* Further enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).

  + Add variable sequence length support for FMHA Backward kernel.
  + Add varlen test support to Backward runner.
  + Codes support empty batch sequences.
* Replace `subbyte_iterator` with `cute::recast_ptr` when constructing logical iterators/arrays.
* CuTe changes:

  + Rewrite ArithTuple and ScaledBasis for robustness and clarity.
  + Remove buggy and kludgy `get_layoutA|B|C_MN` and friends from Atoms/TiledX.
  + Factor out `print_latex` and friends and rewrite.
  + Factor out `print_svg` and friends and rewrite.
* Support Blackwell SM100 SIMT packed fp32x2 kernels.
* Support residual add for implicit gemm kernels.
* Various fixes for CUTLASS C++ Python interface's EVT tracer:

  + Add verifier for sm90 to report the invalid input.
  + When adding an edge to the graph, if the edge already exists, add an identity compute node to avoid having multiple parallel edges.
  + Register operations of tanh, sigmoid, exp, gelu to the python ast frontend.
  + Replace the NotImplemented Error by packing all nodes into a single topological visitor node as a fallback.
* Fix profiler bugs in exhaustive perf search.

  + Fix incorrect cluster shape output issue when doing exhaustive search.
  + Fix a bug in profiler grouped GEMM for setting tile scheduler swizzles, cluster shapes, and raster orders.
* Fix some profiler issues.

  + Complete the reference for Blackwell blockwise gemm kernels.
  + Fix incorrect regex logic for L1 test.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.9.

## [4.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.0.0) (2025-06-03)

### CuTe DSL

* CuTe DSL, a Python DSL centered around CuTe's abstractions

  + [Core DSL implementation files](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)
  + [DSL quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
  + [DSL Overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
* [Overhauled documentation with a new dedicated website](https://docs.nvidia.com/cutlass/latest)
* Set of examples demonstrating how to use CuTe DSL to write peak-performance kernels

  + [Blackwell SM100 persistent dense GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)
  + [Blackwell SM100 grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py)
  + [Blackwell SM100 fused multi-head attention forward pass](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/fmha.py)
  + [Hopper GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/hopper/dense_gemm.py)
  + [Ampere GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py)
  + [FlashAttention-2 implementation targeting Ampere and Ada class GPUs (SM80, SM86, SM89)](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py)
  + [SmemAllocator to facilitate shared memory allocation and management](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/smem_allocator.py)
  + [C-structure based customized interface between JIT function and user codes](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/ffi/jit_argument.py)
* [Educational notebooks for getting started with CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks)
* API updates

  + Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details

### CUTLASS C++

* Support [Family Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) which was introduced in CUDA 12.9

  + 100f, 101f, 120f were added to support Family Specific Architecture Features which allows running the same binary on different chips belonging to the same Family (e.g. sm100) without recompiling. Note 101a is supported since CUTLASS 3.9
* Instruction shapes and redundant accumulation type have been removed from CUTLASS 3.x-style library kernel names to disambiguate kernels and shorten names.

  + For example:

    - `(old) cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_cooperative_epi_tma`
    - `(new) cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_cooperative_epi_tma`
  + If you are using the CUTLASS library kernel names directly (e.g. to compile a subset of the CUTLASS library with `-DCUTLASS_LIBRARY_KERNELS`, filter kernels in the CUTLASS profiler with `--kernels`), please update your uses accordingly, this is a breaking change.
* Further improved [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMMs on Hopper and Blackwell.

  + Added non-power-of-two tile sizes.
  + Improved performance for K-major scale factors.
  + The argument `mma_promotion_interval` has been removed from non-grouped GEMM to align with the grouped and Blackwell SM100 versions.
* Enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).

  + Support LSE output in FMHA Forward kernel.
  + Enhance performance measurement: support of different warmup iterations; buffer rotation to keep L2 cold; separate testing of persistent and non-persistent.
  + Enhance testing of variable sequence length.
  + Disable B2B mode in MLA to simplify the sample.
  + Clarify that `fmha_gen` sample only supports head dim 128.
  + Fixes for split-kv output in MLA.
* Improve Blackwell and Hopper grouped GEMM performance, functionality, and profiler support.

  + Enable runtime datatype for Blackwell SM100 grouped GEMM. Profiler support is also added.
  + Enable kernel parameter exploration for Blackwell SM100 grouped GEMM - raster\_order, swizzle.
* Add [Blackwell SM100 implicit GEMM conv fprop/dgrad/wgrad unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/).
* Add dynamic and preferred cluster support for convolution Blackwell SM100 kernels.
* Fix profiler issues which cause no output or not supported error for some kernels.
* Optimizations for Blackwell SM100 and SM120 block scaled kernels.
* Support for Blackwell SM120 blockwise dense gemm in CUTLASS library and profiler.
* New [Hopper SM90 FMHA example](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha/), similar in design to the existing [Blackwell FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).
* CuTe changes:

  + Rework `cute::copy_if` so that the predicate tensor is also a true CuTe Tensor rather than a lambda and introduces transform-tensors to avoid any extra register or load/store overhead in using bool-tensors.
  + New [CuTe tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/tiled_copy_if.cu) to show the usage of copy\_if in tile copy.
  + Add [CuTe C++ reduce op](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/tensor_reduce.hpp).

    - Add several [unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/tensor_algs.cpp) for CuTe tensor algorithms.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.9.

