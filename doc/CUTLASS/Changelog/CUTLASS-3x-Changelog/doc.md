# CUTLASS 3.x

## [3.9.2](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.2) (2025-05-03)

* Fixed [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMM hang issue when problem size K is 128.
* Optimal code generation with CUDA toolkit versions 12.9.

## [3.9.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.1) (2025-04-30)

* Fixed Group Gemm hang issue in CUTLASS 3.x
* Improved Hopper [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMM performance.

## [3.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.0) (2025-04-24)

* Support for Blackwell SM120 kernels for GeForce GPUs in CUTLASS 3.x API:

  + Collective mainloops that target for:

    - [Blockscaled datatypes with support for dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
    - [Blockscaled datatypes with support for sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp)
  + New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp) and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  + [Blackwell SM120 epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp) and [full set of EVT fusions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM120 architecture:

  + [Blockscaled GEMM with NVFP4 input datatype and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).
  + [Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor with scale factor generation](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu).
  + [Blockscaled GEMM with mixed input datatype (MXFP8 and MXFP6) and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu).
  + [Grouped GEMM with nvfp4 datatype](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu).
  + [Sparse Blockscaled GEMM with mxfp8 input datatype and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu).
  + [Sparse Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu).
* Set of unit tests that demonstrate the usage of both [sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_blockscaled_sparse_tensorop_gemm/) and [dense](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/) Blackwell SM120 blockscaled GEMM.
* Support for Blackwell SM100 Sparse kernels:

  + Collective mainloop that target for

    - [SM100 Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_sparse_mma_warpspecialized.hpp)
* Set of example that demonstrate the usage of the 3.x API for targeting Blackwell SM100 Sparse GEMM:

  + [Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/83_blackwell_sparse_gemm/83_blackwell_sparse_gemm.cu)
  + [Blockscaled Sparse GEMM with NVFP4 input data type](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84a_blackwell_nvfp4_bf16_sparse_gemm.cu)
  + [Blockscaled Sparse GEMM with mixed input data type (MXFP8 and MXFP4)](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84b_blackwell_mixed_mxfp8_bf16_sparse_gemm.cu)
* Set of unit tests that demonstrate the usage of [sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm) and [blockscaled sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm) Blackwell SM100 GEMM.
* A new Multi-head Latent Attention (MLA) for SM100 Blackwell architecture in CUTLASS [example](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/) covers the flashMLA-like weight-absorbed decoding use-case.
* A new FMHA Backward kernel for SM100 Blackwell architecture extends CUTLASS [example](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/) to show how the five backward pass MMAs can be fused into a single kernel to achieve high performance.
* A new [distributed GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/82_blackwell_distributed_gemm/82_blackwell_distributed_gemm.cu) for SM100 Blackwell architecture.
* Enhancement and new support of block-wise and group-wise GEMM for Hopper and Blackwell architectures:

  + Enhancement of [blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) for Hopper architecture.
  + Enhancement of [groupwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) for Hopper architecture.
  + Support for [grouped GEMM with blockwise and groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/) for Hopper architecture.
  + Support for [grouped-wise GEMM](https://github.com/NVIDIA/cutlass/tree/main/tools/profiler/src/blockwise_gemm_operation_profiler.cu) in CUTLASS profiler.
  + Support for [blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu) for Blackwell architecture.
  + Support for [groupwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu) for Blackwell architecture.
  + Support for [grouped GEMM with blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_blockwise.cu) and [groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_groupwise.cu) for Blackwell architecture.
* Added support for enhanced kernel performance search (auto-tuning) in CUTLASS profiler:

  + Sorting performance results by GFLOPs/second: Users can now sort the final performance report based on GFLOPs/second, making it easier to identify the most efficient kernels.
  + Exhaustive search for best kernel performance in GFLOPs/second: The profiler now searches for the best-performing kernel across a range of problem sizes, swizzle sizes, rasterization orders, and dynamic cluster configurations to maximize performance.
  + Performance search under a fixed GEMM shape: Enables exhaustive tuning within a fixed GEMM shape, exploring various kernel parameters to find the best configuration.
  + More detailed introductions and examples to leverage this feature can be found in [profiler.md](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflops-s).
* Support `void` as the D element in sm100 kernel epilogues.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.8U1.

## [3.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.8.0) (2025-01-25)

* Support for new CuTe building blocks specifically for Blackwell SM100 architecture:

  + [5th generation Blackwell Tensor Core instructions (TCGen05)](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/mma_traits_sm100.hpp) via CuTe MMA atoms.
  + Extensions to [Tensor Memory Accelerator](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100_tma.hpp) via CuTe Copy atoms.
  + Exposure of Blackwell's new tensor memory (note: distinct from TMA) as [`tmem`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/pointer.hpp) across CuTe as a first class data locale.
  + Exposure of [`tmem->rmem`, `rmem->tmem` and `smem->tmem data movement instructions`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) as copy atoms in CuTe.
  + [`make_tmem_copy()`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) utility method to ease creation of tiled copies for tmem copy atoms.
  + Support for [new variants of LDSM on Blackwell](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) via CuTe Copy atoms.
* Support for new CUTLASS building blocks specifically for Blackwell SM100 architecture:

  + Various narrow precision [FP4, FP6, and FP8](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/exmy_base.h) formats as well as their [block-scaled variants NVFP4, MXFP4, MXFP6, and MXFP8](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/float_subbyte.h)
  + [Pipelines that implement Blackwell specific synchronization](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/sm100_pipeline.hpp).
  + [Cluster launch control API supporting preferred and fallback cluster shapes](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cluster_launch.hpp).
  + Data types including NVFP4, MXFP4, MXFP6, and MXFP8 and all their supported element and scale factor types.
  + Tile schedulers using [Blackwell's Cluster Launch Control (CLC) feature](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html) to implement dynamic persistence scheduling for [GEMMs](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp), and [stream-K](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp).
  + Extensions to testbeds and reference check code for unit tests and CUTLASS profiler.
* Full support for Blackwell SM100 kernels in CUTLASS 3.x API:

  + [Blackwell specific kernel layers](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) that

    - Implement a new warp-specialization recipe tuned specifically for Blackwell SM100 architecture.
    - Leverage all the new features such as CLC based tile scheduling, preferred cluster, and TMEM based double buffering of accumulators.
    - Support stream-K load balancing for all kernel types everywhere via composable scheduler support.
  + Blackwell collective mainloops that target the TCGen05 MMA instructions (both SS and TS) for

    - [Non-block scaled data types without support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp)
    - [Non-block scaled data types with support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp)
    - [Block scaled data types without support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp)
    - [Block scaled data types with support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp)
  + Blackwell [collective mainloop for convolution kernels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp) supporting non-block scaled data types for fprop, dgrad, and wgrad.
  + New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp), [convolution](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/dispatch_policy.hpp), and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  + [Blackwell epilogue that supports loading accumulators from `tmem`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp) and full set of EVT fusions.
* CUTLASS library and profiler integration for block scaled data types for kernel emission, profiling, and verification.

  + Support for preferred and fallback cluster shapes via profiler command line arguments parsing to set dynamic cluster shapes.
  + Support for dynamic datatypes by parsing profiler via profiler command line arguments parsing to set dynamic datatype setting in TCGen05 MMA instruction descriptors.
  + Support for mixed input GEMM kernels on Hopper in the profiler.
* New CUTLASS profiler flag `use-cuda-graphs` to reduce overheads when benchmarking launch-bound kernels.
* A new 3.x version of grouped GEMM to the CUTLASS library and generates kernels for Hopper and Blackwell. Now grouped GEMM support is enabled in the CUTLASS profiler (`./cutlass_profiler --operation=GroupedGemm --help` for details).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM100 architecture:

  + [Basic FP16 and FP8 GEMMs with minimal changes from Hopper examples](https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm/), demonstrating ease of migration for off the shelf kernels using the 3.x collective builder API.
  + GEMM with [opt-in collective builder schedules showcasing available recipes](https://github.com/NVIDIA/cutlass/tree/main/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) for Blackwell.
  + Block scaled data type GEMMs targeting Blackwell's native block scaled Tensor Cores:

    - [NVFP4 inputs with BF16 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)
    - [NVFP4 inputs with NVFP4 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
    - [Mixed MXFP8 and MXFP6 inputs with BF16 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu)
  + GEMM example demonstrating [Blackwell's new preferred cluster support via dynamic cluster shapes](https://github.com/NVIDIA/cutlass/tree/main/examples/73_blackwell_gemm_preferred_cluster/blackwell_gemm_preferred_cluster.cu) for increased occupancy.
  + [GEMM with CLC based StreamK scheduler for load balancing](https://github.com/NVIDIA/cutlass/tree/main/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu).
  + Grouped GEMM for [vanilla FP8 data inputs](https://github.com/NVIDIA/cutlass/tree/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu) and [NVFP4 block scaled inputs](https://github.com/NVIDIA/cutlass/tree/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu).
  + Convolution kernels for [fprop](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_fprop.cu), [dgrad](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_dgrad.cu), and [wgrad](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_wgrad.cu).
  + [Fused multi-head attention fprop kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/77_blackwell_fmha.cu) supporting fp16/bf16/fp8 data types across head dims of 32,64, and 128.
  + A new BF16x9 GEMM [kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/78_blackwell_emulated_bf16x9_gemm/78_blackwell_emulated_bf16x9_gemm.cu) that emulates FP32 GEMM (SGEMM) using BF16 operations.
* Set of examples that demonstrate the usage of the 3.x API for targeting Hopper architecture:

  + A set of new [Hopper grouped GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/examples/69_hopper_mixed_dtype_grouped_gemm/) that support mixed A and B datatypes.
  + A new [Hopper FP8 GEMM with groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu).
* Documentation updates:

  + [Quickstart - instantiating a Blackwell block-scaled GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html#instantiating-a-blackwell-sm100-gemm-kernel).
  + Detailed [Blackwell block-scaled GEMM functionality documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
  + A new [functionality documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html) specifically for 3.x API comprehensively documenting all supported kernel types, data types, kernel features, minimum CUDA tookit support etc for 3.x supported architectures.
  + Updates to [compatibility](https://docs.nvidia.com/cutlass/latest/overview.html#compatibility) section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures, and [Target Architecture](https://docs.nvidia.com/cutlass/latest/overview.html#target-architecture).
  + Updates to [profiler documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) for testing mixed input GEMM kernels on Hopper.

## [3.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.7.0) (2025-01-11)

* [Hopper blockwise scaling FP8 GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) uses 2D scaling tensor, assigning one value per threadblock. This allows a finer-grained scaling to be applied for each output tile per gemm-k iteration. The operands and scaling tensors are loaded from global memory to shared memory using TMA and cp\_async, respectively. The scaling is applied inside the mainloop. Details with figures are [here](https://github.com/NVIDIA/cutlass/pull/1932#issue-2645398439).
* [Distributed GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm/65_distributed_gemm.cu) is a new (experimental) API which can turn existing CUTLASS GEMM kernels into pipelined Tensor Parallel GEMMs that run efficiently on NVLink-based network of GPUs. Its pipelining schedules can hide most of the communication behind computation, and relies on point-to-point communication, which can simply use CUDA runtime's peer device access feature. It also utilizes remote TMA loads and memcopies with CUDA graphs to handle communication primarily through the Copy Engine, leaving all SMs free for Hopper's persistent kernels. For more details you can refer to the [DistGEMM blog post](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b).
* Improved persistent grid launch for Hopper kernels with large cluster sizes (>= size of 4) using the new `make_kernel_hardware_info` API as shown in [example 48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
* Enabled high precision accumulation for Hopper FP8 Sparse GEMM.
* Potential API breaking changes:

  + Fix `cute::UniversalCopy` for type safety.
  + No longer implicitly select `cute::SM80_CP_ASYNC_*` based on input tensors. This avoids implicit downstream synchronization requirements. To use `SM80_CP_ASYNC`, users must explicitly select the appropriate CopyAtom.
  + Fix `cute::SM80_CP_ASYNC_CACHEALWAYS`, `cute::SM80_CP_ASYNC_CACHEGLOBAL`, `cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL`, `cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` to avoid implicitly selecting `ZFILL` behavior on predication.
  + Remove `cute::copy_vec<T>` in favor of `cute::copy_aligned` and `cute::copy(AutoVectorizingCopyWithAssumedAlignment<NumBits>,...)`.
  + A refactor of default epilogue struct `DefaultEpilogue` [API](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/default_epilogue.hpp) to avoid reading non-void `ElementC` value for `ElementC = void` kernel.
* New CUTLASS profiler flags: `profiling-duration`, `min-iterations`, and `kernels-file` documented in [profiler.md](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#cutlass-profiler).
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.6.

## [3.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.6.0) (2024-10-03)

* [Hopper structured sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/62_hopper_sparse_gemm/62_hopper_sparse_gemm.cu).

  + [FP16](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_f16_f16_f32_tensor_op_f32.cu)
  + [FP8](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_f8_f8_f32_tensor_op_f32.cu)
  + [INT8](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_s8_s8_s32_tensor_op_s32.cu)
  + [TF32](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_tf32_tf32_f32_tensor_op_f32.cu)
* A refactor to the CUTLASS 3.x convolution `kernel::ConvUniversal` [API](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/kernel/sm90_implicit_gemm_tma_warpspecialized.hpp) to bring it in line with `gemm::GemmUniversal`. Now the 3.x convolution API is no longer considered as a beta API.
* [An improved mixed input GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm/README.md) and a [lookup table implementation](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) for `INT4`x`FP8` scale-only mode.
* [EVT nodes for Top-K selection and softmax](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp) and [GEMM example using those](https://github.com/NVIDIA/cutlass/tree/main/examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu).
* [Programmatic Dependent Launch](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/grid_dependency_control.h) (PDL) that leverages a new Hopper feature to speedup two back-to-back kernels, and its corresponding [documentations](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html).
* [A new debugging tool, synclog](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/synclog.hpp), for dumping out all synchronization events from within a kernel to a file. Please see [synclog documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/utilities.html#debugging-asynchronous-kernels-with-cutlass-s-built-in-synclog-tool) for details.
* A new TMA-enabled [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for grouped GEMM that brings significant performance improvement, as well as its EVT support.
* A SIMT-enabled pointer-array [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm70_epilogue_vectorized_array.hpp).
* A new [Ping-Pong kernel schedule for Grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp) and some other optimizations.
* [A new instantiation strategy for CUTLASS profiler kernels](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_shapes.py) along with [improved documentation for instantiation level in CUTLASS profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html#instantiating-more-kernels-with-hopper).
* A new hardware support for comparisons and computations of [`cutlass::bfloat16_t`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/bfloat16.h)
* Fixed use of isnan on Windows for [`half_t`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/core/functional.cu).
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.6.

## [3.5.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.1) (2024-07-25)

* [Minimal SM90 WGMMA + TMA GEMM example in 100 lines of code](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/wgmma_sm90.cu)
* [Exposure of L2 `cache_hint`s in TMA copy atoms](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/copy_sm90_tma.hpp#L48)
* Exposure of raster order and tile swizzle extent in [CUTLASS library profiler](media/docs/cpp/profiler.html#gemm), and
  [example 48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
* [TMA store based and EVT supported epilogues](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for [Hopper pointer array batched kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_ptr_array.cu).
* A new [`GemmSparseUniversal` API for CUTLASS 2.x Ampere kernels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_sparse_universal.h) to enable serial and parallel split-k for sparse tensor cores and new tiny tile sizes to better support LLM inferrence:

  + [FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cu#L269-L393) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f32t_tensor_op_f32_sparse_sm80.cu#L269-L411).
  + [int8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [int4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [FP32 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32t_f32n_f32t_tensor_op_f32_sparse_sm80.cu#L427-L642) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32n_f32t_f32t_tensor_op_f32_sparse_sm80.cu#L427-L456).
* [CUDA host adapter](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cuda_host_adapter.hpp) extensions to support TMA descriptor construction driver APIs.
* Inclusion of more [Hopper fprop, dgrad, and wgrad convolution kernels in CUTLASS library and profiler](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py).
* Support for residual add (beta != 0) in convolution kernels.
* A new convolution [epilogue](https://github.com/NVIDIA/cutlass/tree/main/examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu#L269) for CUTLASS 2.x to support non-packed NHWC output.
* A refactor of [include files throughout CUTLASS core directories](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_mma_decl.hpp) to reduce circular dependencies and [tests to guard against them](https://github.com/NVIDIA/cutlass/tree/main/test/self_contained_includes/CMakeLists.txt).
* [A guide for setting up VSCode to work well with CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html) and [expanded code style guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html).
* Better support for MSVC as a host compiler.
* Many performance optimizations, improvements, and bug fixes including fixes for FlashAttention-2.
* Optimal code generation with CUDA toolkit versions 12.4 and 12.5u1.

## [3.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.0) (2024-04-09)

* Implicit GEMM Convolutions targeting Hopper SM90A via WGMMA + [TMA im2col](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm90_im2col.hpp)

  + Native implementation in CUTLASS 3.x using CuTe, mirroring the [same design hierarchy as that of GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html).
  + Support for 1D, 2D, and 3D convolutions in a [rank-agnostic fashion](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/convnd_problem_shape.hpp).
  + Support for [Fprop](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/fprop/sm90_conv3d_fprop_implicit_gemm_s8_s8_s32_tensorop_s32.cu), [Dgrad](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/dgrad/sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu), and [Wgrad](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/wgrad/sm90_conv1d_wgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu) algorithms
  + [CUTLASS profiler support](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/conv3x_emitter.py) for 2D and 3D convolutions implemented via the 3.x API.
  + NOTE: this is a beta release. Further updates to CUTLASS will include major performance improvements, feature enablement, and possible breaking changes to the API until 3.7 release. Your feedback is welcome on the design!
* Support for [Ada (SM89) FP8 tensor cores via the 2.x API](https://github.com/NVIDIA/cutlass/tree/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu). Requires CUDA 12.4 or newer.
* [Ampere gather/scatter convolution example](https://github.com/NVIDIA/cutlass/tree/main/examples/59_ampere_gather_scatter_conv/README.md) in CuTe and CUTLASS 3.x

  + Showcasing how custom kernels can be written and optimized using CUTLASS 3.x and CuTe and the general strategy for implementing convolutions as specializations of GETTs.
  + Implementation of a coarse grained sparse gather/scatter kernel achieving peak performance on Ampere class tensor cores.
* 32x and 16x tile sizes are added to CUTLASS 2.x to improve the performance of narrow-tall and wide-short matrices.

  + [Ampere FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm80.cu) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm80.cu#L227-L301), [Ampere INT8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm80.cu#L392-L1342), [Ampere INT4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu#L372-L934).
  + [Turing FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm75.cu#L55-L394), [Turing INT8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm75.cu#L166-L537), [Turing INT4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm75.cu#L310-L564).
* Updates to CuTe documentation for [`cute::Tensor<>`](media/docs/cpp/cute/03_tensor.html), [MMA atoms](media/docs/cpp/cute/0t_mma_atom.html), and an overhauled [CuTe GEMM tutorial series](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial).
* Extensions to CuTe to support [L2 prefetching](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/prefetch.hpp) and [TMA store+reductions](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/copy_sm90_tma.hpp#L1337).
* Remove C++11 requirement on a few CUTLASS 2.x API header files. All CUTLASS files now require C++17.
* Fixes to greatly reduce build warnings.
* Updates and bugfixes from the community (thanks!)

## [3.4.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.1) (2024-02-14)

* Statically available [CUTLASS Version macros](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/version.h) that allow for handling API changes between CUTLASS releases on the users' side.
* Improvements for Hopper [Group-GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm) and [Pointer-Array Batched GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/56_hopper_ptr_array_batched_gemm).
* Updates and bugfixes from the community (thanks!).

## [3.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.0) (2024-01-12)

* Expanded [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) support covering {16-bit, 8-bit} x {8-bit, 4-bit} input types with fast numerical converters and group scaling factors.
* Performance improvements to [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm)
* Beta release of [Pointer-Array Batched GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/56_hopper_ptr_array_batched_gemm) now available on Hopper GPUs utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* Beta release of [Group-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm) utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* [Ampere Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm_with_visitor.cu) supports Epilogue Visitor Tree (EVT) now.
* NamedBarriers usability improvement and list of [ReservedNamedBarriers](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/barrier.h) has been officially released.
* Improved CuTe documentation including improved clarity and depth of [Quickstart](media/docs/cpp/cute/00_quickstart.html), [CuTe Layout](media/docs/cpp/cute/01_layout.html), and [CuTe Layout Algebra](media/docs/cpp/cute/02_layout_algebra.html). Associated code comments, post-conditions, and details in [CuTe Core Unit Tests](#./test/unit/cute/core/) also improved.

## [3.3](https://github.com/NVIDIA/cutlass/releases/tag/v3.3.0) (2023-10-31)

* [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) support covering 16-bit x 8-bit input operand types.
* [Mixed-input Ampere GEMMs](https://github.com/NVIDIA/cutlass/pull/1084) with support for canonical layouts (TN). The implementation supports upcast on operandB {fp16, bf16} x {s8, u8}, and upcast on operandA {s8, u8} x {fp16, bf16}.
* [Copy Async based Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_alignx_tensor_op_f32_warpspecialized_cooperative.cu) - which support lower than 16B aligned input tensors.
* Kernel schedules and Builder support for mixed precision and Copy Async GEMMs with < 16B aligned input tensors.
* Profiler support for lower-aligned Hopper GEMMs.
* Performance Improvements to [Scatter-Gather Hopper Example](https://github.com/NVIDIA/cutlass/tree/main/examples/52_hopper_gather_scatter_fusion).
* Sub-Byte type fixes and improvements.
* EVT Support for RELU with Aux bitmap tensor store (used in dRELU). See [SM90 EVT fusions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp) for details.
* Fusion support for backprop fusions including drelu, dgelu, and dbias.
* Support for void-C kernels and SM80 mixed-input GEMMs in the CUTLASS Python interface

## [3.2.2](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.2) (2023-10-25)

* Minor patch for issue/1138

## [3.2.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.1) (2023-09-22)

* Python support SM90 Epilogue Visitor Tree (EVT) on top of the C++ support released in 3.2.0.
* SM80 EVT support in C++ and Python.
* Other SM90 epilogue improvements.
* Splitting CUTLASS library into smaller units based on operation, arch and datatypes. See [1105](https://github.com/NVIDIA/cutlass/discussions/1105) for details.
* Making `tools/library/scripts` packageable - `tools/library/scripts` is now moving to `python/cutlass_library`. See the Python [README](https://github.com/NVIDIA/cutlass/tree/main/python/README.md) for details.
* SM90 TF32 kernel improvements for all layouts.
* SM90 rasterization direction support in the CUTLASS profiler.
* Improvement for CUTLASS profiler build times.
* Remove Python-C++ bindings.

## [3.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.0) (2023-08-03)

* New warp-specialized persistent FP8 GEMM kernel [kernel schedules](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) and [mainloops](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp) targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters. An example showcasing [Hopper warp-specialized FP8 GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm). FP8 GEMMs come with a fast accumulation mode. When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision.
* New [Epilogue Visitor Tree (EVT)](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) support for Hopper TMA epilogues. EVTs allows for user-defined customized epilogue fusion patterns without having to write a new epilogue.
* [Stream-K](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp) feature for Hopper. Note that this is only a functional implementation of stream-K, and should not be used for performance comparison. Optimizations are expected in a future release.
* Improved CTA rasterization and support for CTA swizzling for Hopper kernels using the [Tile Scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp).
* Improved performance for [warp-specialized TensorFloat-32 (TF32) GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* [Hopper GEMM+Permute](https://github.com/NVIDIA/cutlass/tree/main/examples/53_hopper_gemm_permute/53_hopper_gemm_permute.cu), an example of fusing tensor reordering (permutation) with GEMM mainloop or epilogue.
* New CUTLASS 2D Convolution Python interface. New [example](https://github.com/NVIDIA/cutlass/tree/main/examples/python/03_basic_conv2d.ipynb) here.
* Support for Windows (MSVC) builds. Tested with Visual Studio 2019 v16.11.27 on Windows 10.0.
* Optimal performance using [**CUDA 12.2u1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.1.0) (2023-04-14)

* New CUTLASS Python interface that aims to provide an ease-of-use interface for instantiating, emitting, compiling, and running CUTLASS kernels via Python. More details [here](https://github.com/NVIDIA/cutlass/tree/main/python/README.md) and new [examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python).
* New [efficient epilogues](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative.cu#L783) using TMA for Hopper.
* Support for [fused epilogues](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative_bias_elementwise.cu), such Bias, ReLU and GELU, using the new efficient epilogues.
* New [warp-specialized TensorFloat-32 (TF32) GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* New [*warp-specialized persistent cooperative*](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) kernel design that allows for larger tile sizes and improves performance on Hopper.
* An [example](https://github.com/NVIDIA/cutlass/tree/main/examples/51_hopper_gett) showcasing GEMM-Like Tensor-Tensor Contraction (GETT) capability on Hopper.
* Epilogue builders. Similar to mainloop builders (see [example 49](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)), epilogue builders aim to generate the best-possible epilogue while exposing incremental opt-ins for greater customization.
* Profiler support for overriding kernel and epilogue builder auto schedules for 3.x API kernels, allowing specific policies to be run in the CUTLASS profiler.
* Performance optimizations for the [*warp-specialized persistent ping-pong*](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) kernel.
* Changes to the [GEMM API 3.x](media/docs/cpp/gemm_api_3x.html), involving the host-facing arguments and the underlying `Params` structs.
* [FMHA Backward Pass](https://github.com/NVIDIA/cutlass/tree/main/examples/41_fused_multi_head_attention/fused_multi_head_attention_backward.cu) from Meta xFormers.
* [Streamk GEMM with Broadcast](https://github.com/NVIDIA/cutlass/tree/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu) enables epilogue broadcast with StreamK GEMM.
* [Batched B2B GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion) now can run multiple Back-to-Back GEMM with the same problem size in parallel.
* [Batched Strided GEMV](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemv.cu) support both row major and column major input matrix.
* [Permute + GEMM fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/39_gemm_permute) can fuse Permute with following GEMM now. Before, we only support fusing GEMM with Permute in the epilogue.
* [Row Broadcast](https://github.com/NVIDIA/cutlass/blob/8236f30675bbe98f81d11c05764b77bfcb25b8cc/include/cutlass/epilogue/threadblock/predicated_tile_iterator_row_broadcast.h) can be fused in the epilogue.
* The GitHub branch is renamed from `master` to `main` in this release.
* Optimal performance using [**CUDA 12.1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.0.0) (2023-01-23)

* [CuTe](media/docs/cpp/cute/00_quickstart.html), a [new core library and backend](#./include/cute) for CUTLASS 3.0 that defines a single Layout vocabulary type and an associated algebra of layouts for a much more expressive and composable abstraction for tensors, sets of parallel agents, and operations by said agents on tensors.
* [A new conceptual operation hierarchy](media/docs/cpp/cutlass_3x_design.html) that replaces the architecture-centric hierarchy of CUTLASS 2.x and [documentation for CUTLASS 3.0's GEMM API changes](media/docs/cpp/gemm_api_3x.html).
* Strict API backwards compatibility that exposes both 2.x and 3.x API kernels through the same [`device::GemmUniversalAdapter`](_downloads/7dc317145f0068ea90d204efe81e20a8/gemm_universal_adapter.h) and [`kernel::GemmUniversal`](_downloads/eaf22d7ee39450a218d827321a044638/gemm_universal.hpp) types, allowing users to include both APIs in the same translation units. More information can be found in the [3.x backwards compatibility section](media/docs/cpp/cutlass_3x_backwards_compatibility.html).
* Updates to [Functionality](media/docs/cpp/functionality.html) which directs users on which kernels are supported via CUTLASS-2 and CUTLASS-3.
* Updates to [Compatibility](overview.html#compatibility) Section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures and [Target Architecture](overview.html#target-architecture).
* New warp-specialized GEMM [kernel schedules](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp) and [mainloops](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters.
* Extensions to CUTLASS profiler to support threadblock cluster shapes in library and profiler tile configurations.
* [CUTLASS library integration](https://github.com/NVIDIA/cutlass/tree/main/tools/library/src/gemm_operation_3x.hpp) for 3.x API kernels built through the new `CollectiveBuilder` API, enabling CUTLASS profiler.
* Support for [Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) through the new 3.0 API with CuTe-based exposure of the Hopper [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) and [WGMMA Tensor Core](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) features.
* Set of examples that demonstrate the usage of the new 3.0 API to easily build GEMM kernels targeting Hopper: examples [48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm), [49](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder), and [50](https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle).

