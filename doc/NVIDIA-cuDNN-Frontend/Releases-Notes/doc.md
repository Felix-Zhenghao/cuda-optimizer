[NVIDIA](/NVIDIA)
/
**[cudnn-frontend](/NVIDIA/cudnn-frontend)**
Public

* [Notifications](/login?return_to=%2FNVIDIA%2Fcudnn-frontend) You must be signed in to change notification settings
* [Fork
  146](/login?return_to=%2FNVIDIA%2Fcudnn-frontend)
* [Star
   704](/login?return_to=%2FNVIDIA%2Fcudnn-frontend)

# Releases: NVIDIA/cudnn-frontend

Releases · NVIDIA/cudnn-frontend

## v1.22.0-release

03 Apr 02:24

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.22.0](/NVIDIA/cudnn-frontend/tree/v1.22.0)

[`97f6cb3`](/NVIDIA/cudnn-frontend/commit/97f6cb3b88cacff507cca1280db5650a457d92b3)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.22.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.22.0)

[Latest](/NVIDIA/cudnn-frontend/releases/latest)

[Latest](/NVIDIA/cudnn-frontend/releases/latest)

# cuDNN Frontend v1.22.0 Release Notes

cuDNN Frontend v1.22.0 is the recommended version for [cuDNN 9.20.0](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-20-0) and later releases.

## General Improvements 🚀 🚀

* Introducing PyTorch custom operator wrapping cuDNN's Scaled Dot-Product Attention (SDPA). `scaled_dot_product_attention` as the public entry point, closely
  matching the signature of `torch.nn.functional.scaled_dot_product_attention`.

  ```python
  def scaled_dot_product_attention(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      attn_mask: Optional[torch.Tensor] = None,
      dropout_p: float = 0.0,
      is_causal: bool = False,
      scale: Optional[float] = None,
      enable_gqa: bool = False,
      *,
      diagonal_alignment: int = 0,
      left_bound: int = -1,
      right_bound: int = -1,
      seq_len_q: Optional[torch.Tensor] = None,
      seq_len_kv: Optional[torch.Tensor] = None,
      cumulative_seq_len_q: Optional[torch.Tensor] = None,
      cumulative_seq_len_kv: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
  ```
* Introduce a preindexed execute method, that reduces the CPU execution overhead.
* Improve the reproducer tool to report and reproduce SDPA failures for fp8 data types as well.
* 🕒 We will be rolling out new native custom torch ops in upcoming releases – stay tuned! 😃

## Open-Source Kernels 🚀 🚀

* Blackwell sdpa bprop kernel supporting head dim = 256, written in cuteDSL. Support added through the torch-op above or callable as a standalone API. See [samples](/NVIDIA/cudnn-frontend/blob/v1.22.0/test/python/fe_api/test_sdpa_bwd.py) for the API usage. Requires `nvidia-cutlass-dsl[cu13]==4.4.1`
* Grouped Gemm + quantize kernels now support dynamic shape and layout. This is controllable via an environment toggle.
* Grouped Gemm + Glu/Swiglu now supoprt optional bias fusion in both dense and discrete modes, including partial‑N support and optional bias‑gradient generation for discrete backward paths.

## Updates:

* fp8 datatype with packed variable sequences (THD) is no longer supported for SM90 (Hopper) architecture.
* Fix an issue where sdpa fp8 was failing when used with cuda toolkit 12.9

## Acknowledgements:

Blackwell sdpa bprop kernel supporting head dim = 256, written in cuteDSL kernel was jointly developed by Shengbin Di, Yuxi Chi, and Linfeng Zheng in close collaboration with Alibaba. We would like to extend special thanks to the core contributors from Alibaba: Siyu Wang, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, and Wei Lin for their significant contributions to this work.

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.21.0-release

25 Mar 03:18

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.21.0](/NVIDIA/cudnn-frontend/tree/v1.21.0)

[`7b9b711`](/NVIDIA/cudnn-frontend/commit/7b9b711c22b6823e87150213ecd8449260db8610)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.21.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.21.0)

# cuDNN Frontend v1.21.0 Release Notes ([#213](https://github.com/NVIDIA/cudnn-frontend/pull/213))

cuDNN Frontend v1.21.0 is the recommended version for [cuDNN 9.20.0](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-20-0) and later releases.

## General Improvements 🚀

* Dropped dependency on the CUDA driver API for the frontend library, enabling builds without direct CUDA driver linkage.

## Open-Source Kernels

Added new kernels for the GEMM fusions.

**[Grouped GEMM + GLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/grouped_gemm_glu):** Unified grouped GEMM GLU API supporting dense and discrete MoE weight layouts with optional bias.
**[Grouped GEMM + dGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/grouped_gemm_dglu):** Unified grouped GEMM dGLU backward API supporting dense and discrete MoE weight layouts with optional bias.
**[Discrete Grouped GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/discrete_grouped_gemm_swiglu):** Per-expert-pointer SwiGLU grouped GEMM for MoE workloads without weight packing.
**[Discrete Grouped GEMM + dSwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/discrete_grouped_gemm/discrete_grouped_gemm_dswiglu):** Per-expert-pointer dSwiGLU backward grouped GEMM for MoE workloads without weight packing. Uses dSwiGLU/dGeGLU backward epilogue.
**[Grouped GEMM + dSwiglu](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_dswiglu):** dSwiglu activation fused with Grouped GEMM
**[Grouped GEMM + Quant](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/grouped_gemm/grouped_gemm_quant):** Grouped GEMM with output quantization for MoE FC2/dFC1 workloads

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.20.0 release

16 Mar 18:09

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.20.0](/NVIDIA/cudnn-frontend/tree/v1.20.0)

[`d33027a`](/NVIDIA/cudnn-frontend/commit/d33027a41a93af9c85f089c6364ab415fce98982)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.20.0 release](/NVIDIA/cudnn-frontend/releases/tag/v1.20.0)

cuDNN Frontend v1.20.0 is the recommended version for [cuDNN 9.20.0](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-20-0) and later releases.

## Open-Source Kernels 🚀 🚀

* **Fused RMSNorm + SiLU**: The Fused RMSNorm + SiLU engine implements a single-kernel fusion of RMS normalization followed by SiLU (Swish) activation. It is designed and optimized specifically for the WAN VAE decoder's L2Norm + SiLU pattern on B200, but supports arbitrary problem sizes on SM80 to SM103 GPUs.

## Improvements:

* Allow `GEMM + Amax`, `GEMM + SwiGLU`, `Grouped GEMM + SwiGLU`, `Grouped GEMM + dSwiglu`, and `NSA` kernels to run on GB300.
* Improve the reproducer tool to report and reproduce SDPA failures.

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

🚀
1
 YangXu1990uiuc reacted with rocket emoji

All reactions

* 🚀
  1 reaction

1 person reacted

## v1.19.1 release

11 Mar 05:11

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.19.1](/NVIDIA/cudnn-frontend/tree/v1.19.1)

[`7500fd8`](/NVIDIA/cudnn-frontend/commit/7500fd8427a24a76fadac9f2108106fd22c62737)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.19.1 release](/NVIDIA/cudnn-frontend/releases/tag/v1.19.1)

# cuDNN Frontend v1.19.1 Release

Pinning the pybind version to prevent failures with older versions.

Restore support for cuda-12 toolkit that was accidentally dropped in 1.19.0 release.

# cuDNN Frontend v1.19.0 Release Notes

cuDNN Frontend v1.19.0 is the recommended version for [cuDNN 9.19.1](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-19-1) and later releases.

## Open-Source Kernels 🚀 🚀

* **Blackwell and Hopper SDPA Fprop Kernels**: cuDNN's SDPA Fprop implementation is now open source. This kernel supports causal masking and outputs stats for use in bprop. Additional kernels will be added in future releases.
* **[Grouped GEMM + dSwiGLU Fusion](/NVIDIA/cudnn-frontend/blob/v1.19.1/docs/fe-oss-apis/gemm_fusions/grouped_gemm_swiglu.md)**: A contiguous grouped block-scaled GEMM fused with a dSwiGLU backward epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads.

## General Improvements 🚀

* Removed multiple device queries for SM version during graph validation and replaced with a single query that can be skipped by setting `sm_version` on the cuDNN graph.
* Fixed an issue where enabling logging with CUDA graphs in certain scenarios would cause a crash.
* Significantly reduced the CPU overhead of the cuDNN OSS API by using tvm-ffi.
* We are adding a new cudnn-repro tool to have a standalone reproducer from the cudnn frontend logs. See [details](/NVIDIA/cudnn-frontend/blob/v1.19.1/tools/cudnn_repro)

## Enhancements ✨

### Scaled Dot-Product Attention (SDPA)

* **Support Checks**: Improved support checks for cleaner support surface queries.
* **New API**: Added Python bindings for score-mod bprop function to enable the score bprop API.
* **Stats**: Support independent generation of SDPA stats (LSE, SE, Max) in sdpa fprop (Requires 9.20.0 and up).

### Normalization

* **More Benchmarks**: New normalization [benchmark results](/NVIDIA/cudnn-frontend/blob/v1.19.1/benchmark) posted for GB200, GB300, and H200.

## Benchmarking 📊

* Updated the benchmark results for the SDPA improvements added in cuDNN 9.19.1

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.19.0-release

09 Mar 17:35

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.19.0](/NVIDIA/cudnn-frontend/tree/v1.19.0)

[`df73764`](/NVIDIA/cudnn-frontend/commit/df73764adb98b7445b77662ce07b1ea434c8f3ee)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.19.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.19.0)

# cuDNN Frontend v1.19.0 Release Notes

cuDNN Frontend v1.19.0 is the recommended version for [cuDNN 9.19.1](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-19-1) and later releases.

## Open-Source Kernels 🚀 🚀

* **Blackwell and Hopper SDPA Fprop Kernels**: cuDNN's SDPA Fprop implementation is now open source. This kernel supports causal masking and outputs stats for use in bprop. Additional kernels will be added in future releases.
* **[Grouped GEMM + dSwiGLU Fusion](/NVIDIA/cudnn-frontend/blob/v1.19.0/docs/fe-oss-apis/gemm_fusions/grouped_gemm_swiglu.md)**: A contiguous grouped block-scaled GEMM fused with a dSwiGLU backward epilogue on NVIDIA Blackwell GPUs (SM100+), designed for MoE (Mixture of Experts) workloads.

## General Improvements 🚀

* Removed multiple device queries for SM version during graph validation and replaced with a single query that can be skipped by setting `sm_version` on the cuDNN graph.
* Fixed an issue where enabling logging with CUDA graphs in certain scenarios would cause a crash.
* Significantly reduced the CPU overhead of the cuDNN OSS API by using tvm-ffi.
* We are adding a new cudnn-repro tool to have a standalone reproducer from the cudnn frontend logs. See [details](/NVIDIA/cudnn-frontend/blob/v1.19.0/tools/cudnn_repro)

## Enhancements ✨

### Scaled Dot-Product Attention (SDPA)

* **Support Checks**: Improved support checks for cleaner support surface queries.
* **New API**: Added Python bindings for score-mod bprop function to enable the score bprop API.
* **Stats**: Support independent generation of SDPA stats (LSE, SE, Max) in sdpa fprop (Requires 9.20.0 and up).

### Normalization

* **More Benchmarks**: New normalization [benchmark results](/NVIDIA/cudnn-frontend/blob/v1.19.0/benchmark) posted for GB200, GB300, and H200.

## Benchmarking 📊

* Updated the benchmark results for the SDPA improvements added in cuDNN 9.19.1

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.18.0-release

27 Jan 23:08

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.18.0](/NVIDIA/cudnn-frontend/tree/v1.18.0)

[`b8c0656`](/NVIDIA/cudnn-frontend/commit/b8c0656e6f6c84fc194f4d57329b55d609eff596)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.18.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.18.0)

# cuDNN Frontend v1.18.0 Release Notes

cuDNN Frontend v1.18.0 is the recommended version for [cuDNN 9.18.1](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-18-1) and later releases.

## General Improvements 🚀

* Move away from internally using the v0.x API. Rather, now the cudnn backend API is directly called.
* Improve the execution overhead by caching repeated graph query.

### Open-Source Kernels

New open source kernel for Grouped Gemm and Swiglu fussion

* [Grouped GEMM + SwiGLU](/NVIDIA/cudnn-frontend/blob/v1.18.0/gemm_fusions/grouped_gemm_swiglu.md)

## Enhancements ✨

### Scaled Dot-Product Attention (SDPA)

* **New Features**: Allows support for dynamic shapes for fprop. This will help reduce the graph building across different batch and sequence lengths.
* **Support Surface**:

  + Now allows deterministic bprop for SDPA
  + Added support for bprop for ragged tensors in A100
* **More samples**:

  + Open sourcing our sdpa [test harness](/NVIDIA/cudnn-frontend/blob/v1.18.0/test/python/test_mhas_v2.py). Showcase additional testing for determinism, fp8 sizes for MLA
  + Added samples to showcase chunked prefill.

### Mixture of Expers (MoE)

* **New API**: Added support for `moe_grouped_matmul`. See [cpp sample](/NVIDIA/cudnn-frontend/blob/v1.18.0/samples/cpp/moe_grouped_matmul/moe_grouped_matmul.cpp) and documentation for API reference.

### Matmul

* **More samples**: Open sourcing cudnn`s [fuzzy testing of matmuls](/NVIDIA/cudnn-frontend/blob/v1.18.0/test/python/test_matmul_fuzzer.py)

### Convolution

* **More samples**: Open sourcing cudnn`s [fuzzy testing of convolutions](/NVIDIA/cudnn-frontend/blob/v1.18.0/test/python/test_conv_fuzzer.py)

### Additional Improvements

## Benchmarking 📊

* Updated the benchmark results for the sdpa improvements added in cuDNN 9.18.1

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.17.0-release

20 Dec 00:10

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.17.0](/NVIDIA/cudnn-frontend/tree/v1.17.0)

[`b372d39`](/NVIDIA/cudnn-frontend/commit/b372d39879d44c91a8d5b342022e74802b6a8da2)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.17.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.17.0)

# cuDNN Frontend v1.17.0 Release Notes

cuDNN Frontend v1.17.0 is the recommended version for [cuDNN 9.17.0](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-17-0) and later releases.

## New Features 🚀

### Open-Source Kernels

* **Native Sparse Attention** : The Native Sparse Attention (NSA) module implements Native Sparse attention as described in the [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/pdf/2502.11089). Samples of usage for Blackwell architecture in [test/python/fe\_api/nsa](/NVIDIA/cudnn-frontend/blob/v1.17.0/test/python/fe_api/nsa)
* **Gemm/Swiglu** : Gemm\_Swiglu now supports block-scaled FP8/FP4 datatypes.
  API changes:

  + Output tensors have been renamed from "C" and "Glu" to "AB12" and "C", respectively.
  + "use\_2cta\_intrs" Option has been removed. This will be inferred automatically from tile shape.

## Enhancements ✨

### Scaled Dot-Product Attention (SDPA)

* **More samples**: Open sourcing our sdpa [test harness](/NVIDIA/cudnn-frontend/blob/v1.17.0/test/python/test_mhas_v2.py) and fp8 samples in [test/python/test\_sdpa\_fp8.py](/NVIDIA/cudnn-frontend/blob/v1.17.0/test/python/test_sdpa_fp8.py)

### Additional Improvements

* **Tensor properties**: Added vector Dim and vectorization count to the tensor properties.
* **Graph wrapper**: Fixed an issue in the native graph wrapper that caused `BufferError` in non-pytorch tensors.

## Benchmarking 📊

* Updated the benchmark results for the sdpa improvements added in cuDNN 9.17.0. [GB200](https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/sdpa_benchmark_training/artifacts/sdpa_bf16_benchmark_results_NVIDIA_GB200.png) and [GB300](https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/sdpa_benchmark_training/artifacts/sdpa_bf16_benchmark_results_NVIDIA_GB300.png) data.

## Samples

* \*\* cudnn Llama model \*\*: Added reference implementation of the [Llama model completely in cuDNN](/NVIDIA/cudnn-frontend/blob/v1.17.0/samples/llama).

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.16.1-release

01 Dec 21:33

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.16.1](/NVIDIA/cudnn-frontend/tree/v1.16.1)

[`0258951`](/NVIDIA/cudnn-frontend/commit/0258951d4d512f4714eb1574496f4d57669b1b93)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.16.1-release](/NVIDIA/cudnn-frontend/releases/tag/v1.16.1)

## What's Changed

* Find cudnn libraries with NAMES\_PER\_DIR for python site by [@take-cheeze](https://github.com/take-cheeze) in [#180](https://github.com/NVIDIA/cudnn-frontend/pull/180)
* Dont override if users provide max/sum\_exp shape and stride [#181](https://github.com/NVIDIA/cudnn-frontend/pull/181)
* Fix issues in warmup function leading to error in deserialize. [#183](https://github.com/NVIDIA/cudnn-frontend/pull/183)

## New Contributors

* [@take-cheeze](https://github.com/take-cheeze) made their first contribution in [#180](https://github.com/NVIDIA/cudnn-frontend/pull/180)

**Full Changelog**: [v1.16.0...v1.16.1](https://github.com/NVIDIA/cudnn-frontend/compare/v1.16.0...v1.16.1)

### Contributors

* ![@take-cheeze](img/take-cheeze.png)

take-cheeze

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.16.0-release

07 Nov 04:50

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.16.0](/NVIDIA/cudnn-frontend/tree/v1.16.0)

[`be6c079`](/NVIDIA/cudnn-frontend/commit/be6c079be8aaffa0fc079fcf039887e637c289c7)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.16.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.16.0)

# cuDNN Frontend v1.16.0 Release Notes

cuDNN Frontend v1.16.0 is the recommended version for [cuDNN 9.15.0](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-15-0) and later releases.

## New Features 🚀

### Open-Source Kernels

This release introduces open-source implementations of commonly requested fused kernels for select architectures (Blackwell). These experimental kernels may require additional dependencies such as CuteDSL. The initial release includes:

* [GEMM + Amax](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_amax.html)
* [GEMM + SwiGLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_swiglu.html)

Additional dependencies can be installed optionally using `pip install nvidia-cudnn-frontend[cutedsl]`. Usage examples and detailed documentation are available in the [test/python/fe\_api](/NVIDIA/cudnn-frontend/blob/v1.16.0/test/python/fe_api) directory.

Please submit issue reports for additional kernel requests or bug reports.

## Enhancements ✨

### Scaled Dot-Product Attention (SDPA)

* **Block Mask Support**: Starting with cuDNN 9.14.0, SDPA attributes now support block masks to exclude tiles that do not require computation. Refer to the [sample implementation](/NVIDIA/cudnn-frontend/blob/v1.16.0/samples/cpp/sdpa/fp16_fwd_with_block_mask.cpp) for usage details.
* **Bug Fix**: Resolved an invalid memory access (IMA) issue in SDPA backward propagation (fixed in cuDNN backend version 9.15.1 and later) that occurred when `s_kv` is not a multiple of 128, padding mask is disabled, and operations are performed in CUDA graph replay mode.

### Matrix Multiplication

* **CUDA Graph Compatibility**: Added `BehaviorNote_t::CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY` as a behavior note. This enables filtering of engine configurations (execution plans) that use cuBLAS as a backend, available starting with cuDNN version 9.15.0.

### Additional Improvements

* **Block Scale Quantization**: Added Python bindings for block scale quantize operations ([#173](https://github.com/NVIDIA/cudnn-frontend/issues/173)). Refer to the [sample implementation](/NVIDIA/cudnn-frontend/blob/v1.16.0/test/python/test_block_scale_quantize.py) for usage details.
* **Dependency Optimization**: PyTorch is no longer a required dependency for cuDNN Frontend ([#177](https://github.com/NVIDIA/cudnn-frontend/issues/177)).
* **Tensor Alignment**: Enhanced tensor descriptor API to accept alignment as an attribute ([#153](https://github.com/NVIDIA/cudnn-frontend/issues/153)).
* **Plan Generation Control**: Updated `cudnnGetPlan` API to accept an optional maximum plan count parameter, enabling users to limit the number of plans built and autotuned.

## Benchmarking 📊

* Updated [benchmark/sdpa\_benchmark\_training/benchmark\_single\_sdpa.py](/NVIDIA/cudnn-frontend/blob/v1.16.0/benchmark/sdpa_benchmark_training/benchmark_single_sdpa.py) to use correct parameter names and fixed FLOPS calculations for accurate performance measurements.

## Resolved Issues 🔧

* [#153](https://github.com/NVIDIA/cudnn-frontend/issues/153) - Tensor descriptor alignment support
* [#173](https://github.com/NVIDIA/cudnn-frontend/issues/173) - Block scale quantize Python bindings
* [#177](https://github.com/NVIDIA/cudnn-frontend/issues/177) - PyTorch dependency removal

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions

## v1.15.0-release

10 Oct 18:30

![@Anerudhan](img/Anerudhan.png)
[Anerudhan](/Anerudhan)

[v1.15.0](/NVIDIA/cudnn-frontend/tree/v1.15.0)

[`0b1577c`](/NVIDIA/cudnn-frontend/commit/0b1577c8c83401237d601d0d0db5210506705396)

This commit was created on GitHub.com and signed with GitHub’s **verified signature**.

GPG key ID: B5690EEEBB952194

Verified

[Learn about vigilant mode](https://docs.github.com/github/authenticating-to-github/displaying-verification-statuses-for-all-of-your-commits).

Compare

# Choose a tag to compare

## Sorry, something went wrong.

Filter

Loading

## Sorry, something went wrong.

### Uh oh!

There was an error while loading. Please reload this page.

## No results found

[View all tags](/NVIDIA/cudnn-frontend/tags)

[v1.15.0-release](/NVIDIA/cudnn-frontend/releases/tag/v1.15.0)

## cudnn frontend v1.15 release notes

cudnn frontend v1.15 is the preferred cudnn frontend version for [cuDNN version 9.13.1](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-13-1) and above.

## New API

* Introduced a new `cudnn.Graph` API that enables interoperability between `torch.tensors` and the cudnn frontend API. Sample code for performing a matmul with bias addition:

```
B, M, N, K = 16, 128, 128, 512

a_gpu = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
b_gpu = torch.randn(B, K, N, device="cuda", dtype=torch.bfloat16)
d_gpu = torch.randn(1, M, N, device="cuda", dtype=torch.bfloat16)

with cudnn.Graph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
    inputs=["mm::A", "mm::B", "bias::bias"],
    outputs=["bias::OUT_0"],
) as graph:
    AB = graph.matmul(
        name="mm",
        A=a_gpu,
        B=b_gpu,
    )
    C = graph.bias(name="bias", input=AB, bias=d_gpu)
    C.set_output(True)

c_gpu = graph(a_gpu, b_gpu, d_gpu, handle=handle)
```

All notebooks under [samples/python](/NVIDIA/cudnn-frontend/blob/v1.15.0/samples/python) have been updated to showcase the flexibility of this API.

* cudnn frontend now supports building editable pip wheels in place.
* The cudnn frontend `Graph` now includes a `warmup` method that triggers kernel loading by performing a fake graph capture. This improves the startup time for running the initial kernel in the actual run and prevents deadlocks when used with other modules (e.g., NCCL).

## Improvements

### SDPA

* Introduced `set_score_max` and `set_score_sum_exp` to allow the kernel to output `max attention score` and `sum of exponents`.
* Updated support surface checks. (SDPA bprop does not support the combination of `s_q==1` and `s_kv==1`.)
* SDPA bprop now automatically applies a padding mask if the sequence length is not a multiple of the tile size.

### Matmul

* Added support for `COMPLEX_FP32` and `COMPLEX_FP64` datatypes. (Requires cuDNN v9.14.0 or later.)

### Normalizations

* Updated samples to prioritize `fe::HeurMode_t::A` over `fe::HeurMode_t::FALLBACK`.

### Others

* Added support for a new parameter to enable negative scales in the Block Scale DeQuantize operation.
* Improved logging to clearly illustrate the different stages of graph creation.
* The `swish` function now accepts a `swish_beta` parameter.

## Samples

* Added samples demonstrating how to perform sink attention forward and backward propagation with the C++ API. (Requires cuDNN v9.13.0 or later.)
* Added samples demonstrating "Block Scale Matmul Quantize". (Requires cuDNN v9.14.0 or later.)
* Added a sample demonstrating how ragged (packed) tensors work with cuDNN SDPA ([test\_sdpa\_with\_caching.py](/NVIDIA/cudnn-frontend/blob/v1.15.0/test/python/test_sdpa_with_caching.py)). The sample also demonstrates simple caching and graph capture techniques that can improve execution time.

## Bug Fixes

* Fixed an issue where the SDPA node was accessing tensor dimensions before they were inferred, leading to a crash.

## Benchmarks

* Updated results with cuDNN 9.13.1 for B200 and GB300.

## Issues Resolved

* [#160](https://github.com/NVIDIA/cudnn-frontend/issues/160)
* [#152](https://github.com/NVIDIA/cudnn-frontend/issues/152)

Assets
2

Loading

### Uh oh!

There was an error while loading. Please reload this page.

All reactions
