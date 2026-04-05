# Blackwell SM100 GEMMs

[**TLDR; jump to block scaled GEMM example**](#detailed_blockscale_example)

Blackwell SM100 introduces `tcgen05.mma` instructions. `tcgen05.mma` instructions support all legacy types (`tfloat32_t`, `half_t`, `bfloat16_t`, `int8_t`, `uint8_t`) and
the new 4, 6, and 8-bits floating point datatypes with and without scale factors.
This document explains the new `tcgen05.mma` instructions supported by CUTLASS and how one can leverage CUTLASS to create
efficient SM100 GEMM kernels targeting these new mma instructions.

Blackwell SM100 has 7 new `tcgen05.mma` instructions. These instructions are 2x to 4x faster then Hopper Architectureâs WGMMA instructions.

| Ptx Instruction | Throughput | Notes |
| --- | --- | --- |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::tf32 | 2x Hopper Tf32 Tensor Core | MMA with A={tf32} x B={tf32} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::f16 | 2x Hopper Fp16 Tensor Core | MMA with A={f16} x B={f16} or A={bf16} x B={bf16} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::i8 | 2x Hopper I8 Tensor Core | MMA with A={i8} x B={i8} or A={u8} x B={u8} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::f8f6f4 | 2x Hopper Fp8 Tensor Core | Mixed precision MMA with A={f4,f6,f8} x B={f4,f6,f8} TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::mxf8f6f4.block\_scale | 2x Hopper Fp8 Tensor Core | Block scaled mixed precision MMA with A={mxf4,mxf6,mxf8} x B={mxf4,mxf6,mxf8} with TN, NT, TT, NN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::mxf4.block\_scale | 4x Hopper Fp8 Tensor Core | Block scaled MMA with A={mxf4} x B={mxf4} with TN layouts |
| tcgen05.mma(.sp).cta\_group::[1|2].kind::mxf4nvf4.block\_scale.scale\_vec\_size::[2X|4X] | 4x Hopper Fp8 Tensor Core | Block scaled MMA with A={mxf4} x B={mxf4} or A={nvf4} x B={nvf4} with TN layouts |

For more detailed information see [`tcgen05.mma` PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions).

## New in Blackwell SM100

### Block Scaled GEMMs

Instructions with `kind` modifiers `mxf8f6f4`, `mxf4`, and `nvf4mxf4` perform matrix multiplication operations with scale
factors of the form $D = C +( A \times SFA) \* (B \times SFB)$. Scale factors are applied to GEMM-K dimension such that
every 16 or 32 elements of $A$ and $B$ matrices in K dimension have an associated scale factor (32 or 64 elements for sparse as sparse gemm compress 2x along k-dim). For example, an $M\times K$,
$A$ matrix has an associated $M \times \lceil K/32 \rceil$ SFA matrix; and an $N\times K$ $B$, matrix has an associated
$N \times \lceil K/32 \rceil$ SFB matrix. For block scaled GEMMs, an entry of output D matrix is
$D\_{ij} = C\_{ij} + \sum\_{k} (A\_{i,k} \times SFA\_{i,k/SV}) \times (B\_{j,k}\times SFB\_{j,k/SV})$, in index notation, we SV is the scale factor vector size (16 or 32).
Further details can be found in
[PTX documentation on block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-block-scaling).

### Blackwell Narrow Precision Data Types

Narrow-precision `tcgen05.mma` instructions can operate on several 4, 6, and 8-bit data types. Blackwell MMAs can operate
on five different 8-bit floating point values, of which only two (`float_ue8m0_t` and `float_ue4m3_t`) can be used as scale factor data types.
There are two 6-bit floating point types and one 4-bit floating point data type.
See [PTX documentation for narrow precision data types](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats) for details.

**Blackwell Narrow Precision Data Types**

| Data Type | Exponent Bits | Mantissa Bits | Signed | Bit Size |
| --- | --- | --- | --- | --- |
| float\_e4m3\_t | 4 | 3 | Yes | 8 |
| float\_e5m2\_t | 5 | 2 | Yes | 8 |
| float\_e2m3\_t | 2 | 3 | Yes | 6 |
| float\_e3m2\_t | 3 | 2 | Yes | 6 |
| float\_e2m1\_t | 2 | 1 | Yes | 4 |
| float\_ue8m0\_t[[1]](#id3) | 8 | 0 | No | 8 |
| float\_ue4m3\_t[[1]](#id3) | 4 | 3 | No | 8 |

Block scaled MMAs use `mx` and `nv` types which are a pair of float8\_t, float6\_t, float4\_t with 2 of the scale factor data types with a predetermined scale factor vector size. `mx` types follow OCP specification (see [OCP Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)). The following types provided by CUTLASS can be used as inputs to collective builders to generate the block scaled kernels:

**Blackwell Block Scaled Narrow Precision Data Types**

| Mx/Nv Data Type | Scale Factor Type | SF Vector Size (Dense) | SF Vector Size (Sparse) | OCP Compliant |
| --- | --- | --- | --- | --- |
| mx\_float8\_t<Any F8type> | float\_ue8m0\_t | 32 | 64 | Yes |
| mx\_float6\_t<Any F6Type> | float\_ue8m0\_t | 32 | 64 | Yes |
| mx\_float4\_t | float\_ue8m0\_t | 32 | 64 | Yes |
| nv\_float4\_t | float\_ue4m3\_t | 16 | 32 | No |

## Layouts, Tensor Alignment Requirements to Target `tcgen05.mma` Instructions

Tables below list valid data type, and AB layout combinations. Note that the alignment is reported as number of elements. A and B matrix layouts are
represented with T and N. T represents row-major layouts, and N represents column-major layouts. For instance, TN is
row-major A matrix with column-major B matrix.

For legacy types (`tf32`, `f16`, `bf16`, `i8` and `u8`) alignment requirements for A and B matrices are the same as in Hopper.
All four layouts (TT, NN, NT, TT) are supported for all legacy data types.

**Table 1: Valid Data Type, Alignment, and Layout Combinations For MMAs with Legacy Types**

|  | Dense / Sparse | A Type | B Type | AB Layout | A Alignment | B Alignment | Target tcgen05.mma.kind | Unit Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1](#legacy_rows) | Dense | tfloat32\_t | tfloat32\_t | TN, NN, NT, TT | 4 | 4 | tf32 |  |
| [2](#legacy_rows) | Dense | half\_t | half\_t | TN, NN, NT, TT | 8 | 8 | f16 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu) |
| [3](#legacy_rows) | Dense | bfloat16\_t | bfloat16\_t | TN, NN, NT, TT | 8 | 8 | f16 | [Similar to half\_t unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu) |
| [4](#legacy_rows) | Dense | int8\_t | int8\_t | TN, NN, NT, TT | 16 | 16 | i8 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/s8_s8_void_s32.cu) |
| [5](#legacy_rows) | Dense | uint8\_t | uint8\_t | TN, NN, NT, TT | 16 | 16 | i8 | [Similar to int8\_t unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/s8_s8_void_s32.cu) |
| [6](#legacy_rows) | Sparse | tfloat32\_t | tfloat32\_t | TN, NN, NT, TT | 4 (N) / 8 (T) | 4 | tf32 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_f32_f32_f32_f32_f32_tfmma.cu) |
| [7](#legacy_rows) | Sparse | half\_t | half\_t | TN, NN, NT, TT | 8 (N) / 16 (T) | 8 | f16 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_f16_f16_f32_f16_f16_hmma.cu) |
| [8](#legacy_rows) | Sparse | bfloat16\_t | bfloat16\_t | TN, NN, NT, TT | 8 (N) / 16 (T) | 8 | f16 | [Similar to half\_t unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_f16_f16_f32_f16_f16_hmma.cu) |
| [9](#legacy_rows) | Sparse | int8\_t | int8\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 16 | i8 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_s8_s8_s32_s8_s8_imma.cu) |
| [10](#legacy_rows) | Sparse | uint8\_t | uint8\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 16 | i8 | [Similar to int8\_t unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_s8_s8_s32_s8_s8_imma.cu) |

For narrow precision Mmas, not all A/B type, and A/B layout combinations are supported by every `tcgen05.mma` instructions.
Furthermore, tensor copy instructions for subbyte types impose additional alignment requirements while loading narrow-precision
tensors from global memory to shared memory
(see [PTX doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensor-copy-restrictions) for details).

Below tables list valid layout, and alignment values for each A and B data type combination and their target `tcgen05.mma`
instructions supported by CUTLASS.

**Table 2: Valid Data Type, Alignment, and Layout Combinations For Narrow Precision MMAs Without Block Scaling**

|  | Dense / Sparse | A Type | B Type | AB Layout | A Alignment | B Alignment | Target tcgen05.mma.kind | Unit Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1](#nonbs_rows_1_2_3_6) | Dense | float4\_t | float4\_t | TN, NN, NT, TT | 128 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nt_layout.cu)   [NN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nn_layout.cu)   [TT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tt_layout.cu) |
| [2](#nonbs_rows_1_2_3_6) | Dense | float4\_t | float6\_t | TN, NN, NT, TT | 128 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nt_layout.cu)   [NN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nn_layout.cu)   [TT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tt_layout.cu) |
| [3](#nonbs_rows_1_2_3_6) | Dense | float6\_t | float4\_t | TN, NN, NT, TT | 128 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nt_layout.cu)   [NN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nn_layout.cu)   [TT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tt_layout.cu) |
| [4](#nonbs_rows_4_7) | Dense | float4\_t | float8\_t | TN, NN, NT, TT | 128 | 16 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f8_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f8_void_f32_nt_layout.cu) |
| [5](#nonbs_rows_5_8) | Dense | float8\_t | float4\_t | TN, NN, NT, TT | 16 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f8_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f8_f6f4_void_f32_nt_layout.cu) |
| [6](#nonbs_rows_1_2_3_6) | Dense | float6\_t | float6\_t | TN, NN, NT, TT | 128 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nt_layout.cu)   [NN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_nn_layout.cu)   [TT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f6f4_void_f32_tt_layout.cu) |
| [7](#nonbs_rows_4_7) | Dense | float6\_t | float8\_t | TN, NN, NT, TT | 128 | 16 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f8_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f6f4_f8_void_f32_nt_layout.cu) |
| [8](#nonbs_rows_5_8) | Dense | float8\_t | float6\_t | TN, NN, NT, TT | 16 | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f8_f6f4_void_f32_tn_layout.cu)   [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/narrow_precision/f8_f6f4_void_f32_nt_layout.cu) |
| [9](#nonbs_rows_9) | Dense | float8\_t | float8\_t | TN, NN, NT, TT | 16 | 16 | f8f6f4 | [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/f8_f8_void_f32.cu) |
| [10](#nonbs_rows_1_2_3_6) | Sparse | float4\_t | float4\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f4_f4_f32_f16_f16_tn.cu) |
| [11](#nonbs_rows_1_2_3_6) | Sparse | float4\_t | float6\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f4_f6_f32_f16_f16_tn.cu) |
| [12](#nonbs_rows_1_2_3_6) | Sparse | float6\_t | float4\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f6_f4_f32_f16_f16_tn.cu) |
| [13](#nonbs_rows_4_7) | Sparse | float4\_t | float8\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 16 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f4_f8_f32_f16_f16_tn.cu) |
| [14](#nonbs_rows_5_8) | Sparse | float8\_t | float4\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f8_f4_f32_f16_f16_tn.cu) |
| [15](#nonbs_rows_1_2_3_6) | Sparse | float6\_t | float6\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f6_f6_f32_f16_f16_tn.cu) |
| [16](#nonbs_rows_4_7) | Sparse | float6\_t | float8\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 16 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f6_f8_f32_f16_f16_tn.cu) |
| [17](#nonbs_rows_5_8) | Sparse | float8\_t | float6\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 128 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/narrow_precision/sm100_sp_gemm_f8_f6_f32_f16_f16_tn.cu) |
| [18](#nonbs_rows_9) | Sparse | float8\_t | float8\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 16 | f8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm/sm100_sp_gemm_f8_f8_f32_f16_f16_qmma.cu) |
