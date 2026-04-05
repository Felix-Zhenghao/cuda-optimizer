
**Table 3: Valid Data Type, Alignment, and Layout Combinations for Block Scaled Narrow Precision MMAs**

|  | Dense / Sparse | A Type | B Type | AB Layout | A Alignment | B Alignment | Target tcgen05.mma.kind | Unit Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1](#bs_rows_1) | Dense | nv\_float4\_t | nv\_float4\_t | TN | 32 | 32 | mxf4nvf4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/nvf4_nvf4_bf16_bf16.cu) |
| [2](#bs_rows_2) | Dense | mx\_float4\_t | mx\_float4\_t | TN | 32 | 32 | mxf4, mxf4nvf4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf4_void_f16_tn_layout.cu) |
| [3](#bs_rows_3) | Dense | mx\_float4\_t | mx\_float4\_t | TN, NN, NT, TT | 128 | 128 | mxf8f6f4 | [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf4_void_f16_nt_layout.cu) |
| [4](#bs_rows_4_5_7_8_10) | Dense | mx\_float4\_t | mx\_float6\_t | TN, NN, NT, TT | 128 | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf6_f32_f16_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf6_f32_f16_nt_layout.cu) |
| [5](#bs_rows_4_5_7_8_10) | Dense | mx\_float6\_t | mx\_float4\_t | TN, NN, NT, TT | 128 | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf4_f16_f16_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf4_f16_f16_nt_layout.cu) |
| [6](#bs_rows_6_9_11) | Dense | mx\_float4\_t | mx\_float8\_t | TN, NN, NT, TT | 128 | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf8_bf16_bf16_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf4_mxf8_bf16_bf16_nt_layout.cu) |
| [7](#bs_rows_4_5_7_8_10) | Dense | mx\_float8\_t | mx\_float4\_t | TN, NN, NT, TT | 16 | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf4_f16_bf16_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf4_f16_bf16_nt_layout.cu) |
| [8](#bs_rows_4_5_7_8_10) | Dense | mx\_float6\_t | mx\_float6\_t | TN, NN, NT, TT | 128 | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf6_void_bf16_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf6_void_bf16_nt_layout.cu) |
| [9](#bs_rows_6_9_11) | Dense | mx\_float6\_t | mx\_float8\_t | TN, NN, NT, TT | 128 | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf8_void_f32_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf6_mxf8_void_f32_nt_layout.cu) |
| [10](#bs_rows_4_5_7_8_10) | Dense | mx\_float8\_t | mx\_float6\_t | TN, NN, NT, TT | 16 | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf6_f16_f8_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf6_f16_f8_nt_layout.cu) |
| [11](#bs_rows_6_9_11) | Dense | mx\_float8\_t | mx\_float8\_t | TN, NN, NT, TT | 16 | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf8_void_f8_tn_layout.cu) [NT unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf8_void_f8_nt_layout.cu) |
| [12](#bs_rows_1) | Sparse | nv\_float4\_t | nv\_float4\_t | TN | 32 (N) / 64 (T) | 32 | mxf4nvf4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_nvf4_nvf4_f32_void_f16_o_tnn.cu) |
| [13](#bs_rows_2) | Sparse | mx\_float4\_t | mx\_float4\_t | TN | 32 (N) / 64 (T) | 32 | mxf4, mxf4nvf4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf4_mxf4_f32_f16_f16_o_tnn.cu) |
| [14](#bs_rows_3) | Sparse | mx\_float4\_t | mx\_float4\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf4_mxf4_f32_f16_f16_q_tnt.cu) |
| [15](#bs_rows_4_5_7_8_10) | Sparse | mx\_float4\_t | mx\_float6\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf4_mxf6_f32_f16_f16_q_tnt.cu) |
| [16](#bs_rows_4_5_7_8_10) | Sparse | mx\_float6\_t | mx\_float4\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf6_mxf4_f32_f16_f16_q_tnt.cu) |
| [17](#bs_rows_6_9_11) | Sparse | mx\_float4\_t | mx\_float8\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf4_mxf8_f32_f16_f16_q_tnt.cu) |
| [18](#bs_rows_4_5_7_8_10) | Sparse | mx\_float8\_t | mx\_float4\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf8_mxf4_f32_f16_f16_q_tnt.cu) |
| [19](#bs_rows_4_5_7_8_10) | Sparse | mx\_float6\_t | mx\_float6\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf6_mxf6_f32_f16_f16_q_tnt.cu) |
| [20](#bs_rows_6_9_11) | Sparse | mx\_float6\_t | mx\_float8\_t | TN, NN, NT, TT | 128 (N) / 256 (T) | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf6_mxf8_f32_f16_f16_q_tnt.cu) |
| [21](#bs_rows_4_5_7_8_10) | Sparse | mx\_float8\_t | mx\_float6\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 128 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf8_mxf6_f32_f16_f16_q_tnt.cu) |
| [22](#bs_rows_6_9_11) | Sparse | mx\_float8\_t | mx\_float8\_t | TN, NN, NT, TT | 16 (N) / 32 (T) | 16 | mxf8f6f4 | [TN unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/sm100_bssp_gemm_mxf8_mxf8_f32_f16_f16_q_tnn.cu) |

## MMA tile shapes supported

The alignment restrictions also limit the options for Mma Tile Shapes. Tables below list the supported/valid `MmaTileShape`,
Layout, and Dispatch Policy combinations for each row of [Table 1](#legacy_gemm_table), [Table 2](#non_bs_gemm_table), and [Table 3](#bs_gemm_table).

**Table 4: Valid Tile Shapes and Dispatch Policies for legacy types (All rows of Table 1)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x(4\*MMA-K) | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x64x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x192x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x64x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x192x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x(2/4\*MMA-K) | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 5: Valid Tile Shapes and Dispatch Policies for {float4\_t, float6\_t} x {float4\_t, float6\_t} (Rows 1,2,3,6,10,11,12,and 15 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | N | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | N | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | N | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | N | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 6: Valid Tile Shapes and Dispatch Policies for float8\_t x {float4\_t, float6\_t} (Rows 5,8,14,and 17 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | Y | Y | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 7: Valid Tile Shapes and Dispatch Policies for {float4\_t, float6\_t} x float8\_t (Rows 4,7,13,and 16 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | N | N | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x128x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | N | N | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 8: Valid Tile Shapes and Dispatch Policies for float8\_t x float8\_t (Row 9,18 of Table 2)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 64x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 64x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmSm100` |
| Dense | 2SM | 128x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmSm100` |
| Sparse | 1SM | 128x64x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmSm100` |
| Sparse | 2SM | 256x64x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |
| Sparse | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmSm100` |

**Table 9: Valid Tile Shapes for nv\_float4\_t x nv\_float4\_t (Row 1 and 12 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 1SM | 128x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 1SM | 128x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| Dense | 2SM | 256x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Dense | 2SM | 256x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Dense | 2SM | 256x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |

**Table 10: Valid Tile Shapes and Dispatch Policies for mx\_float4\_t x mx\_float4\_t (Row 2 and 13 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 1SM | 128x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 1SM | 128x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized1SmMxf4Sm100` |
| Dense | 2SM | 256x128x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Dense | 2SM | 256x192x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Dense | 2SM | 256x256x256 | Y | N | N | N | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | N | N | N | `KernelSparseTmaWarpSpecialized2SmNvf4Sm100` |

**Table 11: Valid Tile Shapes and Dispatch Policies for mx\_float4\_t x mx\_float4\_t (Row 3 and 14 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |

**Table 12: Valid Tile Shapes and Dispatch Policies for {mx\_float4\_t, mx\_float6\_t, mx\_float8\_t} x {mx\_float4\_t, mx\_float6\_t} (Rows 4, 5, 7, 8, 10, 15, 16, 18, 19, and 21 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | N | N | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | N | N | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |

**Table 13: Valid Tile Shapes and Dispatch Policies for {mx\_float4\_t, mx\_float6\_t, mx\_float8\_t} x mx\_float8\_t (Rows 6, 9, 11, 17, 20, and 22 of Table 3)**

| Dense / Sparse | 1/2 SM | Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1SM | 128x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 1SM | 128x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x128x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x192x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Dense | 2SM | 256x256x128 | Y | Y | Y | Y | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 1SM | 128x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x128x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x192x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| Sparse | 2SM | 256x256x256 | Y | Y | Y | Y | `KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100` |

## Epilogue config supported

**Table 14: Epilogue Dispatch Policy**

| Dense / Sparse | Legacy / Narrow Precision | 1/2 SM | Epilogue Dispatch Policy |
| --- | --- | --- | --- |
| Dense | Legacy & Narrow Precision | 1SM | `cutlass::epilogue::TmaWarpSpecialized1Sm` |
| Dense | Legacy & Narrow Precision | 1SM | `cutlass::epilogue::NoSmemWarpSpecialized1Sm` |
| Dense | Legacy & Narrow Precision | 2SM | `cutlass::epilogue::TmaWarpSpecialized2Sm` |
| Dense | Legacy & Narrow Precision | 2SM | `cutlass::epilogue::NoSmemWarpSpecialized2Sm` |
| Sparse | Legacy | 1SM | `cutlass::epilogue::TmaWarpSpecialized1Sm` |
| Sparse | Legacy | 1SM | `cutlass::epilogue::NoSmemWarpSpecialized1Sm` |
| Sparse | Legacy | 2SM | `cutlass::epilogue::TmaWarpSpecialized2Sm` |
| Sparse | Legacy | 2SM | `cutlass::epilogue::NoSmemWarpSpecialized2Sm` |
| Sparse | Narrow Precision (nvf4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmNvf4` |
| Sparse | Narrow Precision (nvf4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmNvf4` |
| Sparse | Narrow Precision (mxf4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmMxf4` |
| Sparse | Narrow Precision (mxf4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmMxf4` |
| Sparse | Narrow Precision (mxf8f6f4) | 1SM | `cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4` |
| Sparse | Narrow Precision (mxf8f6f4) | 2SM | `cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4` |

**Table 15: Epilogue PerSmTileShape\_MNK**

| 1/2 SM | MMA tile Shape | PerSmTileShape\_MNK |
| --- | --- | --- |
| 1SM | 64x64xMMA\_TileShape\_K | 64x64xMMA\_TileShape\_K |
| 1SM | 64x128xMMA\_TileShape\_K | 64x128xMMA\_TileShape\_K |
| 1SM | 64x192xMMA\_TileShape\_K | 64x192xMMA\_TileShape\_K |
| 1SM | 64x256xMMA\_TileShape\_K | 64x256xMMA\_TileShape\_K |
| 1SM | 128x64xMMA\_TileShape\_K | 128x64xMMA\_TileShape\_K |
| 1SM | 128x128xMMA\_TileShape\_K | 128x128xMMA\_TileShape\_K |
| 1SM | 128x192xMMA\_TileShape\_K | 128x192xMMA\_TileShape\_K |
| 1SM | 128x256xMMA\_TileShape\_K | 128x256xMMA\_TileShape\_K |
| 2SM | 128x64xMMA\_TileShape\_K | 64x64xMMA\_TileShape\_K |
| 2SM | 128x128xMMA\_TileShape\_K | 64x128xMMA\_TileShape\_K |
| 2SM | 128x192xMMA\_TileShape\_K | 64x192xMMA\_TileShape\_K |
| 2SM | 128x256xMMA\_TileShape\_K | 64x256xMMA\_TileShape\_K |
| 2SM | 256x64xMMA\_TileShape\_K | 128x64xMMA\_TileShape\_K |
| 2SM | 256x128xMMA\_TileShape\_K | 128x128xMMA\_TileShape\_K |
| 2SM | 256x192xMMA\_TileShape\_K | 128x192xMMA\_TileShape\_K |
| 2SM | 256x256xMMA\_TileShape\_K | 128x256xMMA\_TileShape\_K |

MMA\_TileShape\_K is is generally 4 \* MMA-Instruction-K. It depends on the config we defined in MMA tile shapes supported section.

### Auto Kernel Dispatch Policies

In addition to direct dispatch policies listed above, the user can also use auto policies for both non-block scaled narrow-precision
GEMMs (both sparse and dense), and block scaled narrow-precision GEMMs (only dense).

CUTLASS will do its best to find the most efficient kernel for given parameters, however, the preferred method for building
these kernels is to use direct kernel dispatch policies shown in the above tables.

* `cutlass::gemm::collective::KernelScheduleAuto`: For a given Mma Tile Size, data type and layout combinations choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) and 1/2 SM `tcgen05.mma(.sp)`.
* `KernelTmaWarpSpecialized1SmBlockScaledSm100`: Use 1 SM `tcgen05.mma` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
* `KernelTmaWarpSpecialized2SmBlockScaledSm100`: Use 2 SM `tcgen05.mma` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
* `KernelSparseTmaWarpSpecialized1SmBlockScaledSm100`: Use 1 SM `tcgen05.mma.sp` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.
* `KernelSparseTmaWarpSpecialized2SmBlockScaledSm100`: Use 2 SM `tcgen05.mma.sp` instruction and choose instr kind (mxf8f6f4, mxf4, nvf4mxf4) automatically.

Similarly for epilogues, we can use `cutlass::epilogue::collective::EpilogueScheduleAuto`.
