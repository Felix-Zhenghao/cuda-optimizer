# Blackwell SM120 GEMMs

The NVIDIA RTX 5000 Series GPUs introduce support for new narrow precision (4bit and 6bit) block-scaled and non-block-scaled tensor cores. The PTX ISA has extended the `mma` instructions to support these data formats which are 1x to 4x faster than Ada architectureâs fp8 tensor cores. For more detailed information see [`mma` PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma).

CUTLASS 4.0 has added support for these newly introduced narrow precision GEMMs. Similar to the Blackwell SM100 GEMMs, the SM120 GEMMs can be built using the collective builder interface. See examples in [examples/79\_blackwell\_geforce\_gemm/](#../../examples/79_blackwell_geforce_gemm/) and unit tests listed below.

The data types supported and tensor alignment requirements are the same as the Blackwell SM100 GEMMs. The scale factor layout is also the same as SM100 mentioned above. `OpClassTensorOp` is used for non-blockscaled narrow precision GEMMs and `OpClassBlockScaledTensorOp` is used for blockscaled narrow precision GEMMs.

| Ptx Instruction | Throughput | Notes | Unit Test |
| --- | --- | --- | --- |
| mma.sync.aligned.kind::f8f6f4 | 1x Ada Fp8 Tensor Core(2x for FP32 accumulator) | Mixed precision MMA with A={f4,f6,f8} x B={f4,f6,f8} TN layouts | [unit test](#../../test/unit/gemm/device/sm120_tensorop_gemm/) |
| mma.sync.aligned.kind::mxf8f6f4.block\_scale | 1x Ada Fp8 Tensor Core(2x for FP32 accumulator) | Block scaled mixed precision MMA with A={mxf4,mxf6,mxf8} x B={mxf4,mxf6,mxf8} with TN layouts | [unit test](#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf6_mxf8_f32_f32.cu) |
| mma.sync.aligned.kind::mxf4.block\_scale | 2x Ada Fp8 Tensor Core(4x for FP32 accumulator) | Block scaled MMA with A={mxf4} x B={mxf4} with TN layouts | [unit test](#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32.cu) |
| mma.sync.aligned.kind::mxf4nvf4.block\_scale.scale\_vec::[2X|4X] | 2x Ada Fp8 Tensor Core(4x for FP32 accumulator) | Block scaled MMA with A={mxf4} x B={mxf4} or A={nvf4} x B={nvf4} with TN layouts | [unit test](#../../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu) |

Besides the similarities, there are some key differences from the Blackwell SM100 GEMMs:

## Cluster Size

On Geforce series graphics card, there is no multicast feature therefore the cluster shape is fixed to 1x1x1.

## Tensor Layout

Only TN layout is supported. Matrix A is row major and matrix B is column major.

## Pingpong v.s. cooperative kernel schedule

Similar to Hopperâs warp-group GEMM, SM120 GEMMs support both pingpong and cooperative kernel schedules. Pingpong kernel schedule has two groups of 4 MMA warps working on different output tiles, overlapping the mainloop and epilogue, while the cooperative kernel schedule has only one group of 8 MMA warps working on the same output tile. If `KernelScheduleAuto` is specified, `KernelTmaWarpSpecializedCooperative` will be selected by default.

## Epilogue schedule:

`EpilogueScheduleAuto` must be used.

## Tile size:

Below are tables that summarize the valid tile shapes and dispatch policies for SM120 GEMMs. If the output is `float_6_t`, the tile size in the leading dimension of output tensor must be 128.

**Table 16: Valid Tile Shapes and Dispatch Policies for {float8\_t, float\_6\_t, float\_4\_t} x {float8\_t, float\_6\_t, float\_4\_t} of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 64x64x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 64x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 128x64x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 17: Valid Tile Shapes for nv\_float4\_t x nv\_float4\_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedCooperative` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 18: Valid Tile Shapes and Dispatch Policies for mx\_float4\_t x mx\_float4\_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedCooperative` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

**Table 19: Valid Tile Shapes and Dispatch Policies for mx\_float4\_t x mx\_float4\_t of SM120 GEMMs**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` or `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120` |
| 256x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` |
| 128x128x256 | Y | N | N | N | `KernelTmaWarpSpecializedMxf8f6f4Sm120` or `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120` |

Specialized policies must be used to generate mixed-input-datatype `mx_float4_t` kernels.

**Table 20: Valid Tile Shapes and Dispatch Policies for {mx\_float4\_t, mx\_float6\_t, mx\_float8\_t} x {mx\_float4\_t, mx\_float6\_t, mx\_float8\_t}**

| Mma Tile Shape | TN | TT | NT | NN | Dispatch Policy |
| --- | --- | --- | --- | --- | --- |
| 128x128x128 | Y | N | N | N | `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative` |

### Copyright

Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```text
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

[1]
([1](#id1),[2](#id2))

Only valid as scale factor data types.
