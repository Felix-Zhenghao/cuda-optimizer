#### 4.11.2.2.1. Encoding a Tensor Map on Device

Previous sections have described how to create a tensor map on the host using the CUDA driver API.

This section explains how to encode a tiled-type tensor map on device. This is useful in situations where the typical
way of transferring the tensor map (using `const __grid_constant__` kernel parameters) is undesirable, for instance,
when processing a batch of tensors of various sizes in a single kernel launch.

The recommended pattern is as follows:

1. Create a tensor map “template”, `template_tensor_map`, using the Driver API on the host.
2. In a device kernel, copy the `template_tensor_map`, modify the copy, store in global memory, and appropriately fence.
3. Use the tensor map in a kernel with appropriate fencing.

The high-level code structure is as follows:

```
// Initialize device context:
CUDA_CHECK(cudaDeviceSynchronize());

// Create a tensor map template using the cuTensorMapEncodeTiled driver function
CUtensorMap template_tensor_map = make_tensormap_template();

// Allocate tensor map and tensor in global memory
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// Fill global buffer with data.
fill_global_buf<<<1, 1>>>(global_buf);

// Define the parameters of the tensor map that will be created on device.
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
p.global_dim[0]     = 256; //
p.global_dim[1]     = 8;   //
p.global_stride[0]  = 256; //
p.element_stride[0] = 1;   //
p.element_stride[1] = 1;   //

// Encode global_tensor_map on device:
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// Use it from another kernel:
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// Check for errors:
CUDA_CHECK(cudaDeviceSynchronize());
```

The following sections describe the high-level steps. Throughout the examples, the following `tensormap_params`
struct contains the new values of the fields to be updated. It is included here to reference when reading the examples.

```
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
```

#### 4.11.2.2.2. Device-side Encoding and Modification of a Tensor Map

The recommended process of encoding a tensor map in global memory proceeds as follows.

1. Pass an existing tensor map, the `template_tensor_map`, to the kernel. In contrast to kernels that use
   the tensor map in a `cp.async.bulk.tensor` instruction, this may be done in any way: a pointer to global
   memory, kernel parameter, a `__const___` variable, and so on.
2. Copy-initialize a tensor map in shared memory with the template\_tensor\_map value.
3. Modify the tensor map in shared memory using the [cuda::ptx::tensormap\_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_replace.html)
   functions. These functions wrap the [tensormap.replace](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)
   PTX instruction, which can be used to modify any field of a tiled-type tensor map, including the
   base address, size, stride, and so on.
4. Using the [cuda::ptx::tensormap\_copy\_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html#tensormap-cp-fenceproxy)
   function, copy the modified tensor map from shared memory to global memory and perform any necessary fencing.

The following code contains a kernel that follows these steps. For completeness, it modifies all the fields
of the tensor map. Typically, a kernel will modify just a few fields.

In this kernel, `template_tensor_map` is passed as a kernel parameter. This is the preferred way of moving `template_tensor_map`
from the host to the device. If the kernel is intended to update an existing tensor map in device memory, it can take a
pointer to the existing tensor map to modify.

Note

The format of the tensor map may change over time. Therefore, the [cuda::ptx::tensormap\_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_replace.html)
functions and corresponding [tensormap.replace.tile](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace)
PTX instructions are marked as specific to sm\_90a. To use them, compile using `nvcc -arch sm_90a ....`.

Tip

On sm\_90a, a zero-initialized buffer in shared memory may also be used as the initial tensor map value. This
enables encoding a tensor map purely on device, without using the driver API to encode the `template_tensor_map value`.

Note

On-device modification is only supported for tiled-type tensor maps; other tensor map types cannot be modified on device. For more
information on the tensor map types, refer to the [Driver API reference](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY).

```
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// launch with 1 warp.
__launch_bounds__(32)
__global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
   __shared__ alignas(128) CUtensorMap smem_tmap;
   if (threadIdx.x == 0) {
      // Copy template to shared memory:
      smem_tmap = template_tensor_map;

      const auto space_shared = ptx::space_shared;
      ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
      // For field .rank, the operand new_val must be ones less than the desired
      // tensor rank as this field uses zero-based numbering.
      ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

      // Set box dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
      // Set global dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
      // Set global stride:
      if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
      // Set element stride:
      if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

      // These constants are documented in this table:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
      auto u8_elem_type = ptx::n32_t<0>{};
      ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
      auto no_interleave = ptx::n32_t<0>{};
      ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
      auto no_swizzle = ptx::n32_t<0>{};
      ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
      auto zero_fill = ptx::n32_t<0>{};
      ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
   }
   // Synchronize the modifications with other threads in warp
   __syncwarp();
   // Copy the tensor map to global memory collectively with threads in the warp.
   // In addition: make the updated tensor map visible to other threads on device that
   // for use with cp.async.bulk.
   ptx::n32_t<128> bytes_128;
   ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}
```

#### 4.11.2.2.3. Usage of a Modified Tensor Map

In contrast to using a tensor map that is passed as a `const __grid_constant__` kernel parameter, using a tensor map in
global memory requires explicitly establishing a release-acquire pattern in the tensor map proxy between the threads
that modify the tensor map and the threads that use it.

The release part of the pattern was shown in the previous section. It is accomplished using
the [cuda::ptx::tensormap.cp\_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html) function.

The acquire part is accomplished using the [cuda::ptx::fence\_proxy\_tensormap\_generic](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/fence.html)
function that wraps the [fence.proxy.tensormap::generic.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
instruction. If the two threads participating in the release-acquire pattern are on the same device, the `.gpu` scope suffices. If the threads are on
different devices, the `.sys` scope must be used. Once a tensor map has been acquired by one thread, it can be used by other threads in the block
after sufficient synchronization, for example, using `__syncthreads()`. The thread that uses the tensor map and the thread that performs the fence
must be in the same block. That is, if the threads are in, for example, two different thread blocks of the same cluster, the same grid, or a
different kernel, synchronization APIs such as `cooperative_groups::cluster` or `grid_group::sync()` or stream-order synchronization do not
suffice to establish ordering for tensor map updates, that is, threads in these other thread blocks still need to acquire the tensor map proxy
at the right scope before using the updated tensor map. If there are no intermediate modifications, the fence does not have to be repeated
before each `cp.async.bulk.tensor` instruction.

The `fence` and subsequent use of the tensor map is shown in the following example.

```
// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence.

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
    // Arrive on barrier. Expect 4 * 128 bytes.
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
```

#### 4.11.2.2.4. Creating a Template Tensor Map Value Using the Driver API

The following code creates a minimal tiled-type tensor map that can be subsequently modified on device.

```
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
```

#### 4.11.2.2.5. Shared-Memory Bank Swizzling

By default, the TMA engine loads data to shared memory in the same order as it is laid out in global memory. However, this
layout may not be optimal for certain shared memory access patterns, as it could cause shared memory bank conflicts. To
improve performance and reduce bank conflicts, we can change the shared memory layout by applying a ‘swizzle pattern’.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a
bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is
used multiple times within a transaction, resulting in reduced bandwidth. See [Shared Memory Access Patterns](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns).

To ensure that data is laid out in shared memory in such a way that user code can avoid shared memory bank conflicts,
the TMA engine can be instructed to ‘swizzle’ the data before storing it in shared memory and ‘unswizzle’ it when copying
the data back from shared memory to global memory. The tensor map encodes the ‘swizzle mode’ indicating which swizzle pattern is used.

Example: Matrix Transpose

An example is the transpose of a matrix where data is mapped from row to column first access. The data is stored row major in
global memory, but we want to also access it column wise in shared memory, which leads to bank conflicts. However, by using
the 128 bytes ‘swizzle’ mode and new shared memory indices, they are eliminated.

In the example, we load an 8x8 matrix of type `int4`, stored as row major in global memory to shared memory. Then, each set of eight
threads loads a row from the shared memory buffer and stores it to a column in a separate transpose shared memory buffer. This
results in an eight-way bank conflict when storing. Finally, the transpose buffer is written back to global memory.

To avoid bank conflicts, the `CU_TENSOR_MAP_SWIZZLE_128B` layout can be used. This layout matches the 128 bytes row length and
changes the shared memory layout in a way that both the column wise and row wise access don’t require the same banks per transaction.

The two tables, [Figure 48](#figure-swizzle-example1) and [Figure 49](#figure-swizzle-example2), below show the normal and the swizzled shared memory layout of the 8x8 matrix of type `int4` and
its transpose matrix. The colors indicate which of the eight groups of four banks the matrix element is mapped to, and
the margin row and margin column list the global memory row and column indices. The entries show the shared memory
indices of the 16-byte matrix elements.

![The shared memory data layout without swizzle](img/Figure-48-In-the-shared-memory-data-layout-without-swizzle-the-shared-memory-indices-are-equivalent-to-the-global-memory.png)

*Figure 48 In the shared memory data layout without swizzle, the shared memory indices are equivalent to the global memory indices.
Per load instruction, one row is read and stored in a column of the transpose buffer. Since all matrix elements of the
column in the transpose fall in the same bank, the store must be serialized, resulting in eight store transactions, giving
an eight-way bank conflict per stored column.*

![The shared memory data layout with CU_TENSOR_MAP_SWIZZLE_128B swizzle.](img/Figure-49-The-shared-memory-data-layout-withCU_TENSOR_MAP_SWIZZLE_128Bswizzle-One-row-is-stored-in-a-column-each-matrix.png)

*Figure 49 The shared memory data layout withCU\_TENSOR\_MAP\_SWIZZLE\_128Bswizzle. One row is stored in a column, each matrix
element is from a different bank for both the rows and columns, and so without any bank conflicts.*

```
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
   }
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     int32_t tensor_coords[2] = { 0, 0 };
     ptx::cp_async_bulk_tensor(
       ptx::space_shared, ptx::space_global,
       &smem_buffer, &tensor_map, tensor_coords,
       cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* Matrix transpose
    *  When using the normal shared memory layout, there are eight
    *  8-way shared memory bank conflict when storing to the transpose.
    *  When enabling the 128-byte swizzle pattern and using the according access pattern,
    *  they are eliminated both for load and store. */
   for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // Wait for shared memory writes to be visible to TMA engine.
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();

   /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
    * it will 'unswizzle' the data. */
   if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer_tr);
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   }

   // Destroy barrier
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {4*8, 8};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```

**Remark.** This example is supposed to show the use of swizzle and ‘as-is’ is not performant nor does it scale beyond the given dimensions.

**Explanation.** During data transfer, the TMA engine shuffles the data according to the swizzle pattern, as described in the following
tables. These swizzle patterns define the mapping of the 16-byte chunks along the swizzle width to subgroups of four banks.
It is of type `CUtensorMapSwizzle` and has four options: none, 32 bytes, 64 bytes and 128 bytes. Note that the shared memory box’s
inner dimension must be less or equal to the span of the swizzle pattern.

The Swizzle Modes

As previously mentioned, there are four swizzle modes. The following tables show the different swizzle patterns, including the relation of the new
shared memory indices. The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

![An Overview of TMA Swizzle Patterns](img/Figure-50-An-Overview-of-TMA-Swizzle-Patterns.png)

*Figure 50 An Overview of TMA Swizzle Patterns*

**Considerations.** When applying a TMA swizzle pattern, it is crucial to adhere to specific memory requirements:

* **Global memory alignment:**
  Global memory must be aligned to 128 bytes.
* **Shared memory alignment:**
  For simplicity shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats. When the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself, there is an offset between the swizzle pattern and the shared memory.
  See [comment](#swizzle-pattern-pointer-offset-computation), below.
* **Inner dimension:**
  The inner dimension of the shared memory block must meet the size requirements specified in [Table 25](#table-swizzle-pattern-properties-and-requirements). If these
  requirements are not met, the instruction is considered invalid. Additionally, if the swizzle width exceeds the inner dimension,
  ensure that the shared memory is allocated to accommodate the full swizzle width.
* **Granularity:**
  The granularity of swizzle mapping is fixed at 16 bytes. This means that data is organized and accessed in chunks
  of 16 bytes, which must be considered when planning memory layout and access patterns.

**Swizzle Pattern Pointer Offset Computation**. Here, we describe how to determine the offset between the swizzle pattern and the shared memory, when the shared memory buffer is not aligned by the number of bytes by which the swizzle pattern repeats itself.
When using TMA, the shared memory is required to be aligned to 128 bytes. To find how many times the shared memory buffer relative to the swizzle pattern is shifted by that, apply the corresponding offset formula.

Table 24 Swizzle Pattern Pointer Offset Formula and Index Relation

| Swizzle Mode | Offset Formula | Index Relation |
| --- | --- | --- |
| CU\_TENSOR\_MAP\_SWIZZLE\_128B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%8` | `smem[y][x] <-> smem[y][((y+offset)%8)^x]` |
| CU\_TENSOR\_MAP\_SWIZZLE\_64B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%4` | `smem[y][x] <-> smem[y][((y+offset)%4)^x]` |
| CU\_TENSOR\_MAP\_SWIZZLE\_32B | `(reinterpret_cast <uintptr_t>(smem_ptr)/128)%2` | `smem[y][x] <-> smem[y][((y+offset)%2)^x]` |

In [Figure 50](#figure-swizzle-overview), this offset represents the initial row offset, thus, in the swizzle index calculation, it is added to the row index `y`.
The following snippet shows how to access the swizzled shared memory in the `CU_TENSOR_MAP_SWIZZLE_128B` mode.

```
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

**Summary.** The following [Table 25](#table-swizzle-pattern-properties-and-requirements) summarizes the requirements and properties of the different swizzle patterns for Compute Capability 9.

Table 25 Requirements and properties of the different swizzle patterns for Compute Capability 9

| Pattern | Swizzle width | Shared box’s inner dimension | Repeats after | Shared memory alignment | Global memory alignment |
| --- | --- | --- | --- | --- | --- |
| CU\_TENSOR\_MAP\_SWIZZLE\_128B | 128 bytes | <=128 bytes | 1024 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_64B | 64 bytes | <=64 bytes | 512 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_32B | 32 bytes | <=32 bytes | 256 bytes | 128 bytes | 128 bytes |
| CU\_TENSOR\_MAP\_SWIZZLE\_NONE (default) |  |  |  | 128 bytes | 16 bytes |

## 4.11.3. Using STAS

CUDA applications using [thread block clusters](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters) may need to move small data elements between thread blocks within the cluster. STAS instructions (CC 9.0+, see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)) enable asynchronous data copies directly from registers to distributed shared memory. STAS is only exposed through a lower-level `cuda::ptx::st_async` API available in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/st_async.html?highlight=st_async#) library.

**Dimensions**. STAS supports copying 4, 8 or 16 bytes.

**Source and destination**. The only direction supported for asynchronous copy operations with STAS is from registers to distributed shared memory. The destination pointer needs to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied.

**Asynchronicity**. Data transfers using STAS are [asynchronous](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async thread operations (see [Async Thread and Async Proxy](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. *Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future*. The completion mechanisms that STAS operations can use to signal that they have completed are [shared memory barriers](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers).

In the following example, we show how to use STAS to implement a producer-consumer pattern within a thread-block cluster. This kernel creates a circular communication pipeline where 8 thread blocks are arranged in a ring, and each block simultaneously:

* Produces data for the next block in the sequence.
* Consumes data from the previous block in the sequence.

To implement this pattern, we need 2 shared memory barriers per thread block, one to notify the consumer block that the data has been copied to the shared memory buffer (`filled`) and one to notify the producer block that the buffer on the consumer is ready to be filled (`ready`).

CUDA C++ `cuda::ptx`

|  |
| --- |
| ``` #include <cooperative_groups.h> #include <cuda/barrier> #include <cuda/ptx>  __global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel()  {     using namespace cooperative_groups;     using namespace cuda::device;     using namespace cuda::ptx;     using barrier_t = cuda::barrier<cuda::thread_scope_block>;      auto cluster = this_cluster();      #pragma nv_diag_suppress static_var_with_dynamic_init     __shared__ int buffer[BLOCK_SIZE];     __shared__ barrier_t filled;     __shared__ barrier_t ready;          // Initialize shared memory barriers.     if (threadIdx.x == 0) {         init(&filled, 1);         init(&ready, BLOCK_SIZE);     }          // Sync cluster to ensure remote barriers are initialized.     cluster.sync();          // Define my own and my neighbor's ranks.     int rk = cluster.block_rank();     int rk_next = (rk + 1) % 8;     int rk_prev = (rk + 7) % 8;            // Get addresses of remote buffer we are writing to and remote barriers of previous and next blocks.     auto buffer_next = cluster.map_shared_rank(buffer, rk_next);     auto bar_next = cluster.map_shared_rank(barrier_native_handle(filled), rk_next);     auto bar_prev = cluster.map_shared_rank(barrier_native_handle(ready), rk_prev);          int phase = 0;     for (int it = 0; it < 1000; ++it) {                  // As producers, send data to our right neighbor.         st_async(&buffer_next[threadIdx.x], rk, bar_next);                  if (threadIdx.x == 0) {             // Thread 0 arrives on local barrier and indicates it expects to receive a certain number of bytes.             mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, barrier_native_handle(filled), sizeof(buffer));         }          // As consumers, wait on local barrier for data from left neighbor to arrive.         while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase, 1000)) {}                  // At this point, the data has been copied to our local buffer.         int r = buffer[threadIdx.x];                  // Use the data to do something.          // As consumers, notify our left neighbor that we are done with the data.         mbarrier_arrive(sem_release, scope_cluster, space_cluster, bar_prev);                  // As producers, wait on local barrier until the right neighbor is ready to receive new data.         while (!mbarrier_try_wait_parity(barrier_native_handle(ready), phase, 1000)) {}         phase ^= 1;     } } ``` |

* Shared memory barriers are initialized by the first thread of each block. Barrier `filled` is initialized to 1 and barrier `ready` is initialized to the number of threads in the block.
* A cluster-wide synchronization is performed to ensure that all barriers are initialized before any thread starts communication.
* Each thread determines its neighbors’ ranks and uses them to map the remote shared memory barriers and the remote shared memory buffer to write data to.
* In each iteration:

  1. As a producer, each thread sends data to its right neighbor.
  2. As a consumer, thread 0 arrives on the local `filled` barrier and indicates it expects to receive a certain number of bytes.
  3. As a consumer, each thread waits on the local `filled` barrier for data from the left neighbor to arrive.
  4. As a consumer, each thread uses the data to do something.
  5. As a consumer, each thread notifies the left neighbor that it is done with the data.
  6. As a producer, each thread waits on the local `ready` barrier until the right neighbor is ready to receive new data.

Note that for each barrier, we need to use the correct space. For mapped remote barriers, we need to use the `space_cluster` space, while for local barriers, we need to use the `space_shared` space.
