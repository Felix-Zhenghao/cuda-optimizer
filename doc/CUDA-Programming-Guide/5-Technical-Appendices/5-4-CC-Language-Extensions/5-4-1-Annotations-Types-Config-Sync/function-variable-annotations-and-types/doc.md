# 5.4. C/C++ Language Extensions

## 5.4.1. Function and Variable Annotations

### 5.4.1.1. Execution Space Specifiers

The execution space specifiers `__host__`, `__device__`, and `__global__` indicate whether a function executes on the host or the device.

Table 38 Execution Space Specifier

| Execution Space Specifier | Executed on | | Callable from | |
| --- | --- | --- | --- | --- |
| Host | Device | Host | Device |
| `__host__`, no specifier | ✓ | ❌ | ✓ | ❌ |
| `__device__` | ❌ | ✓ | ❌ | ✓ |
| `__global__` | ❌ | ✓ | ✓ | ✓ |
| `__host__ __device__` | ✓ | ✓ | ✓ | ✓ |

---

Constraints for `__global__` functions:

* Must return `void`.
* Cannot be a member of a `class`, `struct`, or `union`.
* Requires an execution configuration as described in [Kernel Configuration](#execution-configuration).
* Does not support recursion.
* Refer to `__global__` [function parameters](cpp-language-support.html#global-function-parameters) for additional restrictions.

Calls to a `__global__` function are asynchronous. They return to the host thread before the device completes execution.

---

Functions declared with `__host__ __device__` are compiled for both the host and the device. The `__CUDA_ARCH__` [macro](#cuda-arch-macro) can be used to differentiate host and device code paths:

```
__host__ __device__ void func() {
#if defined(__CUDA_ARCH__)
    // Device code path
#else
    // Host code path
#endif
}
```

### 5.4.1.2. Memory Space Specifiers

The memory space specifiers `__device__`, `__managed__`, `__constant__`, and `__shared__` indicate the storage location of a variable on the device.

The following table summarizes the memory space properties:

Table 39 Memory Space Specifier

| Memory Space Specifier | Location | Accessible by | Lifetime | Unique instance |
| --- | --- | --- | --- | --- |
| `__device__` | Device global memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](../03-advanced/driver-api.html#driver-api-context) | Per device |
| `__constant__` | Device constant memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](../03-advanced/driver-api.html#driver-api-context) | Per device |
| `__managed__` | Host and Device (automatic) | Host/Device Threads | Program | Per program |
| `__shared__` | Device (streaming multiprocessor) | Block Threads | Block | Block |
| no specifier | Device (registers) | Single Thread | Single Thread | Single Thread |

---

* Both `__device__` and `__constant__` variables can be accessed from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) functions `cudaGetSymbolAddress()`, `cudaGetSymbolSize()`, `cudaMemcpyToSymbol()`, and `cudaMemcpyFromSymbol()`.
* `__constant__` variables are read-only in device code and can only be modified from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

The following example illustrates how to use these APIs:

```
__device__   float device_var       = 4.0f; // Variable in device memory
__constant__ float constant_mem_var = 4.0f; // Variable in constant memory
                                            // For readability, the following example focuses on a device variable.
int main() {
    float* device_ptr;
    cudaGetSymbolAddress((void**) &device_ptr, device_var);        // Gets address of device_var

    size_t symbol_size;
    cudaGetSymbolSize(&symbol_size, device_var);                   // Retrieves the size of the symbol (4 bytes).

    float host_var;
    cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // Copies from device to host.

    host_var = 3.0f;
    cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // Copies from host to device.
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/vYjP8GGv3).

#### 5.4.1.2.1. `__shared__` Memory

`__shared__` memory variables can have a static size, which is determined at compile time, or a dynamic size, which is determined at kernel launch time. See the [Kernel Configuration](#execution-configuration) section for details on specifying the shared memory size at run time.

Shared memory constraints:

* Variables with a dynamic size must be declared as an external array or as a pointer.
* Variables with a static size cannot be initialized in their declaration.

The following example illustrates how to declare and size `__shared__` variables:

```
extern __shared__ char dynamic_smem_pointer[];
// extern __shared__ char* dynamic_smem_pointer; alternative syntax

__global__ void kernel() { // or a __device__ function
    __shared__ int smem_var1[4];                  // static size
    auto smem_var2 = (int*) dynamic_smem_pointer; // dynamic size
}

int main() {
    size_t shared_memory_size = 16;
    kernel<<<1, 1, shared_memory_size>>>();
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/nPjvd1frb).

#### 5.4.1.2.2. `__managed__` Memory

`__managed__` variables have the following restrictions:

* The address of a `__managed__` variable is not a constant expression.
* A `__managed__` variable shall not have a reference type `T&`.
* The address or value of a `__managed__` variable shall not be used when the CUDA runtime may not be in a valid state, including the following cases:

  + In static/dynamic initialization or destruction of an object with `static` or `thread_local` storage duration.
  + In code that executes after `exit()` has been called. For example, a function marked with `__attribute__((destructor))`.
  + In code that executes when the CUDA runtime may not be initialized. For example, a function marked with `__attribute__((constructor))`.
* A `__managed__` variable cannot be used as an unparenthesized id-expression argument to a `decltype()` expression.
* `__managed__` variables have the same coherence and consistency behavior as specified for [dynamically allocated managed memory](../02-basics/understanding-memory.html#memory-unified-memory).
* See also the restrictions for [local variables](cpp-language-support.html#local-variables).

Here are examples of legal and illegal uses of `__managed__` variables:

```
#include <cassert>

__device__ __managed__ int global_var = 10; // OK

int* ptr = &global_var;                     // ERROR: use of a managed variable in static initialization

struct MyStruct1 {
    int field;
    MyStruct1() : field(global_var) {};
};

struct MyStruct2 {
    ~MyStruct2() { global_var = 10; }
};

MyStruct1 temp1; // ERROR: use of managed variable in dynamic initialization

MyStruct2 temp2; // ERROR: use of managed variable in the destructor of
                 //        object with static storage duration

__device__ __managed__ const int const_var = 10;         // ERROR: const-qualified type

__device__ __managed__ int&      reference = global_var; // ERROR: reference type

template <int* Addr>
struct MyStruct3 {};

MyStruct3<&global_var> temp;     // ERROR: address of managed variable is not a constant expression

__global__ void kernel(int* ptr) {
    assert(ptr == &global_var);  // OK
    global_var = 20;             // OK
}

int main() {
    int* ptr = &global_var;      // OK
    kernel<<<1, 1>>>(ptr);
    cudaDeviceSynchronize();
    global_var++;                // OK
    decltype(global_var) var1;   // ERROR: managed variable used as unparenthesized argument to decltype

    decltype((global_var)) var2; // OK
}
```

### 5.4.1.3. Inlining Specifiers

The following specifiers can be used to control inlining for `__host__` and `__device__` functions:

* `__noinline__`: Instructs `nvcc` not to inline the function.
* `__forceinline__`: Forces `nvcc` to inline the function within a single translation unit.
* `__inline_hint__`: Enables aggressive inlining across translation units when using [Link-Time Optimization](../02-basics/nvcc.html#nvcc-link-time-optimization).

These specifiers are mutually exclusive.

### 5.4.1.4. `__restrict__` Pointers

`nvcc` supports restricted pointers via the `__restrict__` keyword.

Pointer aliasing occurs when two or more pointers refer to overlapping memory regions. This can inhibit optimizations such as code reordering and common sub-expression elimination.

A restrict-qualified pointer is a promise from the programmer that for the lifetime of the pointer, the memory it points to will only be accessed through that pointer. This allows the compiler to perform more aggressive optimizations.

* all threads that access the device function only read from it; or
* at most one thread writes to it, and no other thread reads from it.

The following example illustrates an aliasing issue and demonstrates how using a restricted pointer can help the compiler reduce the number of instructions:

```
__device__
void device_function(const float* a, const float* b, float* c) {
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}
```

Because the pointers `a`, `b`, and `c` may be aliased, any write through `c` could modify elements of `a` or `b`. To guarantee functional correctness, the compiler cannot load `a[0]` and `b[0]` into registers, multiply them, and store the result in both `c[0]` and `c[1]`. This is because the results would differ from the abstract execution model if `a[0]` and `c[0]` were at the same location. The compiler cannot take advantage of the common sub-expression. Similarly, the compiler cannot reorder the computation of `c[4]` with the computations of `c[0]` and `c[1]` because a preceding write to `c[3]` could alter the inputs to the computation of `c[4]`.

By declaring `a`, `b`, and `c` as restricted pointers, the programmer informs the compiler that the pointers are not aliased. This means that writing to `c` will never overwrite the elements of `a` or `b`. This changes the function prototype as follows:

```
__device__
void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
```

Note that all pointer arguments must be restricted for the compiler optimizer to be effective. With the addition of the `__restrict__` keywords, the compiler can reorder and perform common sub-expression elimination at will while maintaining identical functionality to the abstract execution model.

```
__device__
void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    float t0 = a[0];
    float t1 = b[0];
    float t2 = t0 * t1;
    float t3 = a[1];
    c[0]     = t2;
    c[1]     = t2;
    c[4]     = t2;
    c[2]     = t2 * t3;
    c[3]     = t0 * t3;
    c[5]     = t1;
    ...
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/6KeTqarnW).

The result is a reduced number of memory accesses and computations, balanced by an increase in register pressure from caching loads and common sub-expressions in registers.

Since register pressure is a critical issue in many CUDA codes, the use of restricted pointers can negatively impact performance by reducing occupancy.

---

Accesses to `__global__` function `const` pointers marked with `__restrict__` are compiled as read-only cache loads, similar to the [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc) `ld.global.nc` or `__ldg()` [low-level load and store functions](#low-level-load-store-functions) instructions.

```
__global__
void kernel1(const float* in, float* out) {
    *out = *in; // PTX: ld.global
}

__global__
void kernel2(const float* __restrict__ in, float* out) {
    *out = *in;  // PTX: ld.global.nc
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/drsTEPa8s).

### 5.4.1.5. `__grid_constant__` Parameters

Annotating a `__global__` function parameter with `__grid_constant__` prevents the compiler from creating a per-thread copy of the parameter. Instead, all threads in the grid will access the parameter through a single address, which can improve performance.

The `__grid_constant__` parameter has the following properties:

* It has the lifetime of the kernel.
* It is private to a single kernel, meaning the object is not accessible to threads from other grids, including sub-grids.
* All threads in the kernel see the same address.
* It is read-only. Modifying a `__grid_constant__` object or any of its sub-objects, including `mutable` members, is undefined behavior.

Requirements:

* Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types.
* All function declarations must be consistent with any `__grid_constant__` parameters.
* Function template specializations must match the primary template declaration with respect to any `__grid_constant__` parameters.
* Function template instantiations must also match the primary template declaration with respect to any `__grid_constant__` parameters.

Examples:

```
struct MyStruct {
    int         x;
    mutable int y;
};

__device__ void external_function(const MyStruct&);

__global__ void kernel(const __grid_constant__ MyStruct s) {
    // s.x++; // Compile error: tried to modify read-only memory
    // s.y++; // Undefined Behavior: tried to modify read-only memory

    // Compiler will NOT create a per-thread local copy of "s":
    external_function(s);
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/Goq9jrEeo).

### 5.4.1.6. Annotation Summary

The following table summarizes the CUDA annotations and reports which execution space each annotation applies to and where it is valid.

Table 40 Annotation Summary

| Annotation | `__host__` / `__device__` / `__host__  __device__` | `__global__` |
| --- | --- | --- |
| [\_\_noinline\_\_](#inline-specifiers), [\_\_forceinline\_\_](#inline-specifiers), [\_\_inline\_hint\_\_](#inline-specifiers) | Function | ❌ |
| [\_\_restrict\_\_](#restrict) | Pointer Parameter | Pointer Parameter |
| [\_\_grid\_constant\_\_](#grid-constant) | ❌ | Parameter |
| [\_\_launch\_bounds\_\_](#launch-bounds) | ❌ | Function |
| [\_\_maxnreg\_\_](#maximum-number-of-registers-per-thread) | ❌ | Function |
| [\_\_cluster\_dims\_\_](#cluster-dimensions) | ❌ | Function |

## 5.4.2. Built-in Types and Variables

### 5.4.2.1. Host Compiler Type Extensions

The use of non-standard arithmetic types is permitted by CUDA, as long as the host compiler supports it. The following types are supported:

* 128-bit integer type `__int128`.

  + Supported on Linux when the host compiler defines the `__SIZEOF_INT128__` macro.
* 128-bit floating-point types `__float128` and `_Float128` are available on GPU devices with compute capability 10.0 and later. A constant expression of `__float128` type may be processed by the compiler in a floating-point representation with lower precision.

  + Supported on Linux x86 when the host compiler defines the `__SIZEOF_FLOAT128__` or `__FLOAT128__` macros.
* `_Complex` [types](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html) are only supported in host code.

### 5.4.2.2. Built-in Variables

The values used to specify and retrieve the kernel configuration for the grid and blocks along the x, y, and z dimensions are of type `dim3`. The variables used to obtain the block and thread indices are of type `uint3`. Both `dim3` and `uint3` are trivial structures consisting of three unsigned values named `x`, `y`, and `z`. In C++11 and later, the default value of all components of `dim3` is 1.

Built-in device-only variables:

* `dim3 gridDim`: contains the dimensions of the grid, namely the number of thread blocks, along the x, y, and z dimensions.
* `dim3 blockDim`: contains the dimensions of the thread block, namely the number of threads, along the x, y, and z dimensions.
* `uint3 blockIdx`: contains the block index within the grid, along the x, y, and z dimensions.
* `uint3 threadIdx`: contains the thread index within the block, along the x, y, and z dimensions.
* `int warpSize` : A run-time value defined as the number of threads in a warp, commonly `32`. See also [Warps and SIMT](../01-introduction/programming-model.html#programming-model-warps-simt) for the definition of a warp.

### 5.4.2.3. Built-in Types

CUDA provides vector types derived from basic integer and floating-point types that are supported for both the host and the device. The following table shows the available vector types.

Table 41 Vector Types

| C++ Fundamental Type | Vector X1 | Vector X2 | Vector X3 | Vector X4 |
| --- | --- | --- | --- | --- |
| `signed char` | `char1` | `char2` | `char3` | `char4` |
| `unsigned char` | `uchar1` | `uchar2` | `uchar3` | `uchar4` |
| `signed short` | `short1` | `short2` | `short3` | `short4` |
| `unsigned short` | `ushort1` | `ushort2` | `ushort3` | `ushort4` |
| `signed int` | `int1` | `int2` | `int3` | `int4` |
| `unsigned` | `uint1` | `uint2` | `uint3` | `uint4` |
| `signed long` | `long1` | `long2` | `long3` | `long4_16a/long4_32a` |
| `unsigned long` | `ulong1` | `ulong2` | `ulong3` | `ulong4_16a/ulong4_32a` |
| `signed long long` | `longlong1` | `longlong2` | `longlong3` | `longlong4_16a/longlong4_32a` |
| `unsigned long long` | `ulonglong1` | `ulonglong2` | `ulonglong3` | `ulonglong4_16a/ulonglong4_32a` |
| `float` | `float1` | `float2` | `float3` | `float4` |
| `double` | `double1` | `double2` | `double3` | `double4_16a/double4_32a` |

Note that `long4`, `ulong4`, `longlong4`, `ulonglong4`, and `double4` have been deprecated in CUDA 13, and may be removed in a future release.

---

The following table details the byte size and alignment requirements of the vector types:

Table 42 Alignment Requirements

| Type | Size | Alignment |
| --- | --- | --- |
| `char1`, `uchar1` | 1 | 1 |
| `char2`, `uchar2` | 2 | 2 |
| `char3`, `uchar3` | 3 | 1 |
| `char4`, `uchar4` | 4 | 4 |
| `short1`, `ushort1` | 2 | 2 |
| `short2`, `ushort2` | 4 | 4 |
| `short3`, `ushort3` | 6 | 2 |
| `short4`, `ushort4` | 8 | 8 |
| `int1`, `uint1` | 4 | 4 |
| `int2`, `uint2` | 8 | 8 |
| `int3`, `uint3` | 12 | 4 |
| `int4`, `uint4` | 16 | 16 |
| `long1`, `ulong1` | 4/8 **\*** | 4/8 **\*** |
| `long2`, `ulong2` | 8/16 **\*** | 8/16 **\*** |
| `long3`, `ulong3` | 12/24 **\*** | 4/8 **\*** |
| `long4`, `ulong4` (deprecated) | 16/32 **\*** | 16 **\*** |
| `long4_16a`, `ulong4_16a` | 16/32 **\*** | 16 |
| `long4_32a`, `ulong4_32a` | 16/32 **\*** | 32 |
| `longlong1`, `ulonglong1` | 8 | 8 |
| `longlong2`, `ulonglong2` | 16 | 16 |
| `longlong3`, `ulonglong3` | 24 | 8 |
| `longlong4`, `ulonglong4` (deprecated) | 32 | 16 |
| `longlong4_16a`, `ulonglong4_16a` | 32 | 16 |
| `longlong4_32a`, `ulonglong4_32a` | 32 | 32 |
| `float1` | 4 | 4 |
| `float2` | 8 | 8 |
| `float3` | 12 | 4 |
| `float4` | 16 | 16 |
| `double1` | 8 | 8 |
| `double2` | 16 | 16 |
| `double3` | 24 | 8 |
| `double4` (deprecated) | 32 | 16 |
| `double4_16a` | 32 | 16 |
| `double4_32a` | 32 | 32 |

**\*** `long` is 4 bytes on C++ LLP64 data model (Windows 64-bit), while it is 8 bytes on C++ LP64 data model (Linux 64-bit).

---

Vector types are structures. Their first, second, third, and fourth components are accessible through the `x`, `y`, `z`, and `w` fields, respectively.

```
int sum(int4 value) {
    return value.x + value.y + value.z + value.w;
}
```

They all have a factory function of the form `make_<type_name>()`; for example:

```
int4 add_one(int x, int y, int z, int w) {
    return make_int4(x + 1, y + 1, z + 1, w + 1);
}
```

If host code is not compiled with `nvcc`, the vector types and related functions can be imported by including the `cuda_runtime.h` header provided in the CUDA toolkit.

## 5.4.3. Kernel Configuration
