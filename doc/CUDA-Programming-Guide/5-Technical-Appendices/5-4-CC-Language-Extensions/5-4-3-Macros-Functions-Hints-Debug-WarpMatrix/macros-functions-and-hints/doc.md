## 5.4.7. CUDA-Specific Macros

### 5.4.7.1. `__CUDA_ARCH__`

The macro `__CUDA_ARCH__` represents the [virtual architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros) of the NVIDIA GPU for which the code is being compiled. Its value may differ from the device's actual compute capability. This macro enables the writing of code paths that are specialized for particular GPU architectures, which may be necessary for optimal performance or to use architecture-specific features and instructions. The macro can also be used to distinguish between host and device code.

`__CUDA_ARCH__` is only defined in device code, namely in the `__device__`, `__host__ __device__`, and `__global__` functions. The value of the macro is associated with the `nvcc` option `compute_<version>`, with the relation `__CUDA_ARCH__ = <version> * 10`.

Example:

```
nvcc --generate-code arch=compute_80,code=sm_90 prog.cu
```

defines `__CUDA_ARCH__` as `800`.

---

`__CUDA_ARCH__` **Constraints**

**1.** The type signatures of the following entities shall not depend on whether `__CUDA_ARCH__` is defined, nor on its value.

* `__global__` functions and function templates.
* `__device__` and `__constant__` variables.
* Textures and surfaces.

Example:

```
#if !defined(__CUDA_ARCH__)
    typedef int my_type;
#else
    typedef double my_type;
#endif

__device__ my_type my_var;           // ERROR: my_var's type depends on __CUDA_ARCH__

__global__ void kernel(my_type in) { // ERROR: kernel's type depends on __CUDA_ARCH__
    ...
}
```

**2.** If a `__global__` function template is instantiated and launched from the host, then it must be instantiated with the same template arguments, regardless of whether `__CUDA_ARCH__` is defined or its value.

Example:

```
__device__ int result;

template <typename T>
__global__ void kernel(T in) {
    result = in;
}

__host__ __device__ void host_device_function(void) {
#if !defined(__CUDA_ARCH__)
    kernel<<<1, 1>>>(1); // ERROR: "kernel<int>" instantiation only
                            //        when __CUDA_ARCH__ is undefined!
#endif
}

int main(void) {
    host_device_function();
    cudaDeviceSynchronize();
    return 0;
}
```

**3.** In separate compilation mode, the presence or absence of a function or variable definition with external linkage shall not depend on the definition of `__CUDA_ARCH__` or on its value.

Example:

```
#if !defined(__CUDA_ARCH__)
    void host_function(void) {} // ERROR: The definition of host_function()
                                //        is only present when __CUDA_ARCH__
                                //        is undefined
#endif
```

**4.** In separate compilation, the preprocessor macro `__CUDA_ARCH__` must not be used in headers to prevent objects from having different behaviors. Alternatively, all objects must be compiled for the same virtual architecture. If a weak or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then instances of that function in different objects could conflict if those objects are compiled for different compute architectures.

For example, if a header file `a.h` contains:

```
template<typename T>
__device__ T* get_ptr() {
#if __CUDA_ARCH__ == 900
    return nullptr; /* no address */
#else
    __shared__ T arr[256];
    return arr;
#endif
}
```

Then if `a.cu` and `b.cu` both include `a.h` and instantiate `get_ptr()` for the same type, and `b.cu` expects a non-`NULL` address, and compile with:

```
nvcc -arch=compute_70 -dc a.cu
nvcc -arch=compute_80 -dc b.cu
nvcc -arch=sm_80 a.o b.o

Only one version of the ``get_ptr()`` function is used at link time, so the behavior depends on which version is chosen. To avoid this issue, either ``a.cu`` and ``b.cu`` must be compiled for the same compute architecture, or ``__CUDA_ARCH__`` should not be used in the shared header function.
```

The compiler does not guarantee that a diagnostic will be generated for the unsupported uses of `__CUDA_ARCH__` described above.

### 5.4.7.2. `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__`

The macros `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are defined to identify GPU devices with [architecture-](compute-capabilities.html#compute-capabilities-architecture-specific-features) and [family-](compute-capabilities.html#compute-capabilities-family-specific-features) specific features, respectively. See [Feature Set Compiler Targets](compute-capabilities.html#compute-capabilities-feature-set-compiler-targets) section for more information.

Similarly to `__CUDA_ARCH__`, `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are only defined in the device code, namely in the `__device__`, `__host__ __device__`, and `__global__` functions. The macros are associated with the `nvcc` options `compute_<version>a` and `compute_<version>f`.

```
nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu
```

* `__CUDA_ARCH__ == 1000`.
* `__CUDA_ARCH_SPECIFIC__ == 1000`.
* `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.

```
nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu
```

* `__CUDA_ARCH__ == 1000`.
* `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.
* `__CUDA_ARCH_SPECIFIC__` is not defined.

```
nvcc -arch=sm_100 prog.cu
```

* `__CUDA_ARCH__ == 1000`.
* `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.
* `__CUDA_ARCH_SPECIFIC__` is not defined.

```
nvcc -arch=sm_100a prog.cu
# equivalent to:
nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu
```

* `__CUDA_ARCH__ == 1000`.
* `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.
* `__CUDA_ARCH_SPECIFIC__ == 1000` and `__CUDA_ARCH_SPECIFIC__` not defined are both generated.

### 5.4.7.3. CUDA Feature Testing Macros

`nvcc` provides the following preprocessor macros for feature testing. The macros are defined when a particular feature is supported by the CUDA front-end compiler.

* `__CUDACC_DEVICE_ATOMIC_BUILTINS__`: Supports [device atomic compiler builtins](#built-in-atomic-functions).
* `__NVCC_DIAG_PRAGMA_SUPPORT__`: Supports [diagnostic control pragmas](#nv-diagnostic-pragmas).
* `__CUDACC_EXTENDED_LAMBDA__`: Supports [extended lambdas](cpp-language-support.html#extended-lambdas). Enabled by `--expt-extended-lambda` or `--extended-lambda` flag.
* `__CUDACC_RELAXED_CONSTEXPR__`: Support for [relaxed constexpr functions](cpp-language-support.html#constexpr-functions). Enabled by the `--expt-relaxed-constexpr` flag.

### 5.4.7.4. `__nv_pure__` Attribute

In C/C++, a pure function has no side effects on its parameters and can access global variables, though it does not modify them.

CUDA provides `__nv_pure__` attribute supported for both host and device functions. The compiler translates `__nv_pure__` to the `pure` GNU attribute or to the Microsoft Visual Studio `noalias` attribute.

```
__device__ __nv_pure__
int add(int a, int b) {
    return a + b;
}
```

## 5.4.8. CUDA-Specific Functions

### 5.4.8.1. Address Space Predicate Functions

Address space predicate functions are used to determine the address space of a pointer.

Hint

It is suggested to use the `cuda::device::is_address_from()` and `cuda::device::is_object_from()` functions provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory/is_address_from.html) as a portable and safer alternative to Address Space Predicate intrinsic functions.

```
__device__ unsigned __isGlobal      (const void* ptr);
__device__ unsigned __isShared      (const void* ptr);
__device__ unsigned __isConstant    (const void* ptr);
__device__ unsigned __isGridConstant(const void* ptr);
__device__ unsigned __isLocal       (const void* ptr);
```

The functions return `1` if `ptr` contains the generic address of an object in the specified address space, `0` otherwise. Their behavior is unspecified if the argument is a `NULL` pointer.

* `__isGlobal()`: global memory space.
* `__isShared()`: shared memory space.
* `__isConstant()`: constant memory space.
* `__isGridConstant()`: kernel parameter annotated with `__grid_constant__`.
* `__isLocal()`: local memory space.

### 5.4.8.2. Address Space Conversion Functions

CUDA pointers (`T*`) can access objects regardless of where the objects are stored. For example, an `int*` can access `int` objects whether they reside in global or shared memory.

Address space conversion functions are used to convert between generic addresses and addresses in specific address spaces.
These functions are useful when the compiler cannot determine a pointer's address space, for example, when crossing translation units or interacting with PTX instructions.

```
__device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
__device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
__device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
__device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local
```

```
__device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
__device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
__device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
__device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local
```

As an example of inter-operating with PTX instructions, the `ld.shared.s32 r0, [ptr];` PTX instruction expects `ptr` to refer to the shared memory address space.
A CUDA program with an `int*` pointer to an object in `__shared__` memory needs to convert this pointer to the shared address space before passing it to the PTX instruction by calling `__cvta_generic_to_shared` as follows:

```
__shared__ int smem_var;
smem_var        = 42;
size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
int    output;
asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
assert(output == 42);
```

A common optimization that exploits these address representations is reducing data structure size by leveraging the fact that the address ranges of shared, local, and constant spaces are smaller than 32 bits, which allows storing 32-bit addresses instead of 64-bit pointers and save registers. Additionally, 32-bit arithmetic is faster than 64-bit arithmetic.
To obtain the 32-bit integer representation of these addresses, truncate the 64-bit value to 32 bits by casting from an unsigned 64-bit integer to an unsigned 32-bit integer:

```
__shared__ int smem_var;
uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));
```

To recover a generic address from such a 32-bit representation, zero-extend the address back to an unsigned 64-bit integer and then call the corresponding address space conversion function:

```
size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
assert(generic_ptr == &smem_var);
```

---

### 5.4.8.3. Low-Level Load and Store Functions

```
T __ldg(const T* address);
```

The function `__ldg()` performs a read-only L1/Tex cache load. It supports all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```
T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);
```

The functions perform a load using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```

The functions perform a store using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

### 5.4.8.4. `__trap()`

Hint

It is suggested to use the `cuda::std::terminate()` function provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) ([C++ reference](https://en.cppreference.com/w/cpp/error/terminate.html)) as a portable alternative to `__trap()`.

A trap operation can be initiated by calling the `__trap()` function from any device thread.

```
void __trap();
```

Execution of the kernel is aborted, raising an interrupt in the host program. Calling `__trap()` results in a corrupted CUDA context, causing subsequent CUDA calls and kernel invocations to fail.

### 5.4.8.5. `__nanosleep()`

```
__device__ void __nanosleep(unsigned nanoseconds);
```

The function `__nanosleep(ns)` suspends the thread for a sleep duration of approximately `ns` nanoseconds. The maximum sleep duration is approximately one millisecond.

Example:

The following code implements a mutex with exponential back-off.

```
__device__ void mutex_lock(unsigned* mutex) {
    unsigned ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_unlock(unsigned *mutex) {
    atomicExch(mutex, 0);
}
```

### 5.4.8.6. Dynamic Programming eXtension (DPX) Instructions

The DPX set of functions enables finding minimum and maximum values, as well as fused addition and minimum/maximum for up to three 16- or 32-bit signed or unsigned integer parameters. There is an optional ReLU, namely clamping to zero, feature.

Comparison functions:

* Three parameters. Semantic: `max(a, b, c)`, `min(a, b, c)`.

```
     int __vimax3_s32  (     int,      int,      int);
unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
unsigned __vimax3_u32  (unsigned, unsigned, unsigned);
unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);

     int __vimin3_s32  (     int,      int,      int);
unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
unsigned __vimin3_u32  (unsigned, unsigned, unsigned);
unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);
```

* Two parameters, with ReLU. Semantic: `max(a, b, 0)`, `max(min(a, b), 0)`.

```
     int __vimax_s32_relu  (     int,      int);
unsigned __vimax_s16x2_relu(unsigned, unsigned);

     int __vimin_s32_relu  (     int,      int);
unsigned __vimin_s16x2_relu(unsigned, unsigned);
```

* Three parameters, with ReLU. Semantic: `max(a, b, c, 0)`, `max(min(a, b, c), 0)`.

```
     int __vimax3_s32_relu  (     int,      int,      int);
unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);

     int __vimin3_s32_relu  (     int,      int,      int);
unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);
```

* Two parameters, also returning which parameter was smaller/larger:

```
     int __vibmax_s32  (     int,      int, bool* pred);
unsigned __vibmax_u32  (unsigned, unsigned, bool* pred);
unsigned __vibmax_s16x2(unsigned, unsigned, bool* pred);
unsigned __vibmax_u16x2(unsigned, unsigned, bool* pred);

     int __vibmin_s32  (     int,      int, bool* pred);
unsigned __vibmin_u32  (unsigned, unsigned, bool* pred);
unsigned __vibmin_s16x2(unsigned, unsigned, bool* pred);
unsigned __vibmin_u16x2(unsigned, unsigned, bool* pred);
```

Fused addition and minimum/maximum:

* Three parameters, comparing (first + second) with the third. Semantic: `max(a + b, c)`, `min(a + b, c)`

```
     int __viaddmax_s32  (     int,     int,       int);
unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
unsigned __viaddmax_u32  (unsigned, unsigned, unsigned);
unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);

     int __viaddmin_s32  (     int,     int,       int);
unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
unsigned __viaddmin_u32  (unsigned, unsigned, unsigned);
unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);
```

* Three parameters, with ReLU, comparing (first + second) with the third and a zero. Semantic: `max(a + b, c, 0)`, `max(min(a + b, c), 0)`

```
     int __viaddmax_s32_relu  (     int,      int,      int);
unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);

     int __viaddmin_s32_relu  (     int,      int,      int);
unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);
```

These instructions are hardware-accelerated or software emulated depending on compute capability. See [Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) section for the compute capability requirements.

The full API can be found in [CUDA Math API documentation](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

---

The DPX is an exceptionally useful tool for implementing dynamic programming algorithms such as the Smith-Waterman and Needleman-Wunsch algorithms in genomics and the Floyd-Warshall algorithm in route optimization.

Maximum value of three signed 32-bit integers, with ReLU:

```
int a           = -15;
int b           = 8;
int c           = 5;
int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
int d           = -2;
int e           = -4;
int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0
```

Minimum value of the sum of two 32-bit signed integers, another 32-bit signed integer and a zero (ReLU):

```
int a           = -5;
int b           = 6;
int c           = -2;
int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
int d           = 4;
int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0
```

Minimum value of two unsigned 32-bit integers and determining which value is smaller:

```
unsigned a = 9;
unsigned b = 6;
bool     smaller_value;
unsigned min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true
```

Maximum values of three pairs of unsigned 16-bit integers:

```
unsigned a         = 0x00050002;
unsigned b         = 0x00070004;
unsigned c         = 0x00020006;
unsigned max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006
```

## 5.4.9. Compiler Optimization Hints

Compiler optimization hints decorate code with additional information to help the compiler optimize generated code.

* The built-in functions are always available in the device code.
* Host code support depends on the host compiler.

### 5.4.9.1. `#pragma unroll`

The compiler unrolls small loops with a known trip count by default. However, the `#pragma unroll` directive can be used to control the unrolling of any given loop. This directive must be placed immediately before the loop and only applies to that loop.

An integral constant expression may optionally follow. The following are cases for an integral constant expression:

* If it is absent, the loop will be completely unrolled if its trip count is constant.
* If it evaluates to `0` or `1`, the loop will not be unrolled.
* If it is a non-positive integer or greater than `INT_MAX`, the pragma will be ignored, and a warning will be issued.

Examples:

```
struct MyStruct {
    static constexpr int value = 4;
};

inline constexpr int Count = 4;

__device__ void foo(int* p1, int* p2) {
    // no argument specified, the loop will be completely unrolled
    #pragma unroll
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 2;

    // unroll value = 5
    #pragma unroll (Count + 1)
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 4;

    // unroll value = 1, loop unrolling disabled
    #pragma unroll 1
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 8;

    // unroll value = 4
    #pragma unroll (MyStruct::value)
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 16;

    // negative value, pragma unroll ignored
    #pragma unroll -1
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 2;
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/fPMK55PxE).

### 5.4.9.2. `__builtin_assume_aligned()`

Hint

It is suggested to use the `cuda::std::assume_aligned()` function provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) ([C++ reference](https://en.cppreference.com/w/cpp/memory/assume_aligned.html)) as a portable and safer alternative to the built-in functions.

```
void* __builtin_assume_aligned(const void* ptr, size_t align)
void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)
```

The built-in functions enable the compiler to assume that the returned pointer is aligned to at least `align` bytes.

* The three parameter version enables the compiler to assume that `(char*) ptr - offset` is aligned to at least `align` bytes.

`align` must be a power of two and an integer literal.

Examples:

```
void* res1 = __builtin_assume_aligned(ptr, 32);    // compiler can assume 'res1' is at least 32-byte aligned
void* res2 = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 'res2 = (char*) ptr - 8' is at least 32-byte aligned
```

### 5.4.9.3. `__builtin_assume()` and `__assume()`

```
void __builtin_assume(bool predicate)
void __assume        (bool predicate) // only with Microsoft Compiler
```

The built-in function enables the compiler to assume that the boolean argument is true. If the argument is false at runtime, the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:

```
__device__ bool is_greater_than_zero(int value) {
    return value > 0;
}

__device__ bool f(int value) {
    __builtin_assume(value > 0);
    return is_greater_than_zero(value); // returns true, without evaluating the condition
}
```

### 5.4.9.4. `__builtin_expect()`

```
long __builtin_expect(long input, long expected)
```

The built-in function tells the compiler that `input` is expected to equal `expected`, and returns the value of `input`. It is typically used to provide branch prediction information to the compiler.
It behaves like the C++20 `[[likely]]` and `[[unlikely]]` [attributes](https://en.cppreference.com/w/cpp/language/attributes/likely).

Example:

```
// indicate to the compiler that likely "var == 0"
if (__builtin_expect(var, 0))
    doit();
```

### 5.4.9.5. `__builtin_unreachable()`

```
void __builtin_unreachable(void)
```

The built-in function tells the compiler that the control flow will never reach the point at which the function is called. If the control flow does reach this point at runtime, the program has undefined behavior.

This function is useful for avoiding code generation of unreachable branches and disabling compiler warnings for unreachable code.

Example:

```
// indicates to the compiler that the default case label is never reached.
switch (in) {
    case 1:  return 4;
    case 2:  return 10;
    default: __builtin_unreachable();
}
```

### 5.4.9.6. Custom ABI Pragmas

The `#pragma nv_abi` directive enables applications compiled in [separate compilation](../02-basics/nvcc.html#nvcc-separate-compilation) mode to achieve performance similar to that of [whole program compilation](../02-basics/nvcc.html#nvcc-separate-compilation) by preserving the number of registers used by a function.

The syntax for using this pragma is as follows, where `EXPR` refers to any integral constant expression:

```
#pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)
```

* The arguments that follow `#pragma nv_abi` are optional and may be provided in any order; however, at least one argument is required.
* The `preserve_n` arguments limit the number of registers preserved during a function call:

  + `preserve_n_data(EXPR)` limits the number of data registers.
  + `preserve_n_control(EXPR)` limits the number of control registers.

The `#pragma nv_abi` directive can be placed immediately before a device function declaration or definition.

```
#pragma nv_abi preserve_n_data(16)
__device__ void dev_func();

#pragma nv_abi preserve_n_data(16) preserve_n_control(8)
__device__ int dev_func() {
    return 0;
}
```

Alternatively, it can be placed directly before an indirect function call within a C++ expression statement inside a device function. Note that while indirect function calls to free functions are supported, indirect calls to function references or class member functions are not supported.

```
__device__ int dev_func1();

struct MyStruct {
    __device__ int member_func2();
};

__device__ void test() {
    auto* dev_func_ptr = &dev_func1; // type: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v1 = dev_func_ptr();         // CORRECT, indirect call

    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func1();            // WRONG, direct call; the pragma has no effect
                                     // dev_func1 has type: int(void)

    auto& dev_func_ref = &dev_func1; // type: int (&)(void)
    #pragma nv_abi preserve_n_control(8)
    int v3 = dev_func_ref();         // WRONG, call to a reference
                                     // the pragma has no effect

    auto member_function_ptr = &MyStruct::member_func2; // type: int (MyStruct::*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v4 = member_function_ptr();  // WRONG, indirect call to member function
                                     // the pragma has no effect
}
```

When applied to a device function's declaration or definition, the pragma modifies the custom ABI properties for any calls to that function. When placed at an indirect function call site, it affects the ABI properties only for that specific call. Note that the pragma only affects indirect function calls when placed at a call site; it has no effect on direct function calls.

```
#pragma nv_abi preserve_n_control(8)
__device__ int dev_func3();

__device__ int dev_func4();

__device__ void test() {
    int v1 = dev_func3();            // CORRECT, the pragma affects the direct call

    auto* dev_func_ptr = &dev_func4; // type: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func_ptr();         // CORRECT, the pragma affects the indirect call

    int v3 = dev_func_ptr();         // WRONG, the pragma has no effect
}
```

Note that a program is ill-formed if the pragma arguments for a function declaration and its corresponding definition do not match.

## 5.4.10. Debugging and Diagnostics

### 5.4.10.1. Assertion

```
void assert(int expression);
```

The `assert()` macro stops kernel execution if `expression` is equal to zero. If the program is run within a debugger, a breakpoint is triggered, allowing the debugger to be used to inspect the current state of the device. Otherwise, each thread for which `expression` is equal to zero prints a message to stderr after synchronizing with the host via `cudaDeviceSynchronize()`, `cudaStreamSynchronize()`, or `cudaEventSynchronize()`. The format of this message is as follows:

```
<filename>:<line number>:<function>:
block: [blockIdx.x,blockIdx.y,blockIdx.z],
thread: [threadIdx.x,threadIdx.y,threadIdx.z]
Assertion `<expression>` failed.
```

Execution of the kernel is aborted, raising an interrupt in the host program. The `assert()` macro results in a corrupted CUDA context, causing any subsequent CUDA calls or kernel invocations to fail with `cudaErrorAssert`.

The kernel execution is unaffected if `expression` is different from zero.

For example, the following program from source file `test.cu`

```
#include <assert.h>

 __global__ void testAssert(void) {
     int is_one        = 1;
     int should_be_one = 0;

     // This will have no effect
     assert(is_one);

     // This will halt kernel execution
     assert(should_be_one);
 }

 int main(void) {
     testAssert<<<1,1>>>();
     cudaDeviceSynchronize();
     return 0;
 }
```

will output:

```
test.cu:11: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.
```

Assertions are intended for debugging purposes. Since they can affect performance, it is recommended that they be disabled in production code. They can be disabled at compile time by defining the `NDEBUG` preprocessor macro before including `assert.h` or `<cassert>`, or by using the compiler flag `-DNDEBUG`. Note that the expression should not have side effects; otherwise, disabling the assertion will affect the functionality of the code.

### 5.4.10.2. Breakpoint Function

The execution of a kernel function can be suspended by calling the `__brkpt()` function from any device thread.

```
void __brkpt();
```

### 5.4.10.3. Diagnostic Pragmas

The following pragmas can be used to manage the severity of errors that are triggered when a specific diagnostic message is raised.

```
#pragma nv_diag_suppress
#pragma nv_diag_warning
#pragma nv_diag_error
#pragma nv_diag_default
#pragma nv_diag_once
```

The uses of these pragmas are as follows:

```
#pragma nv_diag_xxx <error_number1>, <error_number2> ...
```

The affected diagnostic is specified using the error number shown in the warning message. Any diagnostic can be changed to an error, but only warnings can have their severity suppressed or restored after being changed to an error. The `nv_diag_default` pragma returns the severity of a diagnostic to the severity that was in effect before any other pragmas were issued, namely, the normal severity of the message as modified by any command-line options. The following example suppresses the `declared but never referenced` warning of `foo()`:

```
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diag_default 177
void bar() {
    int i = 0;
}
```

The following pragmas may be used to save and restore the current diagnostic pragma state:

```
#pragma nv_diagnostic push
#pragma nv_diagnostic pop
```

Examples:

```
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diagnostic pop
void bar() {
    int i = 0; // raise a warning
}
```

Note that these directives only affect the `nvcc` CUDA front-end compiler. They have no effect on the host compiler.

`nvcc` defines the macro `__NVCC_DIAG_PRAGMA_SUPPORT__` when diagnostic pragmas are supported.

