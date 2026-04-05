## 5.3.8. Polymorphic Function Wrappers

The `nvfunctional` header provides a polymorphic function wrapper class template, `nvstd::function`. Instances of this class template can store, copy, and invoke any callable target, such as lambda expressions. `nvstd::function` can be used in both host and device code.

Example:

```
#include <nvfunctional>

__host__            int host_function()        { return 1; }
__device__          int device_function()      { return 2; }
__host__ __device__ int host_device_function() { return 3; }

__global__ void kernel(int* result) {
    nvstd::function<int()> fn1 = device_function;
    nvstd::function<int()> fn2 = host_device_function;
    nvstd::function<int()> fn3 = [](){ return 10; };
    *result                    = fn1() + fn2() + fn3();
}

__host__ __device__ void host_device_test(int* result) {
    nvstd::function<int()> fn1 = host_device_function;
    nvstd::function<int()> fn2 = [](){ return 10; };
    *result                    = fn1() + fn2();
}

__host__ void host_test(int* result) {
    nvstd::function<int()> fn1 = host_function;
    nvstd::function<int()> fn2 = host_device_function;
    nvstd::function<int()> fn3 = [](){ return 10; };
    *result                    = fn1() + fn2() + fn3();
}
```

---

Invalid cases:

* Instances of `nvstd::function` in host code cannot be initialized with the address of a `__device__` function or with a functor whose `operator()` is a `__device__` function.
* Similarly, instances of `nvstd::function` in device code cannot be initialized with the address of a `__host__` function or with a functor whose `operator()` is a `__host__` function.
* `nvstd::function` instances cannot be passed from host code to device code (or vice versa) at runtime.
* `nvstd::function` cannot be used in the parameter type of a `__global__` function if the `__global__` function is launched from host code.

Examples of invalid cases:

```
#include <nvfunctional>

__device__ int device_function() { return 1; }
__host__   int host_function() { return 3; }
auto       lambda_host  = [] { return 0; };

__global__ void k() {
    nvstd::function<int()> fn1 = host_function; // ERROR, initialized with address of __host__ function
    nvstd::function<int()> fn2 = lambda_host;   // ERROR, initialized with address of functor with
                                                //        __host__ operator() function
}

__global__ void kernel(nvstd::function<int()> f1) {}

void foo(void) {
    auto lambda_device = [=] __device__ { return 1; };

    nvstd::function<int()> fn1 = device_function; // ERROR, initialized with address of __device__ function
    nvstd::function<int()> fn2 = lambda_device;   // ERROR, initialized with address of functor with
                                                  //        __device__ operator() function
    kernel<<<1, 1>>>(fn2);                        // ERROR, passing nvstd::function from host to device
}
```

---

`nvstd::function` is defined in the `nvfunctional` header as follows:

```
namespace nvstd {

template <typename RetType, typename ...ArgTypes>
class function<RetType(ArgTypes...)> {
public:
    // constructors
    __device__ __host__ function() noexcept;
    __device__ __host__ function(nullptr_t) noexcept;
    __device__ __host__ function(const function&);
    __device__ __host__ function(function&&);

    template<typename F>
    __device__ __host__ function(F);

    // destructor
    __device__ __host__ ~function();

    // assignment operators
    __device__ __host__ function& operator=(const function&);
    __device__ __host__ function& operator=(function&&);
    __device__ __host__ function& operator=(nullptr_t);
    template<typename F>
    __device__ __host__ function& operator=(F&&);

    // swap
    __device__ __host__ void swap(function&) noexcept;

    // function capacity
    __device__ __host__ explicit operator bool() const noexcept;

    // function invocation
    __device__ RetType operator()(ArgTypes...) const;
};

// null pointer comparisons
template <typename R, typename... ArgTypes>
__device__ __host__
bool operator==(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator==(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator!=(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator!=(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

// specialized algorithms
template <typename R, typename... ArgTypes>
__device__ __host__
void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);

} // namespace nvstd
```

## 5.3.9. C/C++ Language Restrictions

### 5.3.9.1. Unsupported Features

* Run-Time Type Information (RTTI) and exceptions are not supported in device code:

  + `typeid` keyword
  + `dynamic_cast` keyword
  + `try/catch/throw` keywords
* `long double` is not supported in device code.
* Trigraphs are not supported on any platform. Digraphs are not supported on Windows.
* User-defined `operator new`, `operator new[]`, `operator delete`, or `operator delete[]` cannot be used to replace the corresponding built-ins provided by the compiler, and it is considered undefined behavior on both host and device.

### 5.3.9.2. Namespace Reservations

Unless otherwise noted, adding definitions to top-level namespaces `cuda::`, `nv::`, or `cooperative_groups::`, or to any nested namespace within them, is undefined behavior. We allow `cuda::` as a subnamespace as depicted below:

Examples:

```
namespace cuda {   // same for "nv" and "cooperative_groups" namespaces

struct foo;        // ERROR, class declaration in the "cuda" namespace

void bar();        // ERROR, function declaration in the "cuda" namespace

namespace utils {} // ERROR, namespace declaration in the "cuda" namespace

} // namespace cuda
```

```
namespace utils {
namespace cuda {

// CORRECT, namespace "cuda" may be used nested within a non-reserved namespace
void bar();

} // namespace cuda
} // namespace utils

// ERROR, Equivalent to adding symbols to namespace "cuda" at global scope
using namespace utils;
```

### 5.3.9.3. Pointers and Memory Addresses

Pointer dereferencing (`*pointer`, `pointer->member`, `pointer[0]`) is allowed only in the same execution space where the associated memory resides. The following cases result in undefined behavior, most often a segmentation fault and application termination.

* Dereferencing a pointer either to [global memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-global-memory), [shared memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory), or [constant memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-constant-memory) on the host.
* Dereferencing a pointer to host memory in device code.

The following restrictions apply to functions:

* It is not allowed to take the address of a `__device__` function in host code.
* The address of a `__global__` function taken in host code cannot be used in device code. Similarly, the address of a `__global__` function taken in device code cannot be used in host code.

The address of a `__device__` or `__constant__` variable obtained through `cudaGetSymbolAddress()` as described in the [Memory Space Specifiers](cpp-language-extensions.html#memory-space-specifiers) section can only be used in host code.

### 5.3.9.4. Variables

#### 5.3.9.4.1. Local Variables

The `__device__`, `__shared__`, `__managed__`, and `__constant__` memory space specifiers are not allowed on non-`extern` variable declarations within a function that executes on the host.

Examples:

```
__host__ void host_function() {
    int x;                   // CORRECT, __host__ variable
    __device__   int y;      // ERROR,   __device__ variable declaration within a host function
    __shared__   int z;      // ERROR,   __shared__ variable declaration within a host function
    __managed__  int w;      // ERROR,   __managed__ variable  declaration within a host function
    __constant__ int h;      // ERROR,   __constant__ variable declaration within a host function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

The `__device__`, `__constant__`, and `__managed__` memory space specifiers are not allowed on variable declarations that are neither `extern` nor `static` within a function that executes on the device.

```
__device__ void device_function() {
    int x;                   // CORRECT, __device__ variable
    __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
    __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
    extern __device__ int k; // CORRECT, extern __device__ variable
}
```

see also the [static variables](#static-variables) section.

#### 5.3.9.4.2. `const`-qualified Variables

A `const`-qualified variable without memory space annotations (`__device__` or `__constant__`) declared at global, namespace, or class scope is considered to be a host variable. Device code cannot contain a reference or take the address of the variable.

The variable may be directly used in device code, if

* it has been initialized with a constant expression before the point of use,
* the type is not `volatile`-qualified, and
* it has one of the following types:

  + built-in integral type, or
  + built-in floating point type, except when the host compiler is Microsoft Visual Studio.

Starting with C++14, it is recommended to use `constexpr` or `inline constexpr` (C++17) variables instead of `const`-qualified ones. `constexpr` variables are not subject to the same type restrictions and can be utilized directly in device code.

`__managed__` variables don't support `const`-qualified types.

Examples:

```
const            int   ConstVar          = 10;
const            float ConstFloatVar     = 5.0f;
inline constexpr float ConstexprFloatVar = 5.0f; // C++17

struct MyStruct {
    static const            int   ConstVar          = 20;
//  static const             float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
    static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
};

extern const int ExternVar;

__device__ void foo() {
    int array1[ConstVar];                     // CORRECT
    int array2[MyStruct::ConstVar];           // CORRECT

    const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
    constexpr float var2 = ConstexprFloatVar; // CORRECT
//  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
//  int&            var4 = ConstVar;           // ERROR, reference to host variable
//  int*            var5 = &ConstVar;          // ERROR, address of host variable
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/eWG8KxK94).

#### 5.3.9.4.3. `volatile`-qualified Variables

Note

The `volatile` keyword is supported to maintain compatibility with ISO C++. However, few, if any, of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reading and writing to `volatile`-qualified objects are not atomic and are compiled into one or more [volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) that do not guarantee:

* ordering of memory operations, or
* that the number of memory operations performed by the hardware matches the number of PTX instructions.

CUDA C++ `volatile` is NOT suitable for:

* **Inter-Thread Synchronization**: Use atomic operations via [cuda::atomic\_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](cpp-language-extensions.html#atomic-functions) instead.

  Atomic memory operations provide inter-thread synchronization guarantees and deliver better performance than `volatile` operations.
  However, CUDA C++ `volatile` operations do not provide any inter-thread synchronization guarantees and are therefore not suitable for this purpose.
  The following example shows how to pass a message between two threads using atomic operations.

  cuda::atomic\_ref

  |  |
  | --- |
  | ``` #include <cuda/atomic>  __global__ void kernel(int* flag, int* data) {     cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref{*flag};     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomic_ref.load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         atomic_ref.store(1, cuda::memory_order_release);     } } ``` |

  cuda::atomic

  |  |
  | --- |
  | ``` #include <cuda/atomic>  __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(flag->load(cuda::memory_order_acquire) == 0)             ;         if (*data != 42)             __trap(); // Errors if wrong data read     }     else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         flag->store(1, cuda::memory_order_release);     } } ``` |

  Atomic Functions (`atomicAdd` and `atomicExch`)

  |  |
  | --- |
  | ``` __global__ void kernel(int* flag, int* data) {     if (threadIdx.x == 0) {         // Consumer: blocks until flag is set by producer, then reads data         while(atomicAdd(flag, 0) == 0)             ;                // Load with Relaxed Read-Modify-Write         __threadfence();     // SequentiallyConsistent fence         if (*data != 42)             __trap();        // Errors if wrong data read     } else if (threadIdx.x == 1) {         // Producer: writes data then sets flag         *data = 42;         __threadfence();     // SequentiallyConsistent fence         atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write     } } ``` |
* **Memory Mapped IO** (MMIO): Use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.

  PTX MMIO operations strictly preserve the number of memory accesses performed.
  However, CUDA C++ `volatile` operations do not preserve the number of memory accesses performed and may perform more or fewer accesses than requested in an undetermined way. This makes them unsuitable for MMIO.
  The following example shows how to read from and write to a register using PTX MMIO operations.

  ```
  __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
      // Write to MMIO register:
      int value = 13;
      asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
          :
          : "l"(mmio_reg0), "r"(value) : "memory");

      // Read MMIO register:
      asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
          : "=r"(value)
          : "l"(mmio_reg1) : "memory");

      if (value != 42)
          __trap(); // Errors if wrong data read
  }
  ```

#### 5.3.9.4.4. `static` Variables

`static` variables are allowed in device code in the following cases:

* Within `__global__` or `__device__`-only functions.
* Within `__host__ __device__` functions:

  + `static` variables without an explicit memory space (automatic deduction).
  + `static` variables with an explicit memory space, such as `static __device__/__constant__/__shared__/__managed__`, are allowed only when `__CUDA_ARCH__` is defined.

A `static` variable within a `__host__ __device__` function holds a different value depending on the execution space.

Examples of legal and illegal uses of function-scope `static` variables are shown below.

```
struct TrivialStruct {
    int x;
};

struct NonTrivialStruct {
    __device__ NonTrivialStruct(int x) {}
};

__device__ void device_function(int x) {
    static int v1;              // CORRECT, implicit __device__ memory space specifier
    static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
//  static int v3 = x;           // ERROR, dynamic initialization is not allowed

    static __managed__  int v4; // CORRECT, explicit
    static __device__   int v5; // CORRECT, explicit
    static __constant__ int v6; // CORRECT, explicit
    static __shared__   int v7; // CORRECT, explicit

    static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
    static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
//  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
//  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/TdYKaTq3f).

---

```
__host__ __device__ void host_device_function() {
    static            int v1; // CORRECT, implicit __device__ memory space specifier
//  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
#ifdef __CUDA_ARCH__
    static __device__ int v3; // CORRECT, declaration is only visible during device compilation
#else
    static int v4;            // CORRECT, declaration is only visible during host compilation
#endif
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/18qhjn8P1).

---

```
#include <cassert>

__host__ __device__ int host_device_function() {
    static int v = 0;
    v++;
    return v;
}

__global__ void kernel() {
    int ret = host_device_function(); // v = 1
    assert(ret == 4);                 // FAIL
}

int main() {
    host_device_function();           // v = 1
    host_device_function();           // v = 2
    int ret = host_device_function(); // v = 3
    assert(ret == 3);                 // OK
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/Wqo9WjvYY).

#### 5.3.9.4.5. `extern` Variables

When compiling in the [whole program compilation mode](../02-basics/nvcc.html#nvcc-separate-compilation), `__device__`, `__shared__`, `__managed__`, and `__constant__` variables cannot be defined with external linkage using the `extern` keyword.

The only exception is for dynamically allocated `__shared__` variables as described in the [Dynamic Allocation of Shared Memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-dynamic-allocation-shared-memory) section.

```
__device__        int x; // OK
extern __device__ int y; // ERROR in whole program compilation mode
extern __shared__ int z; // OK
```

### 5.3.9.5. Functions

#### 5.3.9.5.1. Recursion

`__global__` functions do not support recursion, while `__device__` and `__host__ __device__` functions do not have such restriction.

#### 5.3.9.5.2. External Linkage

Device variables or functions with external linkage require [separate compilation mode](../02-basics/nvcc.html#nvcc-separate-compilation) across multiple translation units.

In separate compilation mode, if a `__device__` or `__global__` function definition is required to exist in a particular translation unit, then the parameters and return types of the function must be complete in that translation unit. The concept is also known as One Definition Rule-use, or ODR-use.

Example:

```
//first.cu:
struct S;                   // forward declaration
__device__ void foo(S);     // ERROR, type 'S' is an incomplete type
__device__ auto* ptr = foo; // ODR-use, address taken

int main() {}
```

```
//second.cu:
struct S {};               // struct definition
__device__ void foo(S) {}  // function definition
```

```
# compiler invocation
$ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                 first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```

#### 5.3.9.5.3. Formal Parameters

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on formal parameters.

```
void device_function1(__device__ int x) { } // ERROR, __device__ parameter
void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter
```

#### 5.3.9.5.4. `__global__` Function Parameters

A `__global__` function has the following restrictions:

* It cannot have a variable number of arguments, namely the C ellipsis syntax `...` and the `va_list` type. C++11 variadic template is allowed, subject to the restrictions described in the [\_\_global\_\_ Variadic Template](#cpp11-variadic-template) section.
* Function parameters are passed to the device via [constant memory](device-callable-apis.html#constant-memory) and their total size is limited to 32,764 bytes.
* Function parameters cannot be pass-by-reference or by pass-by-rvalue reference.
* Function parameters cannot be of type `std::initializer_list`.
* Polymorphic class parameters (`virtual`) are considered undefined behavior.
* Lambda expressions and closure types are allowed, subject to the restrictions described in the [Lambda Expressions and \_\_global\_\_ Function Parameters](#lambda-expressions-global) section.

#### 5.3.9.5.5. `__global__` Function Arguments Passing

When launching a `__global__` function [from device code](../02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-kernels), each argument must be trivially copyable and trivially destructible.

When a `__global__` function is launched from host code, each argument type may be non-trivially copyable or non-trivially destructible. However, the processing of these types does not follow the standard C++ model, as described below. The user code must ensure that this workflow does not affect program correctness. The workflow diverges from standard C++ in two areas:

1. **Raw memory copy instead of copy constructor invocation**

   The CUDA Runtime passes the kernel arguments to the `__global__` function by copying the raw memory content, eventually using `memcpy`. If an argument is non-trivially copyable and provides a user-defined copy constructor, the operations and side effects of the invocation are skipped in the host-to-device copy.

   Example:

   ```
   #include <cassert>

   struct MyStruct {
       int  value = 1;
       int* ptr;

       MyStruct() = default;

       __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
   };

   __global__ void device_function(MyStruct my_struct) {
       // this assert fails because "my_struct" is obtained by copying
       // the raw memory content and the copy constructor is skipped.
       assert(my_struct.ptr == &my_struct.value); // FAIL
   }

   void host_function(MyStruct my_struct) {
       assert(my_struct.ptr == &my_struct.value); // CORRECT
   }

   int main() {
       MyStruct my_struct;
       host_function(my_struct);
       device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
       cudaDeviceSynchronize();
   }
   ```

   See the example on [Compiler Explorer](https://godbolt.org/z/xhqe16dec).
2. **Destructor may be invoked before the** `__global__` **function has finished**

   Kernel launches are asynchronous with host execution. As a result, if a `__global__` function argument has a non-trivial destructor, the destructor may execute in host code even before the `__global__` function has finished execution. This may break programs where the destructor has side effects.

   Example:

   ```
   #include <cassert>

   __managed__ int var = 0;

   struct MyStruct {
       __host__ __device__ ~MyStruct() { var = 3; }
   };

   __global__ void device_function(MyStruct my_struct) {
       assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
   }

   int main() {
       MyStruct my_struct;
       // GPU kernel execution is asynchronous with host execution.
       // As a result, MyStruct::~MyStruct() could be executed before
       // the kernel finishes executing.
       device_function<<<1, 1>>>(my_struct);
       cudaDeviceSynchronize();
   }
   ```

   See the example on [Compiler Explorer](https://godbolt.org/z/cn6Y5W6zs).

