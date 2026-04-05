## 5.4.5. Atomic Functions

Atomic functions perform read-modify-write operations on shared data, making them appear to execute in a single step. Atomicity ensures that each operation either completes fully or not at all, providing all participating threads with a consistent view of the data.

CUDA provides atomic functions in four ways:

Extended CUDA C++ atomic functions, [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::atomic\_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html).
:   * They are allowed in both host and device code.
    * They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.
    * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.

Standard C++ atomic functions, [cuda::std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [cuda::std::atomic\_ref](https://en.cppreference.com/w/cpp/atomic/atomic_ref.html).
:   * They are allowed in both host and device code.
    * They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.
    * They do not allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.

Compiler [built-in atomic functions](#built-in-atomic-functions), `__nv_atomic_<op>()`.
:   * They have been available since CUDA 12.8.
    * They are only allowed in device code.
    * They follow the [C++ standard atomic memory order](https://en.cppreference.com/w/cpp/atomic/memory_order.html) semantics.
    * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.
    * They have the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html).
    * They support a subset of the data types allowed by [cuda::std::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::std::atomic\_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), except for 128-bit data types.

[Legacy atomic functions](#legacy-atomic-functions), `atomic<Op>()`.
:   * They are only allowed in device code.
    * They only support `memory_order_relaxed` [C++ atomic memory semantics](https://en.cppreference.com/w/cpp/atomic/memory_order.html).
    * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations as part of the function name.
    * Unlike [built-in atomic functions](#built-in-atomic-functions), legacy atomic functions only ensure atomicity and do not introduce synchronization points (fences).
    * They support a subset of the data types allowed by [built-in atomic functions](#built-in-atomic-functions). The atomic `add` operation supports additional data types.

Hint

Using the [Extended CUDA C++ atomic functions](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) provided by `libcu++` is recommended for efficiency, safety, and portability.

### 5.4.5.1. Legacy Atomic Functions

Legacy atomic functions perform atomic read-modify-write operations on a 32-, 64-, or 128-bit word stored in global or shared memory. For example, the `atomicAdd()` function reads a word at a specific address in global or shared memory, adds a number to it, and writes the result back to the same address.

* Atomic functions can only be used in device functions.
* For vector types such as `__half2`, `__nv_bfloat162`, `float2`, and `float4`, the read-modify-write operation is performed on each element of the vector. The entire vector is not guaranteed to be atomic in a single access.

The atomic functions described in this section have a [memory ordering](https://en.cppreference.com/w/cpp/atomic/memory_order) of `cuda::std::memory_order_relaxed` and are only atomic at a particular [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes):

* Atomic APIs without a suffix, for example `atomicAdd`, are atomic at scope `cuda::thread_scope_device`.
* Atomic APIs with the `_block` suffix, for example, `atomicAdd_block`, are atomic at scope `cuda::thread_scope_block`.
* Atomic APIs with the `_system` suffix, for example, `atomicAdd_system`, are atomic at scope `cuda::thread_scope_system` if they meet particular [conditions](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity).

The following example shows the CPU and GPU atomically updating an integer value at address `addr`:

```
#include <cuda_runtime.h>

__global__ void atomicAdd_kernel(int* addr) {
    atomicAdd_system(addr, 10);
}

void test_atomicAdd(int device_id) {
    int* addr;
    cudaMallocManaged(&addr, 4);
    *addr = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    if (deviceProp.concurrentManagedAccess != 1) {
        return; // the device does not coherently access managed memory concurrently with the CPU
    }

    atomicAdd_kernel<<<...>>>(addr);
    __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}
```

---

Note that any atomic operation can be implemented based on `atomicCAS()` (Compare and Swap). For example, `atomicAdd()` for single-precision floating-point numbers can be implemented as follows:

```
#include <cuda/memory>
#include <cuda/std/bit>

__device__ float customAtomicAdd(float* d_ptr, float value) {
    volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
    unsigned  old_value      = *d_ptr_unsigned;
    unsigned  assumed;
    do {
        assumed                          = old_value;
        float    assumed_float           = cuda::std::bit_cast<float>(assumed);
        float    expected_value          = assumed_float + value;
        unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
        old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old_value);
    return cuda::std::bit_cast<float>(old_value);
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/676e5bc7a).

#### 5.4.5.1.1. `atomicAdd()`

```
T atomicAdd(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old + val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicAdd()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`, `float`, `double`, `__half2`, `__half`.
* `__nv_bfloat16`, `__nv_bfloat162` on devices of compute capability 8.x and higher.
* `float2`, `float4` on devices of compute capability 9.x and higher, and only supported for global memory addresses.

The atomicity of `atomicAdd()` applied to vector types, for example `__half2` or `float4`, is guaranteed separately for each of the components; the entire vector is not guaranteed to be atomic as a single access.

#### 5.4.5.1.2. `atomicSub()`

```
T atomicSub(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old - val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicSub()` supports the following data types:

* `int`, `unsigned`

#### 5.4.5.1.3. `atomicInc()`

```
unsigned atomicInc(unsigned* address, unsigned val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old >= val ? 0 : (old + 1)`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

#### 5.4.5.1.4. `atomicDec()`

```
unsigned atomicDec(unsigned* address, unsigned val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `(old == 0 || old > val) ? val : (old - 1)`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

#### 5.4.5.1.5. `atomicAnd()`

```
T atomicAnd(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old & val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicAnd()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`.

#### 5.4.5.1.6. `atomicOr()`

```
T atomicOr(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old | val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicOr()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`.

#### 5.4.5.1.7. `atomicXor()`

```
T atomicXor(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old ^ val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicXor()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`.

#### 5.4.5.1.8. `atomicMin()`

```
T atomicMin(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the minimum of `old` and `val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicMin()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`, `long long`.

#### 5.4.5.1.9. `atomicMax()`

```
T atomicMax(T* address, T val);
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the maximum of `old` and `val`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicMax()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`, `long long`.

#### 5.4.5.1.10. `atomicExch()`

```
T atomicExch(T* address, T val);
```

```
template<typename T>
T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Stores `val` back to memory at the same address.

The function returns the `old` value.

`atomicExch()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`, `float`.

The C++ template function `atomicExch()` supports 128-bit types with the following requirements:

* Compute capability 9.x and higher.
* `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.
* `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.
* For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.

#### 5.4.5.1.11. `atomicCAS()`

```
T atomicCAS(T* address, T compare, T val);
```

```
template<typename T>
T atomicCAS(T* address, T compare, T val);  // only 128-bit types, compute capability 9.x and higher
```

The function performs the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old == compare ? val : old`.
3. Stores the result back to memory at the same address.

The function returns the `old` value.

`atomicCAS()` supports the following data types:

* `int`, `unsigned`, `unsigned long long`, `unsigned short`.

The C++ template function `atomicCAS()` supports 128-bit types with the following requirements:

* Compute capability 9.x and higher.
* `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.
* `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.
* For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.

### 5.4.5.2. Built-in Atomic Functions

CUDA 12.8 and later support CUDA compiler built-in functions for atomic operations, following the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) and the CUDA [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes). The functions follow the [GNU's atomic built-in function signature](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) with an extra argument for thread scope.

`nvcc` defines the macro `__CUDACC_DEVICE_ATOMIC_BUILTINS__` when built-in atomic functions are supported.

Below are listed the raw enumerators for the [memory orders](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes), which are used as the `order` and `scope` arguments of the built-in atomic functions:

```
// atomic memory orders
enum {
   __NV_ATOMIC_RELAXED,
   __NV_ATOMIC_CONSUME,
   __NV_ATOMIC_ACQUIRE,
   __NV_ATOMIC_RELEASE,
   __NV_ATOMIC_ACQ_REL,
   __NV_ATOMIC_SEQ_CST
};
```

```
// thread scopes
enum {
   __NV_THREAD_SCOPE_THREAD,
   __NV_THREAD_SCOPE_BLOCK,
   __NV_THREAD_SCOPE_CLUSTER,
   __NV_THREAD_SCOPE_DEVICE,
   __NV_THREAD_SCOPE_SYSTEM
};
```

* The memory order corresponds to [C++ standard atomic operations' memory order](https://en.cppreference.com/w/cpp/atomic/memory_order).
* The thread scope follows the `cuda::thread_scope` [definition](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).
* `__NV_ATOMIC_CONSUME` memory order is currently implemented using stronger `__NV_ATOMIC_ACQUIRE` memory order.
* `__NV_THREAD_SCOPE_THREAD` thread scope is currently implemented using wider `__NV_THREAD_SCOPE_BLOCK` thread scope.

Example:

```
__device__ T __nv_atomic_load_n(T*  pointer,
                                int memory_order,
                                int thread_scope = __NV_THREAD_SCOPE_SYSTEM);
```

Atomic built-in functions have the following restrictions:

* They can only be used in device functions.
* They cannot operate on local memory.
* The addresses of these functions cannot be taken.
* The `order` and `scope` arguments must be integer literals; they cannot be variables.
* The thread scope `__NV_THREAD_SCOPE_CLUSTER` is supported on architectures `sm_90` and higher.

Example of unsupported cases:

```
 // Not permitted in a host function
 __host__ void bar() {
     unsigned u1 = 1, u2 = 2;
     __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
 }

 // Not permitted to be applied to local memory
__device__ void foo() {
   unsigned a = 1, b;
   __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

 // Not permitted as a template default argument.
 // The function address cannot be taken.
 template<void *F = __nv_atomic_load_n>
 class X {
     void *f = F; // The function address cannot be taken.
 };

 // Not permitted to be called in a constructor initialization list.
 class Y {
     int a;
 public:
     __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
 };
```

#### 5.4.5.2.1. `__nv_atomic_fetch_add()`, `__nv_atomic_add()`

```
__device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old + val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_add` returns the `old` value.
* `__nv_atomic_add` has no return value.

The functions support the following data types:

* `int`, `unsigned`, `unsigned long long`, `float`, `double`.

#### 5.4.5.2.2. `__nv_atomic_fetch_sub()`, `__nv_atomic_sub()`

```
__device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old - val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_sub` returns the `old` value.
* `__nv_atomic_sub` has no return value.

The functions support the following data types:

* `int`, `unsigned`, `unsigned long long`, `float`, `double`.

#### 5.4.5.2.3. `__nv_atomic_fetch_and()`, `__nv_atomic_and()`

```
__device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old & val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_and` returns the `old` value.
* `__nv_atomic_and` has no return value.

The functions support the following data types:

* Any integral type of size 4 or 8 bytes.

#### 5.4.5.2.4. `__nv_atomic_fetch_or()`, `__nv_atomic_or()`

```
__device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old | val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_or` returns the `old` value.
* `__nv_atomic_or` has no return value.

The functions support the following data types:

* Any integral type of size 4 or 8 bytes.

#### 5.4.5.2.5. `__nv_atomic_fetch_xor()`, `__nv_atomic_xor()`

```
__device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes `old ^ val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_xor` returns the `old` value.
* `__nv_atomic_xor` has no return value.

The functions support the following data types:

* Any integral type of size 4 or 8 bytes.

#### 5.4.5.2.6. `__nv_atomic_fetch_min()`, `__nv_atomic_min()`

```
__device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the minimum of `old` and `val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_min` returns the `old` value.
* `__nv_atomic_min` has no return value.

The functions support the following data types:

* `unsigned`, `int`, `unsigned long long`, `long long`.

#### 5.4.5.2.7. `__nv_atomic_fetch_max()`, `__nv_atomic_max()`

```
__device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Computes the maximum of `old` and `val`.
3. Stores the result back to memory at the same address.

* `__nv_atomic_fetch_max` returns the `old` value.
* `__nv_atomic_max` has no return value.

The functions support the following data types:

* `unsigned`, `int`, `unsigned long long`, `long long`

#### 5.4.5.2.8. `__nv_atomic_exchange()`, `__nv_atomic_exchange_n()`

```
__device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. `__nv_atomic_exchange_n` stores `val` to where `address` points to.

   `__nv_atomic_exchange` stores `old` to where `ret` points to and stores the value located at the address `val` to where `address` points to.

* `__nv_atomic_exchange_n` returns the `old` value.
* `__nv_atomic_exchange` has no return value.

The functions support the following data types:

* Any data type of size of 4, 8 or 16 bytes.
* The 16-byte data type is supported on devices of compute capability 9.x and higher.

#### 5.4.5.2.9. `__nv_atomic_compare_exchange()`, `__nv_atomic_compare_exchange_n()`

```
__device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);

__device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. Compare `old` with the value where `expected` points to.
3. If they are equal, the return value is `true` and `desired` is stored to where `address` points to. Otherwise, it returns `false` and `old` is stored to where `expected` points to.

The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

The functions support the following data types:

* Any data type of size of 2, 4, 8 or 16 bytes.
* The 16-byte data type is supported on devices with compute capability 9.x and higher.

#### 5.4.5.2.10. `__nv_atomic_load()`, `__nv_atomic_load_n()`

```
__device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. `__nv_atomic_load` stores `old` to where `ret` points to.

   `__nv_atomic_load_n` returns `old`.

The functions support the following data types:

* Any data type of size 1, 2, 4, 8 or 16 bytes.

`order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.

#### 5.4.5.2.11. `__nv_atomic_store()`, `__nv_atomic_store_n()`

```
__device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

The functions perform the following operations in one atomic transaction:

1. Reads the `old` value located at the address `address` in global or shared memory.
2. `__nv_atomic_store` reads the value where `val` points to and stores to where `address` points to.

   `__nv_atomic_store_n` stores `val` to where `address` points to.

`order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.

#### 5.4.5.2.12. `__nv_atomic_thread_fence()`

```
__device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. The thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.

## 5.4.6. Warp Functions
