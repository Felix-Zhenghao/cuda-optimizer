## 5.3.11. C++14 Restrictions

### 5.3.11.1. Functions with Deduced Return Type

A `__global__` function cannot have a deduced return type `auto`.

Introspection of the return type of a `__device__` function with a deduced return type is not allowed in host code.

Note

The CUDA frontend compiler changes the function declaration to have a `void` return type, before invoking the host compiler. This may break introspection of the deduced return type of the `__device__` function in host code. Thus, the CUDA compiler will issue a compile-time error for referencing such a deduced return type outside of device function bodies.

Examples:

```
 __device__ auto device_function(int x) { // deduced return type
     return x;                            // decltype(auto) has the same behavior
 }

 __global__ void kernel() {
     int x = sizeof(device_function(2));         // CORRECT, device code scope
 }

 // const int size = sizeof(device_function(2)); // ERROR, return type deduction on host

 void host_function() {
 //  using T = decltype(device_function(2));     // ERROR, return type deduction on host
 }

void host_fn1() {
  // ERROR, referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // ERROR, referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// ERROR, referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct MyStruct { };

// ERROR, referenced outside device function bodies
struct S1_derived_t : MyStruct<decltype(fn1)> { };
```

### 5.3.11.2. Variable Templates

A `__device__` or `__constant__` variable template cannot be `const`-qualified when using the Microsoft compiler.

Examples:

```
// ERROR on Windows (non-portable), const-qualified
template <typename T>
__device__ const T var = 0;

 // CORRECT, ptr1 is not const-qualified
template <typename T>
__device__ const T* ptr1 = nullptr;

// ERROR on Windows (non-portable), ptr2 is const-qualified
template <typename T>
__device__ const T* const ptr2 = nullptr;
```

See the example on [Compiler Explorer](https://godbolt.org/z/8hM5Yh7db).

## 5.3.12. C++17 Restrictions

### 5.3.12.1. `inline` Variables

In a single translation unit, using an `inline` variable provides no additional functionality beyond a regular variable and does not provide any practical advantage.

`nvcc` allows `inline` variables with `__device__`, `__constant__`, or `__managed__` memory space only in [Separate Compilation](../02-basics/nvcc.html#nvcc-separate-compilation) mode or for variables with internal linkage.

Note

When using `gcc/g++` host compiler, an `inline` variable declared with `__managed__` memory space specifier may not be visible to the debugger.

Examples:

```
inline        __device__ int device_var1;  // CORRECT, when compiled in Separate Compilation mode (-rdc=true or -dc)
                                           // ERROR, when compiled in Whole Program Compilation mode

static inline __device__ int device_var2;  // CORRECT, internal linkage

namespace {

inline __device__ int device_var3;         // CORRECT, internal linkage

inline __shared__ int shared_var;          // CORRECT, internal linkage

static inline __device__ int device_var4;  // CORRECT, internal linkage

inline __device__ int device_var5;         // CORRECT, internal linkage

} // namespace
```

See the example on [Compiler Explorer](https://godbolt.org/z/oraqeGTzY).

### 5.3.12.2. Structured Binding

A structured binding cannot be declared with a memory space specifier, such as `__device__`, `__shared__`, `__constant__`, or `__managed__`.

Example:

```
struct S {
    int x, y;
};
// __device__ auto [a, b] = S{4, 5}; // ERROR
```

## 5.3.13. C++20 Restrictions

### 5.3.13.1. Three-way Comparison Operator

The three-way comparison operator (`<=>`) is supported in device code, but some uses implicitly rely on functionality from the C++ Standard Library, which is provided by the host implementation. Using those operators may require specifying the flag `--expt-relaxed-constexpr` to silence warnings, and the functionality requires the host implementation to satisfy the requirements of the device code.

Examples:

```
#include <compare> // std::strong_ordering implementation

struct S {
    int x, y;

    auto operator<=>(const S&) const = default; // (a)

    __host__ __device__ bool operator<=>(int rhs) const { return false; } // (b)
};

__host__ __device__ bool host_device_function(S a, S b) {
    if (a <=> 1)  // CORRECT, calls a user-defined host-device overload (b)
        return true;
    return a < b; // CORRECT, call to an implicitly-declared function (a)
                  // Note: it requires a device-compatible std::strong_ordering
                  //       implementation provided in the header <compare>
                  //       and the flag --expt-relaxed-constexpr
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/qzs5arfx4).

### 5.3.13.2. `consteval` Functions

`consteval` functions can be called from both host and device code, independently of their execution space.

Examples:

```
consteval int host_consteval() {
    return 10;
}

__device__ consteval int device_consteval() {
    return 10;
}

__device__ int device_function() {
    return host_consteval();   // CORRECT, even if called from device code
}

__host__ __device__ int host_device_function() {
    return device_function();  // CORRECT, even if called from host-device code
}
```
