### 5.3.9.6. Classes

#### 5.3.9.6.1. Class-type Variables

A variable definition with `__device__`, `__constant__`, `__managed__` or `__shared__` memory space cannot have a class type with a non-empty constructor or a non-empty destructor. A constructor for a class type is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

* The constructor function has been defined.
* The constructor function has no parameters, an empty initializer list, and an empty compound statement function body.
* Its class has no `virtual` functions, `virtual` base classes, or non-`static` data member initializers.
* The default constructors of all of its base classes can be considered empty.
* For all non-`static` data members of the class that are of a class type (or an array thereof), the default constructors can be considered empty.

A class's destructor is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

* The destructor function has been defined.
* The destructor function body is an empty compound statement.
* Its class has no `virtual` functions or `virtual` base classes.
* The destructors of all of its base classes can be considered empty.
* For all non-`static` data members of the class that are of a class type (or an array thereof), the destructor can be considered empty.

#### 5.3.9.6.2. Data Members

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on `class`, `struct`, and `union` data members.

Only `static` data members evaluated at compile time are supported, such as [const-qualified](#const-variables) and `constexpr` variables.

```
struct MyStruct {
   static inline constexpr int value1 = 10; // C++17
   static constexpr        int value2 = 10; // C++11
   static const            int value3 = 10;
// static                  int value4; // ERROR
};
```

#### 5.3.9.6.3. Function Members

`__global__` functions cannot be members of a `struct`, `class`, or `union`.

A `__global__` function is allowed in a `friend` declaration, but cannot be defined.

Example:

```
struct MyStruct {
    friend __global__ void f();   // CORRECT, friend declaration only

//  friend __global__ void g() {} // ERROR, friend definition
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/rv6cP3b9j).

#### 5.3.9.6.4. Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions

Implicitly-declared special member functions are those the compiler declares for a class when the user does not declare them; Explicitly-defaulted functions are ones the user declares but marks with `= default`. The special member functions that are implicitly-declared or explicitly-defaulted are default constructor, copy constructor, move constructor, copy assignment operator, move assignment operator, and destructor.

Let `F` denote a non-`virtual` function that is either implicitly declared or explicitly defaulted on its first declaration.
The execution space specifiers for `F` are the union of the execution space specifiers of all functions that invoke it. Note that for this analysis, a `__global__` caller will be treated as a `__device__` caller. For example:

```
class Base {
    int x;
public:
    __host__ __device__ Base() : x(10) {}
};

class Derived : public Base {
    int y;
};

class Other: public Base {
    int z;
};

__device__ void foo() {
    Derived D1;
    Other D2;
}

__host__ void bar() {
    Other D3;
}
```

In this case, the implicitly declared constructor function `Derived::Derived()` will be treated as a `__device__` function because it is only invoked from the `__device__` function `foo()`. The implicitly declared constructor function `Other::Other()` will be treated as a `__host__ __device__` function since it is invoked both from both a `__device__` function `foo()` and a `__host__` function `bar()`.

Additionally, if `F` is an implicitly-declared `virtual` function (for example, a `virtual` destructor), the execution spaces of each virtual function `D` that is overridden by `F` are added to the set of execution spaces for `F` if `D` not implicitly-declared.

For example:

```
struct Base1 {
    virtual __host__ __device__ ~Base1() {}
};

struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                            // ~Derived1() has __host__ __device__  execution space specifiers

struct Base2 {
    virtual __device__ ~Base2() = default;
};

struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                            // ~Derived2() has __device__ execution space specifiers
```

#### 5.3.9.6.5. Polymorphic Classes

Polymorphic classes, namely those with `virtual` functions, derived from other polymorphic classes, or with polymorphic data members, are subject to the following restrictions:

* Copying polymorphic objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.
* The execution space of an overridden `virtual` function must match the execution space of the function in the base class.

Example:

```
struct MyClass {
    virtual __host__ __device__ void f() {}
};

__global__ void kernel(MyClass my_class) {
    my_class.f(); // undefined behavior
}

int main() {
    MyClass my_class;
    kernel<<<1, 1>>>(my_class);
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/To39sGTrW).

---

```
struct BaseClass {
    virtual __host__ __device__ void f() {}
};

struct DerivedClass : BaseClass {
    __device__ void f() override {} // ERROR
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/xfKhEGfdG).

#### 5.3.9.6.6. Windows-Specific Class Layout

The CUDA compiler follows the IA64 ABI for class layout, while Microsoft Visual Studio does not. This prevents bitwise copy of special objects between host and device code as described below.

Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

* `T` is a [polymorphic class](#polymorphic-classes)
* `T` has multiple inheritance with more than one direct or indirect [empty base class](#class-type-variables).
* All direct and indirect base classes `B` are [empty](#class-type-variables) and the type of the first field `F` of `T` uses `B` in its definition, such that `B` is laid out at offset 0 in the definition of `F`.

Classes of type `T`, with a base class of type `T`, or with data members of type `T`, may have a different class layout and size between host and device when compiled with Microsoft Visual Studio.

Copying such objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.

### 5.3.9.7. Templates

A type cannot be used as template argument of a `__global__` function or a `__device__/__constant__` variable (C++14) if either:

* The type is defined within a `__host__` or `__host__ __device__` function scope.
* The type is unnamed, such as an anonymous struct or a lambda expression, unless the type is local to a `__device__` or `__global__` function.
* The type is a class member with `private` or `protected`, unless the class is local to a `__device__` or `__global__` function.
* The type is compounded from any of the types above.

Example:

```
template <typename T>
__global__ void kernel() {}

template <typename T>
__device__ int device_var; // C++14

struct {
    int v;
} unnamed_struct;

void host_function() {
    struct LocalStruct {};
//  kernel<LocalStruct><<<1, 1>>>(); // ERROR, LocalStruct is defined within a host function
    int data = 4;
//  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // ERROR, same as above

    auto lambda = [](){};
//  kernel<decltype(lambda)><<<1, 1>>>();         // ERROR, unnamed type
//  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // ERROR, unnamed type
}

class MyClass {
private:
    struct PrivateStruct {};
public:
    static void launch() {
//      kernel<PrivateStruct><<<1, 1>>>(); // ERROR, private type
    }
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/EhTn3GT3z).

## 5.3.10. C++11 Restrictions

### 5.3.10.1. `inline` Namespaces

It is not allowed to define one of the following entities within an `inline` namespace when another entity of the same name and type signature is defined in an enclosing namespace:

* `__global__` function.
* `__device__`, `__constant__`, `__managed__`, `__shared__` variables.
* Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.

Example:

```
__device__ int my_var; // global scope

inline namespace NS {

__device__ int my_var; // namespace scope

} // namespace NS
```

### 5.3.10.2. `inline` Unnamed Namespaces

The following entities cannot be declared in namespace scope within an `inline` unnamed namespace:

* `__global__` function.
* `__device__`, `__constant__`, `__managed__`, `__shared__` variables.
* Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.

### 5.3.10.3. `constexpr` Functions

By default, a `constexpr` function cannot be called from a function with incompatible execution space, in the same way as standard functions.

* Calling a device-only `constexpr` function from a host-function during host code generation phase, namely when `__CUDA_ARCH__` macro is undefined. Example:

  ```
  constexpr __device__ int device_function() { return 0; }

  int main() {
      int x = device_function();  // ERROR, calling a device-only constexpr function from host code
  }
  ```
* Calling a host-only `constexpr` function from a `__device__` or `__global__` function, during device code generation phase, namely when `__CUDA_ARCH__` macro is defined. Example:

  ```
  constexpr int host_function() { return 0; }

  __device__ void device_function() {
      int x = host_function();  // ERROR, calling a host-only constexpr function from device code
  }
  ```

Note that a function template specialization may not be a `constexpr` function even if the corresponding template function is marked with the keyword `constexpr`.

**Relaxed constexpr-Function Support**

The experimental `nvcc` flag `--expt-relaxed-constexpr` can be used to relax this constraint for both `__host__` and `__device__` functions. However, a `__global__` function cannot be declared as `constexpr`. `nvcc` will also define the macro `__CUDACC_RELAXED_CONSTEXPR__`.

When this flag is specified, the compiler will support cross execution space calls described above, as follows:

1. A call to a `constexpr` function in a cross-execution space is supported if it occurs in a context that requires constant evaluation, such as the initializer of a `constexpr` variable. Example:

   ```
   constexpr __host__ int host_function(int x) { return x + 1; };

   __global__ void kernel() {
       constexpr int val = host_function(1); // CORRECT, the call is in a context that requires constant evaluation.
   }

   constexpr __device__ int device_function(int x) { return x + 1; }

   int main() {
       constexpr int val = device_function(1); // CORRECT, the call is in a context that requires constant evaluation.
   }
   ```
2. Device code is generated during device code generation for the body of a host-only constexpr function, unless it is not used or is only called in a `constexpr` context. Example:

   ```
   // NOTE: "host_function" is emitted in generated device code because
   //       it is called from device code in a non-constexpr context
   constexpr int host_function(int x) { return x + 1; }

   __device__ int device_function(int in) {
       return host_function(in);  // CORRECT, even though argument is not a constant expression
   }
   ```
3. All code restrictions that apply to a device function also apply to the `constexpr` host-only function called from the device code. However, the compiler may not emit any build-time diagnostics for restrictions related to the compilation process.

   For example, the following code patterns are not supported in the body of the host function. This is similar to any device function; however, no compiler diagnostic may be generated.

   * One-Definition Rule (ODR)-use of a host variable or host-only non-`constexpr` function. Example:

     ```
     int host_var1, host_var2;

     constexpr int* host_function(bool b) { return b ? &host_var1 : &host_var2; };

     __device__ int device_function(bool flag) {
         return *host_function(flag); // ERROR, host_function() attempts to refer to the host variables
                                      //        'host_var1' and 'host_var2'.
                                      //        The code will compile, but will NOT execute correctly.
     }
     ```
   * Use of exceptions `throw/catch` and Run-Time Type Information `typeid/dynamic_cast`. Example:

     ```
     struct Base { };
     struct Derived : public Base { };

     // NOTE: "host_function" is emitted in generated device code
     constexpr int host_function(bool b, Base *ptr) {
         if (b) {
             return 1;
         }
         else if (typeid(ptr) == typeid(Derived)) { // ERROR, use of typeid in code executing on the GPU
             return 2;
         }
         else {
             throw int{4}; // ERROR, use of throw in code executing on the GPU
         }
     }

     __device__ void device_function(bool flag) {
         Derived d;
         int val = host_function(flag, &d); //ERROR, host_function() attempts use typeid and throw(),
                                            //       which are not allowed in code that executes on the GPU
     }
     ```
4. During host code generation, the body of a device-only `constexpr` function is preserved in the code sent to the host compiler. However, if the body of a device function attempts to ODR-use a namespace-scope device variable or a non-`constexpr` device function, the call to the device function from host code is not supported. While the code may build without compiler diagnostics, it may behave incorrectly at runtime. Example:

   ```
   __device__ int device_var1, device_var2;

   constexpr __device__ int* device_function(bool b) { return b ? &device_var1 : &device_var2; };

   int host_function(bool flag) {
       return *device_function(flag); // ERROR, device_function() attempts to refer to device variables
                                      //        'device_var1' and 'device_var2'
                                      // The code will compile, but will NOT execute correctly.
   }
   ```

Warning

Due to the above restrictions and the lack of compiler diagnostics for incorrect usage, it is recommended to avoid calling a function in the Standard C++ headers `std::` from device code. The implementation of such functions varies depending on the host platform. Instead, it is strongly suggested to call the equivalent functionality in the CUDA C++ Standard Library [libcu++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#libcu), in the `cuda::std::` namespace.

### 5.3.10.4. `constexpr` Variables

By default, a `constexpr` variable cannot be used in a function with incompatible execution space, in the same way of standard variables.

A `constexpr` variable can be directly used in device code in the following cases:

* C++ scalar types, excluding pointer and pointer-to-member types:

  + `nullptr_t`.
  + `bool`.
  + Integral types: `char`, `signed char`, `unsigned`, `long long`, etc.
  + Floating point types: `float`, `double`.
  + Enumerators: `enum` and `enum class`.
* Class types: `class`, `struct`, and `union` with a `constexpr` constructor.
* Raw array of the types above, for example `int[]`, only when they are used inside a `constexpr` `__device__` or `__host__ __device__` function.

`constexpr __managed__` and `constexpr __shared__` variables are not allowed.

Examples:

```
constexpr int ConstexprVar = 4; // scalar type

struct MyStruct {
    static constexpr int ConstexprVar = 100;
};

constexpr MyStruct my_struct = MyStruct{}; // class type

constexpr int array[] = {1, 2, 3};

__device__ constexpr int get_value(int idx) {
    return array[idx];                      // CORRECT
}

__device__ void foo(int idx) {
    int        v1 = ConstexprVar;           // CORRECT
    int        v2 = MyStruct::ConstexprVar; // CORRECT
//  const int &v3 = ConstexprVar1;          // ERROR, reference to host constexpr variable
//  const int *v4 = &ConstexprVar1;         // ERROR, address of host constexpr variable
    int        v5 = get_value(2);           // CORRECT, 'get_value(2)' is a constant expression.
//  int        v6 = get_value(idx);         // ERROR, 'get_value(idx)' is not a constant expression
//  int        v7 = array[2];               // ERROR, 'array' is not scalar type.
    MyStruct   v8 = my_struct;              // CORRECT
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/MWa1o3c9z).

### 5.3.10.5. `__global__` Variadic Template

A variadic `__global__` function template has the following restrictions:

* Only a single pack parameter is allowed.
* The pack parameter must be listed last in the template parameter list.

Examples:

```
template <typename... Pack>
__global__ void kernel1(); // CORRECT

// template <typename... Pack, template T>
// __global__ void kernel2(); // ERROR, parameter pack is not the last parameter

template <typename... TArgs>
struct MyStruct {};

// template <typename... Pack1, typename... Pack2>
// __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // ERROR, more than one parameter pack
```

See the example on [Compiler Explorer](https://godbolt.org/z/x48KnPbbY).

### 5.3.10.6. Defaulted Functions `= default`

The CUDA compiler infers the execution space of explicitly-defaulted member functions as described in [Implicitly-declared and explicitly-defaulted functions](#compiler-generated-functions).

Execution space specifiers on explicitly-defaulted functions are ignored by the compiler, except in the case the function is defined out-of-line or is a `virtual` function.

Examples:

```
struct MyStruct1 {
    MyStruct1() = default;
};

void host_function() {
    MyStruct1 my_struct; // __host__ __device__ constructor
}

__device__ void device_function() {
    MyStruct1 my_struct; // __host__ __device__ constructor
}

struct MyStruct2 {
    __device__ MyStruct2() = default; // WARNING: __device__ annotation is ignored
};

struct MyStruct3 {
    __host__ MyStruct3();
};
MyStruct3::MyStruct3() = default; // out-of-line definition, not ignored

__device__ void device_function2() {
//  MyStruct3 my_struct; // ERROR, __host__ constructor
}

struct MyStruct4 {
    //  MyStruct4::~MyStruct4 has host execution space, not ignored because virtual
    virtual __host__ ~MyStruct4() = default;
};

__device__ void device_function3() {
    MyStruct4 my_struct4;
    // implicit destructor call for 'my_struct4':
    //    ERROR: call from a __device__ function 'device_function3' to a
    //    __host__ function 'MyStruct4::~MyStruct4'
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/q1M4j8YYf).

### 5.3.10.7. `[cuda::]std::initializer_list`

By default, the CUDA compiler implicitly considers the member functions of `[cuda::]std::initializer_list` to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code.

The `nvcc` flag `--no-host-device-initializer-list` disables this behavior; member functions of `[cuda::]std::initializer_list` will then be considered as `__host__` functions and will not be directly invocable from device code.

A `__global__` function cannot have a parameter of type `[cuda::]std::initializer_list`.

Example:

```
#include <initializer_list>

__device__ void foo(std::initializer_list<int> in) {}

__device__ void bar() {
    foo({4,5,6}); // (a) initializer list containing only constant expressions.
    int i = 4;
    foo({i,5,6}); // (b) initializer list with at least one  non-constant element.
                  // This form may have better performance than (a).
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/xeah7r44T).

### 5.3.10.8. `[cuda::]std::move`, `[cuda::]std::forward`

By default, the CUDA compiler implicitly considers `std::move` and `std::forward` function templates to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The `nvcc` flag `--no-host-device-move-forward` disables this behavior; `std::move` and `std::forward` will then be considered as `__host__` functions and will not be directly invocable from device code.

Hint

`cuda::std::move` and `cuda::std::forward` on the contrary always have `__host__ __device__` execution space.

