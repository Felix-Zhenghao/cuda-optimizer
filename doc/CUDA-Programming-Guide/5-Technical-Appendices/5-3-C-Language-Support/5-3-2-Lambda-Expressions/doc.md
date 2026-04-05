## 5.3.7. Lambda Expressions

The compiler determines the execution space of a lambda expression or closure type (C++11) by associating it with the execution space of the innermost enclosing function scope. If there is no enclosing function scope, the execution space is specified as `__host__`.

The execution space can also be specified explicitly with the [extended lambda syntax](#extended-lambdas).

Examples:

```
auto global_lambda = [](){ return 0; }; // __host__

void host_function() {
    auto lambda1 = [](){ return 1; };   // __host__
    [](){ return 3; };                  // __host__, closure type (body of a lambda expression)
}

__device__ void device_function() {
    auto lambda2 = [](){ return 2; };   // __device__
}

__global__ void kernel_function(void) {
    auto lambda3 = [](){ return 3; };   // __device__
}

__host__ __device__ void host_device_function() {
    auto lambda4 = [](){ return 4; };   // __host__ __device__
}

using function_ptr_t = int (*)();

__device__ void device_function(float          value,
                                function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}
```

See the example on [Compiler Explorer](https://godbolt.org/z/scv4vcczr).

### 5.3.7.1. Lambda Expressions and `__global__` Function Parameters

A lambda expression or a closure type can only be used as an argument to a `__global__` function if its execution space is `__device__` or `__host__ __device__`. Global or namespace scope lambda expressions cannot be used as arguments in a `__global__` function.

Examples:

```
template <typename T>
 __global__ void kernel(T input) {}

 __device__ void device_function() {
     // device kernel call requires separate compilation (-rdc=true flag)
     kernel<<<1, 1>>>([](){});
     kernel<<<1, 1>>>([] __device__() {});          // extended lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // extended lambda
 }

 auto global_lambda = [] __host__ __device__() {};

 void host_function() {
     kernel<<<1, 1>>>([] __device__() {});          // CORRECT, extended lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // CORRECT, extended lambda
 //  kernel<<<1, 1>>>([](){});                      // ERROR, closure type with host execution space
 //  kernel<<<1, 1>>>(global_lambda);               // ERROR, extended lambda, but at global scope
 }
```

See the example on [Compiler Explorer](https://godbolt.org/z/ajrsn5z5Y).

### 5.3.7.2. Extended Lambdas

The `nvcc` flag `--extended-lambda` allows explicit annotations of execution spaces in a lambda expression. These annotations should appear after the lambda introducer and before the optional lambda declarator.

`nvcc` defines the macro `__CUDACC_EXTENDED_LAMBDA__` when the `--extended-lambda` flag is specified.

* An *extended lambda* is defined within the scope of an immediate or nested block of a `__host__` or `__host__ __device__` function.
* An *extended device lambda* is a lambda expression annotated with the `__device__` keyword.
* An *extended host-device lambda* is a lambda expression annotated with the `__host__ __device__` keywords.

Unlike standard lambda expressions, extended lambdas can be used as type arguments in `__global__` functions.

Example:

```
void host_function() {
    auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
    auto lambda2 = [] __device__ {};           // extended lambda
    auto lambda3 = [] __host__ __device__ {};  // extended lambda
    auto lambda4 = [] __host__ {};             // NOT an extended lambda
}

__host__ __device__ void host_device_function() {
    auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
    auto lambda2 = [] __device__ {};           // extended lambda
    auto lambda3 = [] __host__ __device__ {};  // extended lambda
    auto lambda4 = [] __host__ {};             // NOT an extended lambda
}

__device__ void device_function() {
    // none of the lambdas within this function are extended lambdas,
    // because the enclosing function is not a __host__ or __host__ __device__  function.
    auto lambda1 = [] {};
    auto lambda2 = [] __device__ {};
    auto lambda3 = [] __host__ __device__ {};
    auto lambda4 = [] __host__ {};
}

auto global_lambda = [] __host__ __device__ { }; // NOT an extended lambda because it is not defined
                                                 // within a __host__ or __host__ __device__ function
```

### 5.3.7.3. Extended Lambda Type Traits

The compiler provides type traits to detect closure types for extended lambdas at compile time.

```
bool __nv_is_extended_device_lambda_closure_type(type);
```

The function returns `true` if `type` is the closure class created for an extended `__device__` lambda, `false` otherwise.

```
bool __nv_is_extended_device_lambda_with_preserved_return_type(type);
```

The function returns `true` if `type` is the closure class created for an extended `__device__` lambda and the lambda is defined with trailing return type, `false` otherwise. If the trailing return type definition refers to any lambda parameter name, the return type is not preserved.

```
bool __nv_is_extended_host_device_lambda_closure_type(type);
```

The function returns `true` if `type` is the closure class created for an extended `__host__ __device__` lambda, `false` otherwise.

---

The lambda type traits can be used in all compilation modes, regardless of whether lambdas or extended lambdas are enabled. The traits will always return `false` if extended lambda mode is inactive.

Example:

```
auto lambda0 = [] __host__ __device__ { };

void host_function() {
    auto lambda1 = [] { };
    auto lambda2 = [] __device__ { };
    auto lambda3 = [] __host__ __device__ { };
    auto lambda4 = [] __device__ () -> double { return 3.14; }
    auto lambda5 = [] __device__ (int x) -> decltype(&x) { return 0; }

    using lambda0_t = decltype(lambda0);
    using lambda1_t = decltype(lambda1);
    using lambda2_t = decltype(lambda2);
    using lambda3_t = decltype(lambda3);
    using lambda4_t = decltype(lambda4);
    using lambda5_t = decltype(lambda5);

    // 'lambda0' is not an extended lambda because it is defined outside function scope
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda0_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda0_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda0_t));

    // 'lambda1' is not an extended lambda because it has no execution space annotations
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda1_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda1_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda1_t));

    // 'lambda2' is an extended device-only lambda
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda2_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda2_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda2_t));

    // 'lambda3' is an extended host-device lambda
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda3_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda3_t));
    static_assert(__nv_is_extended_host_device_lambda_closure_type(lambda3_t));

    // 'lambda4' is an extended device-only lambda with preserved return type
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda4_t));
    static_assert(__nv_is_extended_device_lambda_with_preserved_return_type(lambda4_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda4_t));

    // 'lambda5' is not an extended device-only lambda with preserved return type
    // because it references the operator()'s parameter types in the trailing return type.
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda5_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda5_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda5_t));
}
```

### 5.3.7.4. Extended Lambda Restrictions

Before invoking the host compiler, the CUDA compiler replaces an extended lambda expression with an instance of a placeholder type defined in namespace scope. The placeholder type's template argument requires taking the address of a function that encloses the original extended lambda expression. This is necessary for correctly executing any `__global__` function template whose template argument involves the closure type of an extended lambda. The enclosing function is computed as follows.

By definition, an extended lambda is present within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

* If the function is not the `operator()` of a lambda expression, it is considered the enclosing function for the extended lambda.
* Otherwise, the extended lambda is defined within the immediate or nested block scope of the `operator()` of one or more enclosing lambda expressions.

  + If the outermost lambda expression is defined within the immediate or nested block scope of a function `F`, then `F` is the computed enclosing function.
  + Otherwise, the enclosing function does not exist.

Example:

```
void host_function() {
    auto lambda1 = [] __device__ { }; // enclosing function for lambda1 is "host_function()"
    auto lambda2 = [] {
        auto lambda3 = [] {
            auto lambda4 = [] __host__ __device__ { }; // enclosing function for lambda4 is "host_function"
        };
    };
}

auto global_lambda = [] {
    auto lambda5 = [] __host__ __device__ { }; // enclosing function for lambda5 does not exist
};
```

---

Extended Lambda Restrictions

1. An extended lambda cannot be defined inside another extended lambda expression. Example:

   ```
   void host_function() {
       auto lambda1 = [] __host__ __device__  {
            // ERROR, extended lambda defined within another extended lambda
           auto lambda2 = [] __host__ __device__ { };
       };
   }
   ```
2. An extended lambda cannot be defined inside a generic lambda expression. Example:

   ```
   void host_function() {
       auto lambda1 = [] (auto) {
            // ERROR, extended lambda defined within a generic lambda
           auto lambda2 = [] __host__ __device__ { };
       };
   }
   ```
3. If an extended lambda is defined within the immediate or nested block scope of one or more nested lambda expressions, then the outermost lambda expression must be defined within the immediate or nested block scope of a function. Example:

   ```
   auto lambda1 = []  {
       // ERROR, outer enclosing lambda is not defined within a non-lambda-operator() function
       auto lambda2 = [] __host__ __device__ { };
   };
   ```
4. The enclosing function of the extended lambda must be named, and its address must be accessible. If the enclosing function is a class member, the following conditions must be met:

   * All classes enclosing the member function must have a name.
   * The member function must not have private or protected access within its parent class.
   * All enclosing classes must not have private or protected access within their respective parent classes.

   Example:

   ```
   void host_function() {
       auto lambda1 = [] __device__ { return 0; }; // OK
       {
           auto lambda2 = [] __device__          { return 0; }; // OK
           auto lambda3 = [] __device__ __host__ { return 0; }; // OK
       }
   }

   struct MyStruct1 {
       MyStruct1() {
           auto lambda4 = [] __device__ { return 0; }; // ERROR, address of the enclosing function is not accessible
       }
   };

   class MyStruct2 {
       void foo() {
           auto temp1 = [] __device__ { return 10; }; // ERROR, enclosing function has private access in parent class
       }

       struct MyStruct3 {
           void foo() {
               auto temp1 = [] __device__ { return 10; };  // ERROR, enclosing class MyStruct3 has private access in its parent class
           }
       };
   };
   ```
5. At the point where the extended lambda has been defined, it must be possible to unambiguously take the address of the enclosing routine. However, this may not always be feasible, for example, when an alias declaration shadows a template type argument with the same name. Example:

   ```
   template <typename T>
   struct A {
       using Bar = void;
       void test();
   };

   template<>
   struct A<void> { };

   template <typename Bar>
   void A<Bar>::test() {
       // In code sent to host compiler, nvcc will inject an address expression here, of the form:
       //   (void (A< Bar> ::*)(void))(&A::test))
       //  However, the class typedef 'Bar' (to void) shadows the template argument 'Bar',
       //  causing the address expression in A<int>::test to actually refer to:
       //    (void (A< void> ::*)(void))(&A::test))
       //  which doesn't take the address of the enclosing routine 'A<int>::test' correctly.
       auto lambda1 = [] __host__ __device__ { return 4; };
   }

   int main() {
       A<int> var;
       var.test();
   }
   ```
6. An extended lambda cannot be defined in a class that is local to a function. Example:

   ```
   void host_function() {
       struct MyStruct {
           void bar() {
               // ERROR, bar() is member of a class that is local to a function
               auto lambda2 = [] __host__ __device__ { return 0; };
           }
       };
   }
   ```
7. The enclosing function for an extended lambda cannot have deduced return type. Example:

   ```
   auto host_function() {
       // ERROR, the return type of host_function() is deduced
       auto lambda3 = [] __host__ __device__ { return 0; };
   }
   ```
8. A host-device extended lambda cannot be a generic lambda, namely a lambda with an `auto` parameter type. Example:

   ```
   void host_function() {
       // ERROR, __host__ __device__ extended lambdas cannot be a generic lambda
       auto lambda1 = [] __host__ __device__ (auto i) { return i; };

       // ERROR, a host-device extended lambda cannot be a generic lambda
       auto lambda2 = [] __host__ __device__ (auto... i) {
           return sizeof...(i);
       };
   }
   ```
9. If the enclosing function is an instantiation of a function or member template, or if the function is a member of a class template, then the template(s) must satisfy the following constraints:

   * The template must have at most one variadic parameter, and it must be listed last in the template parameter list.
   * The template parameters must be named.
   * The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are `private` or `protected` class members.

   Example 1:

   ```
   template <template <typename...> class T,
             typename... P1,
             typename... P2>
   void bar1(const T<P1...>, const T<P2...>) {
       // ERROR, enclosing function has multiple parameter packs
       auto lambda = [] __device__ { return 10; };
   }

   template <template <typename...> class T,
             typename... P1,
             typename    T2>
   void bar2(const T<P1...>, T2) {
       // ERROR, for enclosing function, the parameter pack is not last in the template parameter list
       auto lambda = [] __device__ { return 10; };
   }

   template <typename T, T>
   void bar3() {
       // ERROR, for enclosing function, the second template parameter is not named
       auto lambda = [] __device__ { return 10; };
   }
   ```

   Example 2:

   ```
   template <typename T>
   void bar4() {
       auto lambda1 = [] __device__ { return 10; };
   }

   class MyStruct {
       struct MyNestedStruct {};

       friend int main();
   };

   int main() {
       struct MyLocalStruct {};
       // ERROR, enclosing function for device lambda in bar4() is instantiated with a type local to main
       bar4<MyLocalStruct>();

       // ERROR, enclosing function for device lambda in bar4 is instantiated with a type
       //        that is a private member of a class
       bar4<MyStruct::MyNestedStruct>();
   }
   ```
10. With Microsoft Visual Studio host compilers, the enclosing function must have external linkage. This restriction exists because the host compiler does not support using the addresses of non-extern linkage functions as template arguments. The CUDA compiler transformations require these addresses to support extended lambdas.
11. With Microsoft Visual Studio host compilers, an extended lambda shall not be defined within the body of an `if constexpr` block.
12. An extended lambda has the following restrictions on captured variables:

    * The variable may be passed by value to a sequence of helper functions in the code sent to the host compiler before being used to directly initialize the field of the class type representing the closure type for the extended lambda. However, the C++ standard specifies that the captured variable should be used for direct initialization of the closure type's field.
    * A variable can only be captured by value.
    * A variable of array type cannot be captured if the number of array dimensions is greater than 7.
    * For an array-type variable, the array field of the closure type is first default-initialized and then each array element is copy-assigned from the corresponding element of the captured array variable in the code sent to the host compiler. Therefore, the array element type must be both default-constructible and copy-assignable in the host code.
    * A function parameter that is an element of a variadic argument pack cannot be captured.
    * The captured variable type cannot be local to a function, except for extended lambda closure types, or `private` or `protected` class members.
    * Init-capture is not supported for host-device extended lambdas. However, it is supported for device extended lambdas, except when the initializer is an array or of type `std::initializer_list`.
    * The function call operator for an extended lambda is not a `constexpr`. The closure type of an extended lambda is not a literal type. The `constexpr` and `consteval` specifiers cannot be used when declaring an extended lambda.
    * A variable cannot be implicitly captured inside an `if-constexpr` block that is lexically nested inside an extended lambda unless the variable has been implicitly captured outside the `if-constexpr` block or appears in the extended lambda's explicit capture list.

    Examples:

    ```
    void host_function() {
        // CORRECT, an init-capture is allowed for an extended device-only lambda
        auto lambda1 = [x = 1] __device__ () { return x; };

        // ERROR, an init-capture is not allowed for an extended host-device lambda
        auto lambda2 = [x = 1] __host__ __device__ () { return x; };

        int a = 1;
        // ERROR, an extended __device__ lambda cannot capture variables by reference
        auto lambda3 = [&a] __device__ () { return a; };

        // ERROR, by-reference capture is not allowed for an extended device-only lambda
        auto lambda4 = [&x = a] __device__ () { return x; };

        struct MyStruct {};
        MyStruct s1;
        // ERROR, a type local to a function cannot be used in the type of a captured variable
        auto lambda6 = [s1] __device__ () { };

        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda7 = [x = {11}] __device__ () { };

        std::initializer_list<int> b = {11,22,33};
        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda8 = [x = b] __device__ () { };

        int  var     = 4;
        auto lambda9 = [=] __device__ {
            int result = 0;
            if constexpr(false) {
                //ERROR, An extended device-only lambda cannot first-capture 'var' in if-constexpr context
                result += var;
            }
            return result;
        };

        auto lambda10 = [var] __device__ {
            int result = 0;
            if constexpr(false) {
                // CORRECT, 'var' already listed in explicit capture list for the extended lambda
                result += var;
            }
            return result;
        };

        auto lambda11 = [=] __device__ {
            int result = var;
            if constexpr(false) {
                // CORRECT, 'var' already implicit captured outside the 'if-constexpr' block
                result += var;
            }
            return result;
        };
    }
    ```
13. When parsing a function, the CUDA compiler assigns a counter value to each extended lambda in the function. This counter value is used in the substituted named type that is passed to the host compiler. Therefore, the presence or absence of an extended lambda within a function should not depend on a particular value of `__CUDA_ARCH__`, nor on `__CUDA_ARCH__` being undefined. Example:

    ```
    template <typename T>
    __global__ void kernel(T in) { in(); }

    __host__ __device__ void host_device_function() {
        // ERROR, the number and relative declaration order of
        //        extended lambdas depend on __CUDA_ARCH__
    #if defined(__CUDA_ARCH__)
        auto lambda1 = [] __device__ { return 0; };
        auto lambda2 = [] __host__ __device__ { return 10; };
    #endif
        auto lambda3 = [] __device__ { return 4; };
        kernel<<<1, 1>>>(lambda3);
    }
    ```
14. As described above, the CUDA compiler replaces a device extended lambda defined in a host function with a placeholder type defined in namespace scope. The placeholder type does not define an `operator()` function equivalent to the original lambda declaration unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true` for the closure type of the extended lambda. Therefore, an attempt to determine the return type or parameter types of the `operator()` function of such a lambda may work incorrectly in host code because the code processed by the host compiler is semantically different from the input code processed by the CUDA compiler. However, introspecting the return type or parameter types of the `operator()` function within device code is acceptable. Note that this restriction does not apply to host or device extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`. Example:

    ```
    #include <cuda/std/type_traits>

    const char& getRef(const char* p) { return *p; }

    void foo() {
        auto lambda1 = [] __device__ { return "10"; };

        // ERROR, attempt to extract the return type of a device lambda in host code
        cuda::std::result_of<decltype(lambda1)()>::type xx1 = "abc";

        auto lambda2 = [] __host__ __device__ { return "10"; };

        // CORRECT, lambda2 represents a host-device extended lambda
        cuda::std::result_of<decltype(lambda2)()>::type xx2 = "abc";

        auto lambda3 = [] __device__ () -> const char* { return "10"; };

        // CORRECT, lambda3 represents a device extended lambda with preserved return type
        cuda::std::result_of<decltype(lambda3)()>::type xx2 = "abc";
        static_assert(cuda::std::is_same_v<cuda::std::result_of<decltype(lambda3)()>::type, const char*>);

        auto lambda4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
        // lambda4's return type is not preserved because it references the operator()'s
        // parameter types in the trailing return type.
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(decltype(lambda4)));
    }
    ```
15. For an extended device-only lambda:

    * Introspection of the parameter type of `operator()` is only supported in device code.
    * Introspection of the return type of `operator()` is supported only in device code, unless the trait function `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`.
16. If an extended lambda is passed from host to device code as an argument to a `__global__` function, for example, then any expression in the lambda's body that captures variables must remain unchanged, regardless of whether the `__CUDA_ARCH__` macro is defined and what value it has. This restriction arises because the lambda's closure class layout depends on the order in which the compiler encounters the captured variables when processing the lambda expression. The program may execute incorrectly if the closure class layout differs between device and host compilations. Example:

    ```
    __device__ int result;

    template <typename T>
    __global__ void kernel(T in) { result = in(); }

    void foo(void) {
        int x1 = 1;
        // ERROR, "x1" is only captured when __CUDA_ARCH__ is defined.
        auto lambda1 = [=] __host__ __device__ {
    #ifdef __CUDA_ARCH__
            return x1 + 1;
    #else
            return 10;
    #endif
        };
        kernel<<<1, 1>>>(lambda1);
    }
    ```
17. As previously described, the CUDA compiler replaces an extended device-only lambda expression with a placeholder type instance in the code sent to the host compiler. The placeholder type does not define a pointer-to-function conversion operator in the host code; however, the conversion operator is provided in the device code. Note that this restriction does not apply to host-device extended lambdas. Example:

    ```
    template <typename T>
    __global__ void kernel(T in) {
        int (*fp)(double) = in;
        fp(0); // CORRECT, conversion in device code is supported
        auto lambda1 = [](double) { return 1; };
    }

    void foo() {
        auto lambda_device      = [] __device__ (double) { return 1; };
        auto lambda_host_device = [] __host__ __device__ (double) { return 1; };
        kernel<<<1, 1>>>(lambda_device);
        kernel<<<1, 1>>>(lambda_host_device);

        // CORRECT, conversion for a __host__ __device__ lambda is supported in host code
        int (*fp1)(double) = lambda_host_device;

        // ERROR, conversion for a device lambda is not supported in host code
        int (*fp2)(double) = lambda_device;
    }
    ```
18. As previously described, the CUDA compiler replaces an extended device-only or host-device lambda expression with a placeholder type instance in the code sent to the host compiler. This placeholder type may define C++ special member functions, such as constructors and destructors. Consequently, some standard C++ type traits may yield different results for the closure type of the extended lambda in the CUDA front-end compiler than in the host compiler. The following type traits are affected: : `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`. Care must be taken to ensure that the results of these traits are not used in the instantiation of the `__global__`, `__device__`, `__constant__`, or `__managed__` function or variable templates. Example:

    ```
    #include <cstdio>
    #include <type_traits>

    template <bool b>
    void __global__ kernel() { printf("hi"); }

    template <typename T>
    void kernel_launch() {
        // ERROR, this kernel launch may fail, because CUDA frontend compiler and host compiler
        //        may disagree on the result of std::is_trivially_copyable_v trait on the
        //        closure type of the extended lambda
        kernel<std::is_trivially_copyable_v<T>><<<1,1>>>();
        cudaDeviceSynchronize();
    }

    int main() {
        int  x       = 0;
        auto lambda1 = [=] __host__ __device__ () { return x; };
        kernel_launch<decltype(lambda1)>();
    }
    ```

The CUDA compiler will generate compiler diagnostics for a subset of cases described in `1-12`; no diagnostic will be generated for cases `13-17`, but the host compiler may fail to compile the generated code.

### 5.3.7.5. Host-Device Lambda Optimization Notes

Unlike device-only lambdas, host-device lambdas can be called from host code. As previously mentioned, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended host-device lambda invokes the original lambda's `operator()` with an indirect function call. The traits will always return false if extended lambda mode is not active.

The presence of an indirect function call may cause the host compiler to optimize an extended host-device lambda less than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the lambda body into the calling context. However, when it encounters an extended host-device lambda, the host compiler may not be able to easily inline the original lambda body.

### 5.3.7.6. `*this` Capture By-Value

According to C++11/C++14 rules, when a lambda is defined within a non-`static` class member function and the lambda's body refers to a class member variable, the `this` pointer of the class must be captured by value rather than the referenced member variable. If the lambda is an extended device-only or host-device lambda defined in a host function and executed on the GPU, accessing the referenced member variable on the GPU will cause a runtime error if the `this` pointer points to host memory.

Example:

```
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("value = %d\n", in()); }

struct MyStruct {
    int var;

    __host__ __device__ MyStruct() : var(10) {};

    void run() {
        auto lambda1 = [=] __device__ {
            // reference to "var" causes the 'this' pointer (MyStruct*) to be captured by value
            return var + 1;
        };
        // Kernel launch fails at run time because 'this->var' is not accessible from the GPU
        foo<<<1, 1>>>(lambda1);
        cudaDeviceSynchronize();
    }
};

int main() {
    MyStruct s1;
    s1.run();
}
```

C++17 solves this problem by introducing a new `*this` capture mode. In this mode, the compiler copies the object denoted by `*this` instead of capturing the `this` pointer by value. The `*this` capture mode is described in more detail in [P0018R3](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html).

The CUDA compiler supports the `*this` capture mode for lambdas defined within `__device__` and `__global__` functions and for extended device-only lambdas defined in host code, when the `--extended-lambda` flag is used.

Here's the above example modified to use `*this` capture mode:

```
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct MyStruct {
    int var;
    __host__ __device__ MyStruct() : var(10) { };

    void run() {
        // note the "*this" capture specification
        auto lambda1 = [=, *this] __device__ {
            // reference to "var" causes the object denoted by '*this' to be captured by
            // value, and the GPU code will access 'copy_of_star_this->var'
            return var + 1;
        };
        // Kernel launch succeeds
        foo<<<1, 1>>>(lambda1);
        cudaDeviceSynchronize();
    }
};

int main() {
    MyStruct s1;
    s1.run();
}
```

`*this` capture mode is not allowed for non-annotated lambdas defined in host code, or for extended host-device lambdas, unless `*this` capture is enabled by the selected language dialect. The following are examples of supported and unsupported usage:

```
struct MyStruct {
    int var;
    __host__ __device__ MyStruct() : var(10) { };

    void host_function() {
        // CORRECT, use in an extended device-only lambda
        auto lambda1 = [=, *this] __device__ { return var; };

        // Use in an extended host-device lambda
        // Error if *this capture not enabled by language dialect
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // Use in an non-annotated lambda in host function
        // Error if *this capture not enabled by language dialect
        auto lambda3 = [=, *this]  { return var; };
    }

    __device__ void device_function() {
        // CORRECT, use in a lambda defined in a device-only function
        auto lambda1 = [=, *this] __device__ { return var; };

        // CORRECT, use in a lambda defined in a device-only function
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // CORRECT, use in a lambda defined in a device-only function
        auto lambda3 = [=, *this]  { return var; };
    }

    __host__ __device__ void host_device_function() {
        // CORRECT, use in an extended device-only lambda
        auto lambda1 = [=, *this] __device__ { return var; };

        // Use in an extended host-device lambda
        // Error if *this capture not enabled by language dialect
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // Use in an unannotated lambda in a host-device function
        // Error if *this capture not enabled by language dialect
        auto lambda3 = [=, *this]  { return var; };
    }
};
```

### 5.3.7.7. Argument Dependent Lookup (ADL)

As previously mentioned, the CUDA compiler replaces an extended lambda expression with a placeholder type before invoking the host compiler. One template argument of the placeholder type uses the address of the function that encloses the original lambda expression. This may cause additional namespaces to participate in [Argument-Dependent Lookup (ADL)](https://en.cppreference.com/w/cpp/language/adl.html) for any host function call whose argument types involve the closure type of the extended lambda expression. Consequently, an incorrect function may be selected by the host compiler.

Example:

```
namespace N1 {

struct MyStruct {};

template <typename T>
void my_function(T);

}; // namespace N1

namespace N2 {

template <typename T>
int my_function(T);

template <typename T>
void run(T in) { my_function(in); }

} // namespace N2

void bar(N1::MyStruct in) {
    // For extended device-only lambda, the code sent to the host compiler is replaced with
    // the placeholder type instantiation expression
    //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
    //
    // As a result, the namespace 'N1' participates in ADL lookup of the
    // call to "my_function()" in the body of N2::run, causing ambiguity.
    auto lambda1 = [=] __device__ { };
    N2::run(lambda1);
}
```

In the above example, the CUDA compiler replaced the extended lambda with a placeholder type involving the `N1` namespace. Consequently, the `N1` namespace participates in the ADL lookup for `my_function(in)` in the body of `N2::run()`, resulting in a host compilation failure due to the discovery of multiple overload candidates: `N1::my_function` and `N2::my_function`.

