
CUDA supports the Bfloat16, half-, single-, double-, and quad-precision floating-point data types.
The following table summarizes the supported floating-point data types in CUDA and their requirements.

Table 43 Supported Floating-Point Types

| Precision / Name | Data Type | IEEE-754 | Header / Built-in | Requirements |
| --- | --- | --- | --- | --- |
| Bfloat16 | `__nv_bfloat16` | ❌ | `<cuda_bf16.h>` | Compute Capability 8.0 or higher. |
| Half Precision | `__half` | ✓ | `<cuda_fp16.h>` |  |
| Single Precision | `float` | ✓ | Built-in |  |
| Double Precision | `double` | ✓ | Built-in |  |
| Quad Precision | `__float128`/`_Float128` | ✓ | Built-in  `<crt/device_fp128_functions.h>` for mathematical functions | Host compiler support and Compute Capability 10.0 or higher.  The C or C++ spelling, `_Float128` and `__float128` respectively, also depends on the host compiler support. |

CUDA also supports [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (`TF32`), [microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) floating-point types, and other [lower precision numerical formats](https://resources.nvidia.com/en-us-blackwell-architecture) that are not intended for general-purpose computation, but rather for specialized purposes involving tensor cores. These include 4-, 6-, and 8-bit floating-point types. See the [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html) for more details.

The following figure reports the mantissa and exponent sizes of the supported floating-point data types.

![Floating-Point Types: Mantissa and Exponent sizes](img/Floating-Point-Types-Mantissa-and-Exponent-sizes.png)

The following table reports the ranges of the supported floating-point data types.

Table 44 Supported Floating-Point Types Properties

| Precision / Name | Largest Value | | Smallest Positive Value | | Smallest Positive Denormal | Epsilon |
| --- | --- | --- | --- | --- | --- | --- |
| Bfloat16 | \(\approx 2^{128}\) | \(\approx 3.39 \cdot 10^{38}\) | \(2^{-126}\) | \(\approx 1.18 \cdot 10^{-38}\) | \(2^{-133}\) | \(2^{-7}\) |
| Half Precision | \(\approx 2^{16}\) | \(65504\) | \(2^{-14}\) | \(\approx 6.1 \cdot 10^{-5}\) | \(2^{-24}\) | \(2^{-10}\) |
| Single Precision | \(\approx 2^{128}\) | \(\approx 3.40 \cdot 10^{38}\) | \(2^{-126}\) | \(\approx 1.18 \cdot 10^{-38}\) | \(2^{-149}\) | \(2^{-23}\) |
| Double Precision | \(\approx 2^{1024}\) | \(\approx 1.8 \cdot 10^{308}\) | \(2^{-1022}\) | \(\approx 2.22 \cdot 10^{-308}\) | \(2^{-1074}\) | \(2^{-52}\) |
| Quad Precision | \(\approx 2^{16384}\) | \(\approx 1.19 \cdot 10^{4932}\) | \(2^{-16382}\) | \(\approx 3.36 \cdot 10^{-4932}\) | \(2^{-16494}\) | \(2^{-112}\) |

Hint

The [CUDA C++ Standard Library](cpp-language-support.html#cpp-standard-library) provides `cuda::std::numeric_limits` in the `<cuda/std/limits>` header to query the properties and the ranges of the supported floating-point types, including [microscaling formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). See the [C++ reference](https://en.cppreference.com/w/cpp/types/numeric_limits.html) for the list of queryable properties.

**Complex numbers support:**

* The [CUDA C++ Standard Library](cpp-language-support.html#cpp-standard-library) supports complex numbers with the [cuda::std::complex](https://en.cppreference.com/w/cpp/numeric/complex) type in the `<cuda/std/complex>` header. See also the [libcu++ documentation](https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html) for more details.
* CUDA also provides basic support for complex numbers with the `cuComplex` and `cuDoubleComplex` types in the `cuComplex.h` header.

---

## 5.5.3. CUDA and IEEE-754 Compliance

All GPU devices follow the [IEEE 754-2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) standard for binary floating-point arithmetic with the following limitations:

* There is no dynamically configurable rounding mode; however, most of the operations support multiple constant IEEE rounding modes, selectable via specifically named [device intrinsics functions](#mathematical-functions-appendix-intrinsic-functions).
* There is no mechanism to detect floating-point exceptions, so all operations behave as if IEEE-754 exceptions are always masked. If there is an exceptional event, the default masked response defined by IEEE-754 is delivered. For this reason, although signaling NaN `SNaN` encodings are supported, they are not signaling and are handled as quiet exceptions.
* Floating-point operations may alter the bit patterns of input NaN payloads. Operations such as absolute value and negation may also not comply with the IEEE 754 requirement, which could result in the sign of a NaN being updated in an implementation-defined manner.

To maximize the portability of results, users are recommended to use the default settings of the `nvcc` compiler's floating-point options: `-ftz=false`, `-prec-div=true`, and `-prec-sqrt=true`, and not use the `--use_fast_math` option. Note that floating-point expression re-associations and contractions are allowed by default, similarly to the `--fmad=true` option. See also the `nvcc` [User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) for a detailed description of these compilation flags.

The IEEE-754 and C/C++ language standards do not explicitly address the conversion of a floating-point value to an integer value in cases where the rounded-to-integer value falls outside the range of the target integer format. The clamping behavior to the range of GPU devices is delineated in the [PTX ISA conversion instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) section. However, compiler optimizations may leverage the unspecified behavior clause when out-of-range conversion is not invoked directly via a PTX instruction, consequently resulting in undefined behavior and an invalid CUDA program. The CUDA Math documentation issues warnings to users on a per-function/intrinsic basis. For instance, consider the [\_\_double2int\_rz()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__double2int_rzd) instruction. This may differ from how host compilers and library implementations behave.

**Atomic Functions Denormals Behavior**:

Atomic operations have the following behavior regarding floating-point denormals, regardless of the setting of the compiler flag `-ftz`:

* Atomic single-precision floating-point adds on global memory always operate in flush-to-zero mode, namely behave equivalent to PTX `add.rn.ftz.f32` semantic.
* Atomic single-precision floating-point adds on shared memory always operate with denormal support, namely behave equivalent to PTX `add.rn.f32` semantic.

## 5.5.4. CUDA and C/C++ Compliance

**Floating-Point Exceptions:**

Unlike the host implementation, the mathematical operators and functions supported in device code do not set the global `errno` variable nor report [floating-point exceptions](https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions) to indicate errors. Thus, if error diagnostic mechanisms are required, users should implement additional input and output screening for the functions.

**Undefined Behavior with Floating-Point Operations:**

Common conditions of undefined behavior for mathematical operations include:

* Invalid arguments to mathematical operators and functions:

  + Using an uninitialized floating-point variable.
  + Using a floating-point variable outside its lifetime.
  + Signed integer overflow.
  + Dereferencing an invalid pointer.
* Floating-point specific undefined behavior:

  + Converting a floating-point value to an integer type for which the result is not representable is undefined behavior. This also includes NaN and infinity.

Users are responsible for ensuring the validity of a CUDA program. Invalid arguments may result in undefined behavior and be subject to compiler optimizations.

Contrary to integer division by zero, floating-point division by zero is not undefined behavior and not subject to compiler optimizations; rather, it is implementation-specific behavior. C++ implementations that conform to [IEC-60559](https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html) (IEEE-754), including CUDA, produce infinity. Note that invalid floating-point operations produce NaN and should not be misinterpreted as undefined behavior. Examples include zero divided by zero and infinity divided by infinity.

**Floating-Point Literals Portability:**

Both C and C++ allow for the representation of floating-point values in either decimal or hexadecimal notation. Hexadecimal floating-point literals, which are supported in [C99](https://en.cppreference.com/w/c/language/floating_constant.html) and [C++17](https://en.cppreference.com/w/cpp/language/floating_literal.html), denote a real value in scientific notation that can be precisely expressed in base-2. However, this does not guarantee that the literal will map to an actual value stored in a target variable (see the next paragraph). Conversely, a decimal floating-point literal may represent a numeric value that cannot be expressed in base-2.

According to the [C++ standard rules](https://eel.is/c++draft/lex.fcon#3), hexadecimal and decimal floating-point literals are rounded to the nearest representable value, larger or smaller, chosen in an implementation-defined manner. This rounding behavior may differ between the host and the device.

```
float f1 = 0.5f;    // 0.5, '0.5f' is a decimal floating-point literal
float f2 = 0x1p-1f; // 0.5, '0x1p-1f' is a hexadecimal floating-point literal
float f3 = 0.1f;
// f1, f2 are represented as 0 01111110 00000000000000000000000
// f3     is represented as  0 01111011 10011001100110011001101
```

The run-time and compile-time evaluations of the same floating-point expression are subject to the following portability issues:

* The run-time evaluation of a floating-point expression may be affected by the selected rounding mode, floating-point contraction (FMA) and reassociation compiler settings, as well as floating-point exceptions. Note that CUDA does not support floating-point exceptions and the [rounding mode](#floating-point-rounding) is set to *round-to-nearest-ties-to-even* by default. Other rounding modes can be selected using [intrinsic functions](#mathematical-functions-appendix-intrinsic-functions).
* The compiler may use a higher-precision internal representation for constant expressions.
* The compiler may perform optimizations, such as constant folding, constant propagation, and common subexpression elimination, which can lead to a different final value or comparison result.

**C Standard Math Library Notes:**

The host implementations of common mathematical functions are mapped to [C Standard Math Library functions](https://en.cppreference.com/w/c/header/math.html) in a platform-specific way. These functions are provided by the host compiler and the respective host `libm`, if available.

* Functions not available from the host compilers are implemented in the `crt/math_functions.h` header file. For example, `erfinv()` is implemented there.
* Less common functions, such as `rhypot()` and `cyl_bessel_i0()`, are only available in the device code.

As previously mentioned, the host and device implementations of mathematical functions are independent. For more details on the behavior of these functions, please refer to the host implementation's documentation.

---

## 5.5.5. Floating-Point Functionality Exposure

The mathematical functions supported by CUDA are exposed through the following methods:

[Built-in C/C++ language arithmetic operators](#builtin-math-operators):

* `x + y`, `x - y`, `x * y`, `x / y`, `x++`, `x--`, `x += y`, `x -= y`, `x *= y`, `x /= y`.
* Support single-, double-, and quad-precision types, `float`, `double`, and `__float128/_Float128` respectively.

  + `__half` and `__nv_bfloat16` types are also supported by including the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.
  + `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table.
* They are available in both host and device code.
* Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).

[CUDA C++ Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions):

* Expose the full set of C++ `<cmath>` [header functions](https://en.cppreference.com/w/cpp/header/cmath) through the `<cuda/std/cmath>` header and the `cuda::std::` namespace.
* Support IEEE-754 standard floating-point types, `__half`, `float`, `double`, `__float128`, as well as Bfloat16 `__nv_bfloat16`.

  + `__float128` support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table.
* They are available in both host and device code.
* They often rely on the [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html). Therefore, there could be different levels of accuracy between the host and device code.
* Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).
* A subset of functionalities is also supported in constant expressions, such as `constexpr` functions, in accordance with the C++23 and C++26 standard specifications.

[CUDA C Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

* Expose a subset of the C `<math.h>` [header functions](https://en.cppreference.com/w/c/header/math.html).
* Support single and double-precision types, `float` and `double` respectively.

  + They are available in both host and device code.
  + They don't require additional headers.
  + Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).
* A subset of the `<math.h>` header functionalities is also available for `__half`, `__nv_bfloat16`, and `__float128/_Float128` types. These functions have names that resemble those of the C Standard Library.

  + `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively. Their host and device code availability is defined on a per-function basis.
  + `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header and they are only available in device code.
* They can have a different accuracy between host and device code.

[Non-standard CUDA Mathematical functions](#mathematical-functions-appendix-additional-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

* Expose mathematical functionalities that are not part of the C/C++ Standard Library.
* Mainly support single- and double-precision types, `float` and `double` respectively.

  + Their host and device code availability is defined on a per-function basis.
  + They don't require additional headers.
  + They can have a different accuracy between host and device code.
* `__nv_bfloat16`, `__half`, `__float128/_Float128` are supported for a limited set of functions.

  + `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.
  + `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header.
  + They are only available in device code.
* Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).

[Intrinsic Mathematical functions](#mathematical-functions-appendix-intrinsic-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

* Support single- and double-precision types, `float` and `double` respectively.
* They are only available in device code.
* They are faster but less accurate than the respective [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html).
* Their behavior is not affected by the `nvcc` [floating-point optimization flags](../02-basics/nvcc.html#optimization-options) `-prec-div=false`, `-prec-sqrt=false`, and `-fmad=true`. The only exception is `-ftz=true`, which is also included in `-use_fast_math`.

Table 45 Summary of Math Functionality Features

| Functionality | Supported Types | Host | Device | Affected by Floating-Point Optimization Flags   (only for `float` and `double`) |
| --- | --- | --- | --- | --- |
| [Built-in C/C++ language arithmetic operators](#builtin-math-operators) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128/_Float128`, `cuda::std::complex` | ✓ | ✓ | ✓ |
| [CUDA C++ Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128`, `cuda::std::complex` | ✓ | ✓ | ✓ |
| `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` **\*** |
| [CUDA C Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double` | ✓ | ✓ | ✓ |
| `__nv_bfloat16`, `__half` with limited support and similar names | On a per-function basis | |
| `__float128/_Float128` with limited support and similar names | ❌ | ✓ |
| [Non-standard CUDA Mathematical functions](#mathematical-functions-appendix-additional-functions) | `float`, `double` | On a per-function basis | | ✓ |
| `__nv_bfloat16`, `__half`, `__float128/_Float128` with limited support | ❌ | ✓ |
| [Intrinsic functions](#mathematical-functions-appendix-intrinsic-functions) | `float`, `double` | ❌ | ✓ | Only with `-ftz=true`, also included in `-use_fast_math` |

**\*** The [CUDA C++ Standard Library functions](cpp-language-support.html#cpp-standard-library) support queries for small floating-point types, such as [numeric\_limits<T>](https://en.cppreference.com/w/cpp/types/numeric_limits.html), [fpclassify()](https://en.cppreference.com/w/cpp/numeric/math/fpclassify), [isfinite()](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html), [isnormal()](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html), [isinf()](https://en.cppreference.com/w/cpp/numeric/math/isinf.html), and [isnan()](https://en.cppreference.com/w/cpp/numeric/math/isnan.html).

The following sections provide accuracy information for some of these functions, when applicable. It uses ULP for quantification. For more information on the definition of the [Unit in the Last Place (ULP)](https://en.wikipedia.org/wiki/Unit_in_the_last_place), please see Jean-Michel Muller's paper [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf).

---

