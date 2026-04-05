# 5.5. Floating-Point Computation

## 5.5.1. Floating-Point Introduction

Since the adoption of the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) for Binary Floating-Point Arithmetic in 1985, virtually all mainstream computing systems, including NVIDIA's CUDA architectures, have implemented the standard. The IEEE-754 standard specifies how the results of floating-point arithmetic should be approximated.

To get accurate results and achieve the highest performance with the required precision, it is important to consider many aspects of floating-point behavior. This is particularly important in a heterogeneous computing environment where operations are performed on different types of hardware.

The following sections review the basic properties of floating-point computation and cover Fused Multiply-Add (FMA) operations and the dot product. These examples illustrate how different implementation choices affect accuracy.

### 5.5.1.1. Floating-Point Format

Floating-point format and functionality are defined in the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229).

The standard mandates that binary floating-point data be encoded on three fields:

* **Sign**: one bit to indicate a positive or negative number.
* **Exponent**: encodes the base 2 exponent offset by a numeric bias.
* **Significand** (also called *mantissa* or *fraction*): encodes the fractional value of the number.

![Floating-Point Encoding](img/Floating-Point-Encoding.png)

The latest IEEE-754 standard defines the encodings and properties of the following binary formats:

* 16-bit, also known as half-precision, corresponding to the `__half` data type in CUDA.
* 32-bit, also known as single-precision, corresponding to the `float` data type in C, C++, and CUDA.
* 64-bit, also known as double-precision, corresponding to the `double` data type in C, C++, and CUDA.
* 128-bit, also known as quad-precision, corresponding to the `__float128` or `_Float128` data types in CUDA.

These types have the following bit lengths:

![IEEE-754 Floating-Point Encodings](img/IEEE-754-Floating-Point-Encodings.png)

The numeric value associated with floating-point encoding for [normal](#normal-subnormal) values is computed as follows:

\[(-1)^\mathrm{sign} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}\]

For [subnormal](#normal-subnormal) values, the formula is modified to:

\[(-1)^\mathrm{sign} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}\]

The exponents are biased by \(127\) and \(1023\) for single- and double-precision, respectively. The integral part of \(1.\) is implicit in the fraction.

For example, the value \(-192 = (-1)^1 \times 2^7 \times 1.5\), and is encoded as a negative sign, an exponent of \(7\), and a fractional part \(0.5\). Hence the exponent \(7\) is represented by bit strings with values `7 + 127 = 134 = 10000110` for `float` and `7 + 1023 = 1030 = 10000000110` for `double`. The mantissa `0.5 = 2^-1` is represented by a binary value with `1` in the first position. The binary encodings of \(-192\) in single-precision and double-precision are shown in the following figure:

![Floating-Point Representation for ``-192``](img/Floating-Point-Representation-for--192.png)

Since the fraction field uses a limited number of bits, not all real numbers can be represented exactly. For instance, the binary representation of the mathematical value of the fraction \(2 / 3\) is `0.10101010...`, which has an infinite number of bits after the binary point. Therefore, \(2 / 3\) must be rounded before it can be represented as a floating-point number with limited precision. The rounding rules and modes are specified in IEEE-754. The most frequently used mode is *round-to-nearest-ties-to-even*, abbreviated round-to-nearest.

### 5.5.1.2. Normal and Subnormal Values

Any floating-point value with an exponent field that is neither all zeros nor all ones is called *normal*.

An important aspect of floating-point values is the wide gap between the smallest representable positive normal number, `FLT_MIN`, and zero. This gap is much wider than the gap between `FLT_MIN` and the second-smallest normal number.

Floating-point *subnormal* numbers, also called *denormals*, were introduced to address this issue. A subnormal floating-point value is represented with all bits in the exponent set to zero and at least one bit set in the significand. Subnormals are a required part of the IEEE-754 floating-point standard.

Subnormal numbers allow for a gradual loss of precision as an alternative to sudden rounding toward zero. However, subnormal numbers are computationally more expensive. Therefore, applications that don't require strict accuracy may choose to avoid them to improve performance. The `nvcc` compiler allows disabling subnormal numbers by setting the `-ftz=true` option (flush-to-zero), which is also included in `--use_fast_math`.

A simplified visualization of the encoding of the smallest normal value and subnormal values in single-precision is shown in the following figure:

![minimum normal value and subnormal values representations](img/minimum-normal-value-and-subnormal-values-representations.png)

where `X` represents both `0` and `1`.

### 5.5.1.3. Special Values

The IEEE-754 standard defines three special values for floating-point numbers:

**Zero:**

* Mathematical zero.
* Note that there are two possible representations of floating-point zero: `+0` and `-0`. This differs from the representation of integer zero.
* `+0 == -0` evaluates to `true`.
* Zero is encoded with all bits set to `0` in the exponent and significand.

**Infinity:**

* Floating-point numbers behave according to saturation arithmetic, in which operations that overflow the representable range result in `+Infinity` or `-Infinity`.
* Infinity is encoded with all bits in the exponent set to `1` and all bits in the significand set to `0`. There are exactly two encodings for infinity values.
* Arithmetic operations involving infinity and finite nonzero values typically result in infinity. Indeterminate forms such as `Inf * 0.0`, `Inf - Inf`, `Inf / Inf`, and `0.0 / 0.0` result in NaN.

**Not-a-Number (NaN):**

* NaN is a special symbol that represents an undefined or non-representable value. Common examples are `0.0 / 0.0`, `sqrt(-1.0)`, or `+Inf - Inf`.
* NaN is encoded with all bits in the exponent set to `1` and any bit pattern in the significand, except for all bits set to 0. There are \(2^{\mathrm{mantissa} + 1} - 2\) possible encodings.
* Any arithmetic operation involving a NaN will result in NaN.
* Any ordered comparison (`<`, `<=`, `>`, `>=`, `==`) involving a NaN will result in `false`, including `NaN == NaN` (non-reflexive). The unordered comparison `NaN != NaN` returns `true`.
* NaNs are provided in two forms:

  + Quiet NaNs `qNaN` are used to propagate errors resulting from invalid operations or values. Invalid arithmetic operations generally produce a quiet NaN. They are encoded with the most significant bit of the significand set to `1`.
  + Signaling NaNs `sNaN` are designed to raise an invalid-operation exception. Signaling NaNs are generally explicitly created. They are encoded with the most significant bit of the significand set to `0`.
  + The exact bit patterns for Quiet and Signaling NaNs are implementation-defined. CUDA provides the [cuda::std::numeric\_limits<T>::quiet\_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN.html) and [cuda::std::numeric\_limits<T>::signaling\_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN.html) constants to get their special values.

A simplified visualization of the encodings of special values is shown in the following figure:

![Floating-Point Representation for Infinity and NaN](img/Floating-Point-Representation-for-Infinity-and-NaN.png)

where `X` represents both `0` and `1`.

### 5.5.1.4. Associativity

It is important to note that the rules and properties of mathematical arithmetic do not directly apply to floating-point arithmetic due to its limited precision. The example below shows single-precision values `A`, `B`, and `C` and the exact mathematical value of their sum computed using different associativity.

\[\begin{split}\begin{aligned}
A &= 2^{1} \times 1.00000000000000000000001 \\
B &= 2^{0} \times 1.00000000000000000000001 \\
C &= 2^{3} \times 1.00000000000000000000001 \\
(A + B) + C &= 2^{3} \times 1.01100000000000000000001011 \\
A + (B + C) &= 2^{3} \times 1.01100000000000000000001011
\end{aligned}\end{split}\]

Mathematically, \((A + B) + C\) is equal to \(A + (B + C)\).

Let \(\mathrm{rn}(x)\) denote one rounding step on \(x\). Performing the same computations in single-precision floating-point arithmetic in round-to-nearest mode according to IEEE-754, we obtain:

\[\begin{split}\begin{aligned}
A + B &= 2^{1} \times 1.1000000000000000000000110000\ldots \\
\mathrm{rn}(A+B) &= 2^{1} \times 1.10000000000000000000010 \\
B + C &= 2^{3} \times 1.0010000000000000000000100100\ldots \\
\mathrm{rn}(B+C) &= 2^{3} \times 1.00100000000000000000001 \\
A + B + C &= 2^{3} \times 1.0110000000000000000000101100\ldots \\
\mathrm{rn}\big(\mathrm{rn}(A+B) + C\big) &= 2^{3} \times 1.01100000000000000000010 \\
\mathrm{rn}\big(A + \mathrm{rn}(B+C)\big) &= 2^{3} \times 1.01100000000000000000001
\end{aligned}\end{split}\]

For reference, the exact mathematical results are also computed above. The results computed according to IEEE-754 differ from the exact mathematical results. Additionally, the results corresponding to the sums \(\mathrm{rn}(\mathrm{rn}(A + B) + C)\) and \(\mathrm{rn}(A + \mathrm{rn}(B + C))\) differ from each other. In this case, \(\mathrm{rn}(A + \mathrm{rn}(B + C))\) is closer to the correct mathematical result than \(\mathrm{rn}(\mathrm{rn}(A + B) + C)\).

This example shows that seemingly identical computations can produce different results, even when all basic operations comply with IEEE-754.

### 5.5.1.5. Fused Multiply-Add (FMA)

The Fused Multiply-Add (FMA) operation computes the result with only one rounding step. Without the FMA, the result would require two rounding steps: one for multiplication and one for addition. Because the FMA uses only one rounding step, it produces a more accurate result.

The Fused Multiply-Add operation can affect the propagation of NaNs differently than two separate operations. However, FMA NaN handling is not universally identical across all targets. Different implementations with multiple NaN operands may prefer a quiet NaN or propagate one operand's payload. Additionally, IEEE-754 does not strictly mandate a deterministic payload selection order when multiple NaN operands are present. NaNs may also occur in intermediate computations, for example, \(\infty \times 0 + 1\) or \(1 \times \infty - \infty\), resulting in an implementation-defined NaN payload.

---

For clarity, first consider an example using decimal arithmetic to illustrate how the FMA operation works. We will compute \(x^2 - 1\) using five total digits of precision, with four digits after the decimal point.

* For \(x = 1.0008\), the correct mathematical result is \(x^2 - 1 = 1.60064 \times 10^{-4}\). The closest number using only four digits after the decimal point is \(1.6006 \times 10^{-4}\).
* The Fused Multiply-Add operation achieves the correct result using only one rounding step \(\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}\).
* The alternative is to compute the multiply and add steps separately. \(x^2 = 1.00160064\) translates to \(\mathrm{rn}(x \times x) = 1.0016\). The final result is \(\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}\).

Rounding the multiply and add separately yields a result that is off by \(0.00064\). The corresponding FMA computation is wrong by only \(0.00004\) and its result is closest to the correct mathematical answer. The results are summarized below:

\[\begin{split}\begin{aligned}
x &= 1.0008 \\
x^{2} &= 1.00160064 \\
x^{2} - 1 &= 1.60064 \times 10^{-4} && \text{true value} \\
\mathrm{rn}\big(x^{2} - 1\big) &= 1.6006 \times 10^{-4} && \text{fused multiply-add} \\
\mathrm{rn}\big(x^{2}\big) &= 1.0016 \\
\mathrm{rn}\big(\mathrm{rn}(x^{2}) - 1\big) &= 1.6000 \times 10^{-4} && \text{multiply, then add}
\end{aligned}\end{split}\]

---

Below is another example, using binary single precision values:

\[\begin{split}\begin{aligned}
A &= 2^{0} \times 1.00000000000000000000001 \\
B &= -2^{0} \times 1.00000000000000000000010 \\
\mathrm{rn}\big(A \times A + B\big) &= 2^{-46} \times 1.00000000000000000000000 && \text{fused multiply-add} \\
\mathrm{rn}\big(\mathrm{rn}(A \times A) + B\big) &= 0 && \text{multiply, then add}
\end{aligned}\end{split}\]

* Computing multiplication and addition separately results in the loss of all bits of precision, yielding \(0\).
* Computing the FMA, on the other hand, provides a result equal to the mathematical value.

Fused multiply-add helps prevent loss of precision during subtractive cancellation. Subtractive cancellation occurs when quantities of similar magnitude with opposite signs are added. In this case, many of the leading bits cancel out, resulting in fewer meaningful bits. The fused multiply-add computes a double-width product during multiplication. Thus, even if subtractive cancellation occurs during addition, there are enough valid bits remaining in the product to yield a precise result.

---

**Fused Multiply-Add Support in CUDA:**

CUDA provides the Fused Multiply-Add operation in several ways for both `float` and `double` data types:

* `x * y + z` when compiled with the flags `-fmad=true` or `--use_fast_math`.
* `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).
* `__fmaf_[rd, rn, ru, rz]`, `__fmaf_ieee_[rd, rn, ru, rz]`, and `__fma_[rd, rn, ru, rz]` [CUDA mathematical intrinsic functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).
* `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).

---

**Fused Multiply-Add Support on Host Platforms:**

Whether to use the fused operation depends on the availability of the operation on the platform and how the code is compiled. It is important to understand the host platform's support for Fused Multiply-Add when comparing CPU and GPU results.

* Compiler flags and Fused Multiply-Add hardware support:

  + `-mfma` with [GCC](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mmmx) and [Clang](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffp-contract), `-Mfma` with [NVC++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and `/fp:contract` with [Microsoft Visual Studio](https://learn.microsoft.com/en-us/cpp/preprocessor/fp-contract).
  + x86 platforms with the AVX2 ISA, for example, code compiled with the `-mavx2` flag using GCC or Clang, and `/arch:AVX2` with Microsoft Visual Studio.
  + Arm64 (AArch64) platforms with Advanced SIMD (Neon) ISA.
* `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).
* `std::fma(x, y, z)` and `std::fmaf(x, y, z)` [C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).
* `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).

### 5.5.1.6. Dot Product Example

Consider the problem of finding the dot product of two short vectors \(\overrightarrow{a}\) and \(\overrightarrow{b}\) both with four elements.

\[\begin{split}\overrightarrow{a} = \begin{bmatrix} a\_{1} \\ a\_{2} \\ a\_{3} \\ a\_{4} \end{bmatrix}
\qquad
\overrightarrow{b} = \begin{bmatrix} b\_{1} \\ b\_{2} \\ b\_{3} \\ b\_{4} \end{bmatrix}
\qquad
\overrightarrow{a} \cdot \overrightarrow{b} = a\_{1}b\_{1} + a\_{2}b\_{2} + a\_{3}b\_{3} + a\_{4}b\_{4}\end{split}\]

Although this operation is easy to write down mathematically, implementing it in software involves several alternatives that could lead to slightly different results. All of the strategies presented here use operations that are fully compliant with IEEE-754.

**Example Algorithm 1:** The simplest way to compute the dot product is to use a sequential sum of products, keeping the multiplications and additions separate.

> The final result can be represented as \(((((a\_1 \times b\_1) + (a\_2 \times b\_2)) + (a\_3 \times b\_3)) + (a\_4 \times b\_4))\).

**Example Algorithm 2:** Compute the dot product sequentially using fused multiply-add.

> The final result can be represented as \((a\_4 \times b\_4) + ((a\_3 \times b\_3) + ((a\_2 \times b\_2) + (a\_1 \times b\_1 + 0)))\).

**Example Algorithm 3:** Compute the dot product using a divide-and-conquer strategy. First, we find the dot products of the first and second halves of the vectors. Then, we combine these results using addition. This algorithm is called the "parallel algorithm" because the two subproblems can be computed in parallel since they are independent of each other. However, the algorithm does not require a parallel implementation; it can be implemented with a single thread.

> The final result can be represented as \(((a\_1 \times b\_1) + (a\_2 \times b\_2)) + ((a\_3 \times b\_3) + (a\_4 \times b\_4))\).

### 5.5.1.7. Rounding

The IEEE-754 standard requires support for several operations. These include arithmetic operations such as addition, subtraction, multiplication, division, square root, fused multiply-add, finding the remainder, conversion, scaling, sign, and comparison operations. The results of these operations are guaranteed to be consistent across all implementations of the standard for a given format and rounding mode.

---

**Rounding Modes**

The IEEE-754 standard defines four rounding modes: *round-to-nearest*, *round towards positive*, *round towards negative*, and *round towards zero*. CUDA supports all four modes. By default, operations use *round-to-nearest*. [Intrinsic mathematical functions](#mathematical-functions-appendix-intrinsic-functions) can be used to select other rounding modes for individual operations.

| Rounding Mode | Interpretation |
| --- | --- |
| `rn` | Round to nearest, ties to even |
| `rz` | Round towards zero |
| `ru` | Round towards \(\infty\) |
| `rd` | Round towards \(-\infty\) |

### 5.5.1.8. Notes on Host/Device Computation Accuracy

The accuracy of a floating-point computation result is affected by several factors. This section summarizes important considerations for achieving reliable results in floating-point computations. Some of these aspects have been described in greater detail in previous sections.

These aspects are also important when comparing the results between CPU and GPU. Differences between host and device execution must be interpreted carefully. The presence of differences does not necessarily mean the GPU's result is incorrect or that there is a problem with the GPU.

**Associativity**:

> Floating-point addition and multiplication in finite precision are not [associative](#associativity) because they often result in mathematical values that cannot be directly represented in the target format, requiring rounding. The order in which these operations are evaluated affects how rounding errors accumulate and can significantly alter the final result.

**Fused Multiply-Add**:

> [Fused Multiply-Add](#fused-multiply-add) computes \(a \times b + c\) in a single operation, resulting in greater accuracy and a faster execution time. The accuracy of the final result can be affected by its use. Fused Multiply-Add relies on hardware support and can be enabled either explicitly by calling the related function or implicitly through compiler optimization flags.

**Precision**:

> Increasing the floating-point precision can potentially improve the accuracy of the results. Higher precision reduces loss of significance and enables the representation of a wider range of values. However, higher precision types have lower throughput and consume more registers. Additionally, using them to explicitly store input and output increases memory usage and data movement.

**Compiler Flags and Optimizations**:

> All major compilers provide a variety of optimization flags to control the behavior of floating-point operations.
>
> * The highest optimization level for GCC (`-O3`), Clang (`-O3`), nvcc (`-O3`), and Microsoft Visual Studio (`/O2`) does not affect floating-point semantics. However, inlining, loop unrolling, vectorization, and common subexpression elimination could affect the results. The NVC++ compiler also requires the flags `-Kieee -Mnofma` for IEEE-754-compliant semantics.
> * Refer to the [GCC](https://gcc.gnu.org/wiki/FloatingPointMath), [Clang](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior), [Microsoft Visual Studio Compiler](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior), [nvc++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and [Arm C/C++ compiler](https://developer.arm.com/documentation/101458/2404/Compiler-options?lang=en) documentation for detailed information about options that affect floating-point behavior.
> * See also the `nvcc` [User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) for detailed descriptions of compiler flags that specifically affect floating-point behavior in CUDA device code: `-ftz`, `-prec-div`, `-prec-sqrt`, `-fmad`, `--use_fast_math`. Besides these floating-point options, it is also important to verify the effects of other compiler optimizations in the context of the user program. Users are encouraged to verify the correctness of their results with extensive testing and compare results obtained with optimizations enabled versus all device code optimizations disabled; see also the `-G` compiler flag.

**Library Implementations**:

> Functions defined outside the IEEE-754 standard are not guaranteed to be correctly rounded and depend on implementation-defined behavior. Therefore, the results may differ across different platforms, including between host, device, and different device architectures.

**Deterministic Results**:

> A deterministic result refers to computing the same bit-wise numerical outputs every time when run with the same inputs under the same specified conditions. Such conditions include:
>
> * Hardware dependencies, such as execution on the same CPU processor or GPU device.
> * Compiler aspects, such as the version of the compiler and the [Compiler Flags and Optimizations](#compiler-flags-and-optimizations).
> * Run-time conditions that affect the computation, such as [rounding mode](#floating-point-rounding) or environment variables.
> * Identical inputs to the computation.
> * Thread configuration, including the number of threads involved in the computation and their organization, for example block and grid size.
> * The ordering of [arithmetic atomic operations](cpp-language-extensions.html#atomic-functions) depends on hardware scheduling which can vary between runs.

**Taking Advantage of the CUDA Libraries**:

> The [CUDA Math Libraries](https://developer.nvidia.com/gpu-accelerated-libraries), [C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html), and [C++ Standard Library Mathematical functions](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) are designed to boost developer productivity for common functionalities, particularly for floating-point math and numerics-intensive routines. These functionalities provide a consistent high-level interface, are optimized, and are widely tested across platforms and edge cases. Users are encouraged to take full advantage of these libraries and avoid tedious manual reimplementations.

## 5.5.2. Floating-Point Data Types
