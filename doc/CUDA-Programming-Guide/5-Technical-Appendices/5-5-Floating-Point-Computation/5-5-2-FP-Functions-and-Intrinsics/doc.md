## 5.5.6. Built-In Arithmetic Operators

The built-in C/C++ language operators, such as `x + y`, `x - y`, `x * y`, `x / y`, `x++`, `x--`, and reciprocal `1 / x`, for single-, double-, and quad-precision types comply with the IEEE-754 standard. They guarantee a maximum ULP error of zero using a *round-to-nearest-ties-to-even* rounding mode. They are available in both host and device code.

The `nvcc` compilation flag `-fmad=true`, also included in `--use_fast_math`, enables contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations and has the following effect on the maximum ULP error for the single-precision type `float`:

* `x * y + z` → [\_\_fmaf\_rn(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff): 0 ULP

The `nvcc` compilation flag `-prec-div=false`, also included in `--use_fast_math`, has the following effect on the maximum ULP error for the division operator `/` for the single-precision type `float`:

* `x / y` → [\_\_fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107): 2 ULP
* `1 / x`: 1 ULP

---

## 5.5.7. CUDA C++ Mathematical Standard Library Functions

CUDA provides comprehensive support for [C++ Standard Library mathematical functions](https://en.cppreference.com/w/cpp/header/cmath.html) through the `cuda::std::` namespace. The functionalities are part of the `<cuda/std/cmath>` header.
They are available in both host and device code.

The following sections specify the mapping with the [CUDA Math APIs](https://docs.nvidia.com/cuda/cuda-math-api/index.html) and the error bounds of each function when executed on the device.

* The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the *round-to-nearest ties-to-even* rounding mode.
* The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.

### 5.5.7.1. Basic Operations

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for basic operations are available in both host and device code, except for `__float128`.

All the following functions have a maximum ULP error of zero.

Table 46 C++ Mathematical Standard Library Functions
 C Math API Mapping
 **Basic Operations**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [fabs(x)](https://en.cppreference.com/w/cpp/numeric/math/fabs.html) | \(|x|\) | [\_\_habs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__habsK13__nv_bfloat16) | [\_\_habs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__habsK6__half) | [fabsf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fabsff) | [fabs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fabsd) | [\_\_nv\_fp128\_fabs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fabsg) |
| [fmod(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmod.html) | Remainder of \(\dfrac{x}{y}\), computed as \(x - \mathrm{trunc}\left(\dfrac{x}{y}\right) \cdot y\) | N/A | N/A | [fmodf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmodfff) | [fmod(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmoddd) | [\_\_nv\_fp128\_fmod(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmodgg) |
| [remainder(x, y)](https://en.cppreference.com/w/cpp/numeric/math/remainder.html) | Remainder of \(\dfrac{x}{y}\), computed as \(x - \mathrm{rint}\left(\dfrac{x}{y}\right) \cdot y\) | N/A | N/A | [remainderf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410remainderfff) | [remainder(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49remainderdd) | [\_\_nv\_fp128\_remainder(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv420__nv_fp128_remaindergg) |
| [remquo(x, y, iptr)](https://en.cppreference.com/w/cpp/numeric/math/remquo.html) | Remainder and quotient of \(\dfrac{x}{y}\) | N/A | N/A | [remquof(x, y, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47remquofffPi) | [remquo(x, y, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46remquoddPi) | N/A |
| [fma(x, y, z)](https://en.cppreference.com/w/cpp/numeric/math/fma.html) | \(x \cdot y + z\) | [\_\_hfma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__hfmaK13__nv_bfloat16K13__nv_bfloat16K13__nv_bfloat16), device-only | [\_\_hfma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__hfmaK6__halfK6__halfK6__half), device-only | [fmaf(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff) | [fma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43fmaddd) | [\_\_nv\_fp128\_fma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_fmaggg) |
| [fmax(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmax.html) | \(\max(x, y)\) | [\_\_hmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hmaxK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hmaxK6__halfK6__half) | [fmaxf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmaxfff) | [fmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmaxdd) | [\_\_nv\_fp128\_fmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmaxgg) |
| [fmin(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmin.html) | \(\min(x, y)\) | [\_\_hmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hminK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hminK6__halfK6__half) | [fminf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fminfff) | [fmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmindd) | [\_\_nv\_fp128\_fmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmingg) |
| [fdim(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fdim.html) | \(\max(x-y, 0)\) | N/A | N/A | [fdimf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fdimfff) | [fdim(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fdimdd) | [\_\_nv\_fp128\_fdim(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fdimgg) |
| [nan(str)](https://en.cppreference.com/w/cpp/numeric/math/nan.html) | NaN value from string representation | N/A | N/A | [nanf(str)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44nanfPKc) | [nan(str)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43nanPKc) | N/A |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.2. Exponential Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for exponential functions are available in both host and device code only for `float` and `double` types.

Table 47 C++ Mathematical Standard Library Functions
 C Math API Mapping and Accuracy (Maximum ULP)
 **Exponential Functions**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [exp(x)](https://en.cppreference.com/w/cpp/numeric/math/exp.html) | \(e^x\) | [hexp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hexpK13__nv_bfloat16)     0 ULP | [hexp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hexpK6__half)     0 ULP | [expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff)     2 ULP | [exp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43expd)     1 ULP | [\_\_nv\_fp128\_exp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_expg)     1 ULP |
| [exp2(x)](https://en.cppreference.com/w/cpp/numeric/math/exp2.html) | \(2^x\) | [hexp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hexp2K13__nv_bfloat16)     0 ULP | [hexp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hexp2K6__half)     0 ULP | [exp2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45exp2ff)     2 ULP | [exp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44exp2d)     1 ULP | [\_\_nv\_fp128\_exp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_exp2g)     1 ULP |
| [expm1(x)](https://en.cppreference.com/w/cpp/numeric/math/expm1.html) | \(e^x - 1\) | N/A | N/A | [expm1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46expm1ff)     1 ULP | [expm1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45expm1d)     1 ULP | [\_\_nv\_fp128\_expm1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_expm1g)     1 ULP |
| [log(x)](https://en.cppreference.com/w/cpp/numeric/math/log.html) | \(\ln(x)\) | [hlog(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hlogK13__nv_bfloat16)     0 ULP | [hlog(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hlogK6__half)     0 ULP | [logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff)     1 ULP | [log(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43logd)     1 ULP | [\_\_nv\_fp128\_log(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_logg)     1 ULP |
| [log10(x)](https://en.cppreference.com/w/cpp/numeric/math/log10.html) | \(\log\_{10}(x)\) | [hlog10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hlog10K13__nv_bfloat16)     0 ULP | [hlog10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hlog10K6__half)     0 ULP | [log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff)     2 ULP | [log10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log10d)     1 ULP | [\_\_nv\_fp128\_log10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log10g)     1 ULP |
| [log2(x)](https://en.cppreference.com/w/cpp/numeric/math/log2.html) | \(\log\_2(x)\) | [hlog2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hlog2K13__nv_bfloat16)     0 ULP | [hlog2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hlog2K6__half)     0 ULP | [log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff)     1 ULP | [log2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44log2d)     1 ULP | [\_\_nv\_fp128\_log2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_log2g)     1 ULP |
| [log1p(x)](https://en.cppreference.com/w/cpp/numeric/math/log1p.html) | \(\ln(1+x)\) | N/A | N/A | [log1pf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log1pff)     1 ULP | [log1p(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log1pd)     1 ULP | [\_\_nv\_fp128\_log1p(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log1pg)     1 ULP |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.3. Power Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for power functions are available in both host and device code only for `float` and `double` types.

Table 48 C++ Mathematical Standard Library Functions
 C Math API Mapping and Accuracy (Maximum ULP)
 **Power Functions**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [pow(x, y)](https://en.cppreference.com/w/cpp/numeric/math/pow.html) | \(x^y\) | N/A | N/A | [powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff)     4 ULP | [pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43powdd)     2 ULP | [\_\_nv\_fp128\_pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_powgg)     1 ULP |
| [sqrt(x)](https://en.cppreference.com/w/cpp/numeric/math/sqrt.html) | \(\sqrt{x}\) | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hsqrtK13__nv_bfloat16)     0 ULP | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hsqrtK6__half)     0 ULP | [sqrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sqrtff)     ≤ 0 ULP   ≤ 1 ULP with `--use_fast_math` | [sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sqrtd)     0 ULP | [\_\_nv\_fp128\_sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sqrtg)     0 ULP |
| [cbrt(x)](https://en.cppreference.com/w/cpp/numeric/math/cbrt.html) | \(\sqrt[3]{x}\) | N/A | N/A | [cbrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45cbrtff)     1 ULP | [cbrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44cbrtd)     1 ULP | N/A |
| [hypot(x, y)](https://en.cppreference.com/w/cpp/numeric/math/hypot.html) | \(\sqrt{x^2 + y^2}\) | N/A | N/A | [hypotf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46hypotfff)     3 ULP | [hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45hypotdd)     2 ULP | [\_\_nv\_fp128\_hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_hypotgg)     1 ULP |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.4. Trigonometric Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for trigonometric functions are available in both host and device code only for `float` and `double` types.

Table 49 C++ Mathematical Standard Library Functions
 C Math API Mapping and Accuracy (Maximum ULP)
 **Trigonometric Functions**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [sin(x)](https://en.cppreference.com/w/cpp/numeric/math/sin.html) | \(\sin(x)\) | [hsin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hsinK13__nv_bfloat16)     0 ULP | [hsin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hsinK6__half)     0 ULP | [sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff)     2 ULP | [sin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43sind)     2 ULP | [\_\_nv\_fp128\_sin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_sing)     1 ULP |
| [cos(x)](https://en.cppreference.com/w/cpp/numeric/math/cos.html) | \(\cos(x)\) | [hcos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hcosK13__nv_bfloat16)     0 ULP | [hcos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hcosK6__half)     0 ULP | [cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff)     2 ULP | [cos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43cosd)     2 ULP | [\_\_nv\_fp128\_cos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_cosg)     1 ULP |
| [tan(x)](https://en.cppreference.com/w/cpp/numeric/math/tan.html) | \(\tan(x)\) | N/A | N/A | [tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff)     4 ULP | [tan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43tand)     2 ULP | [\_\_nv\_fp128\_tan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_tang)     1 ULP |
| [asin(x)](https://en.cppreference.com/w/cpp/numeric/math/asin.html) | \(\sin^{-1}(x)\) | N/A | N/A | [asinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45asinff)     2 ULP | [asin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44asind)     2 ULP | [\_\_nv\_fp128\_asin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_asing)     1 ULP |
| [acos(x)](https://en.cppreference.com/w/cpp/numeric/math/acos.html) | \(\cos^{-1}(x)\) | N/A | N/A | [acosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45acosff)     2 ULP | [acos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44acosd)     2 ULP | [\_\_nv\_fp128\_acos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_acosg)     1 ULP |
| [atan(x)](https://en.cppreference.com/w/cpp/numeric/math/atan.html) | \(\tan^{-1}(x)\) | N/A | N/A | [atanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45atanff)     2 ULP | [atan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44atand)     2 ULP | [\_\_nv\_fp128\_atan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_atang)     1 ULP |
| [atan2(y, x)](https://en.cppreference.com/w/cpp/numeric/math/atan2.html) | \(\tan^{-1}\left(\dfrac{y}{x}\right)\) | N/A | N/A | [atan2f(y, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atan2fff)     3 ULP | [atan2(y, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atan2dd)     2 ULP | N/A |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.5. Hyperbolic Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for hyperbolic functions are available in both host and device code only for `float` and `double` types.

Table 50 C++ Mathematical Standard Library Functions
 C Math API Mapping and Accuracy (Maximum ULP)
 **Hyperbolic Functions**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [sinh(x)](https://en.cppreference.com/w/cpp/numeric/math/sinh.html) | \(\sinh(x)\) | N/A | N/A | [sinhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sinhff)     3 ULP | [sinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sinhd)     2 ULP | [\_\_nv\_fp128\_sinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sinhg)     1 ULP |
| [cosh(x)](https://en.cppreference.com/w/cpp/numeric/math/cosh.html) | \(\cosh(x)\) | N/A | N/A | [coshf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45coshff)     2 ULP | [cosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44coshd)     1 ULP | [\_\_nv\_fp128\_cosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_coshg)     1 ULP |
| [tanh(x)](https://en.cppreference.com/w/cpp/numeric/math/tanh.html) | \(\tanh(x)\) | [htanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45htanhK13__nv_bfloat16)     0 ULP | [htanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45htanhK6__half)     0 ULP | [tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff)     2 ULP | [tanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44tanhd)     1 ULP | [\_\_nv\_fp128\_tanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_tanhg)     1 ULP |
| [asinh(x)](https://en.cppreference.com/w/cpp/numeric/math/asinh.html) | \(\operatorname{sinh}^{-1}(x)\) | N/A | N/A | [asinhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46asinhff)     3 ULP | [asinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45asinhd)     3 ULP | [\_\_nv\_fp128\_asinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_asinhg)     1 ULP |
| [acosh(x)](https://en.cppreference.com/w/cpp/numeric/math/acosh.html) | \(\operatorname{cosh}^{-1}(x)\) | N/A | N/A | [acoshf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46acoshff)     4 ULP | [acosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45acoshd)     3 ULP | [\_\_nv\_fp128\_acosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_acoshg)     1 ULP |
| [atanh(x)](https://en.cppreference.com/w/cpp/numeric/math/atanh.html) | \(\operatorname{tanh}^{-1}(x)\) | N/A | N/A | [atanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atanhff)     3 ULP | [atanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atanhd)     2 ULP | [\_\_nv\_fp128\_atanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_atanhg)     1 ULP |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.6. Error and Gamma Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for error and gamma functions are available in both host and device code for `float` and `double` types.

Error and Gamma functions are not natively available for CUDA-extended floating-point types, such as `__half` and `__nv_bfloat16`. In these cases, the functions are emulated by converting to a `float` type and then converting the result back.

Table 51 C++ Mathematical Standard Library Functions
 C Math API Mapping and Accuracy (Maximum ULP)
 **Error and Gamma Functions**

| `cuda::std` Function | Meaning | `float` | `double` |
| --- | --- | --- | --- |
| [erf(x)](https://en.cppreference.com/w/cpp/numeric/math/erf.html) | \(\dfrac{2}{\sqrt{\pi}} \int\_0^x e^{-t^2} dt\) | [erff(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44erfff)     2 ULP | [erf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43erfd)     2 ULP |
| [erfc(x)](https://en.cppreference.com/w/cpp/numeric/math/erfc.html) | \(1 - \mathrm{erf}(x)\) | [erfcf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45erfcff)     4 ULP | [erfc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44erfcd)     5 ULP |
| [tgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/tgamma.html) | \(\Gamma(x)\) | [tgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47tgammaff)     5 ULP | [tgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46tgammad)     10 ULP |
| [lgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/lgamma.html) | \(\ln |\Gamma(x)|\) | [lgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lgammaff)     ≤ 6 ULP for \(x \notin [-10.001, -2.264]\)   ≤ larger otherwise | [lgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lgammad)     ≤ 4 ULP for \(x \notin [-23.0001, -2.2637]\)   ≤ larger otherwise |

### 5.5.7.7. Nearest Integer Floating-Point Operations

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for nearest integer floating-point operations are available in both host and device code only for `float` and `double` types.

All the following functions have a maximum ULP error of zero.

Table 52 C++ Mathematical Standard Library Functions
 C Math API Mapping
 **Nearest Integer Floating-Point Operations**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [ceil(x)](https://en.cppreference.com/w/cpp/numeric/math/ceil.html) | \(\lceil x \rceil\) | [hceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hceilK13__nv_bfloat16) | [hceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hceilK6__half) | [ceilf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45ceilff) | [ceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44ceild) | [\_\_nv\_fp128\_ceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_ceilg) |
| [floor(x)](https://en.cppreference.com/w/cpp/numeric/math/floor.html) | \(\lfloor x \rfloor\) | [hfloor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hfloorK13__nv_bfloat16) | [hfloor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hfloorK6__half) | [floorf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46floorff) | [floor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45floord) | [\_\_nv\_fp128\_floor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_floorg) |
| [trunc(x)](https://en.cppreference.com/w/cpp/numeric/math/trunc.html) | Truncate to integer | [htrunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46htruncK13__nv_bfloat16) | [htrunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46htruncK6__half) | [truncf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46truncff) | [trunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45truncd) | [\_\_nv\_fp128\_trunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_truncg) |
| [round(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero | N/A | N/A | [roundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46roundff) | [round(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45roundd) | [\_\_nv\_fp128\_round(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_roundg) |
| [nearbyint(x)](https://en.cppreference.com/w/cpp/numeric/math/nearbyint.html) | Round to nearest integer, ties to even | N/A | N/A | [nearbyintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nearbyintff) | [nearbyint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nearbyintd) | N/A |
| [rint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even | [hrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hrintK13__nv_bfloat16) | [hrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hrintK6__half) | [rintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45rintff) | [rint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44rintd) | [\_\_nv\_fp128\_rint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_rintg) |
| [lrint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even (returns `long int`) | N/A | N/A | [lrintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46lrintff) | [lrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45lrintd) | N/A |
| [llrint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even (returns `long long int`) | N/A | N/A | [llrintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47llrintff) | [llrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46llrintd) | N/A |
| [lround(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero (returns `long int`) | N/A | N/A | [lroundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lroundff) | [lround(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lroundd) | N/A |
| [llround(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero (returns `long long int`) | N/A | N/A | [llroundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48llroundff) | [llround(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47llroundd) | N/A |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

**Performance Considerations**

The recommended way to round a single- or double-precision floating-point operand to an integer is to use the functions `rintf()` and `rint()`, not `roundf()` and `round()`. This is because `roundf()` and `round()` map to multiple instructions in device code, whereas `rintf()` and `rint()` map to a single instruction. `truncf()`, `trunc()`, `ceilf()`, `ceil()`, `floorf()`, and `floor()` each map to a single instruction as well.

### 5.5.7.8. Floating-Point Manipulation Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for floating-point manipulation functions are available in both host and device code, except for `__float128`.

Floating-point manipulation functions are not natively available for CUDA-extended floating-point types, such as `__half` and `__nv_bfloat16`. In these cases, the functions are emulated by converting to a `float` type and then converting the result back.

All the following functions have a maximum ULP error of zero.

Table 53 C++ Mathematical Standard Library Functions
 C Math API Mapping
 **Floating-Point Manipulation Functions**

| `cuda::std` Function | Meaning | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- |
| [frexp(x, exp)](https://en.cppreference.com/w/cpp/numeric/math/frexp.html) | Extract mantissa and exponent | [frexpf(x, exp)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46frexpffPi) | [frexp(x, exp)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45frexpdPi) | [\_\_nv\_fp128\_frexp(x, nptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_frexpgPi) |
| [ldexp(x, n)](https://en.cppreference.com/w/cpp/numeric/math/ldexp.html) | \(x \cdot 2^{\mathrm{n}}\) | [ldexpf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ldexpffi) | [ldexp(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ldexpdi) | [\_\_nv\_fp128\_ldexp(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ldexpgi) |
| [modf(x, iptr)](https://en.cppreference.com/w/cpp/numeric/math/modf.html) | Extract integer and fractional parts | [modff(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45modfffPf) | [modf(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44modfdPd) | [\_\_nv\_fp128\_modf(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_modfgPg) |
| [scalbn(x, n)](https://en.cppreference.com/w/cpp/numeric/math/scalbn.html) | \(x \cdot 2^n\) | [scalbnf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47scalbnffi) | [scalbn(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46scalbndi) | N/A |
| [scalbln(x, n)](https://en.cppreference.com/w/cpp/numeric/math/scalbn.html) | \(x \cdot 2^n\) | [scalblnf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48scalblnffl) | [scalbln(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47scalblndl) | N/A |
| [ilogb(x)](https://en.cppreference.com/w/cpp/numeric/math/ilogb.html) | \(\lfloor \log\_2(|x|) \rfloor\) | [ilogbf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ilogbff) | [ilogb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ilogbd) | [\_\_nv\_fp128\_ilogb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ilogbg) |
| [logb(x)](https://en.cppreference.com/w/cpp/numeric/math/logb.html) | \(\lfloor \log\_2(|x|) \rfloor\) | [logbf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45logbff) | [logb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44logbd) | N/A |
| [nextafter(x, y)](https://en.cppreference.com/w/cpp/numeric/math/nextafter.html) | Next representable value toward \(y\) | [nextafterf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nextafterfff) | [nextafter(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nextafterdd) | N/A |
| [copysign(x, y)](https://en.cppreference.com/w/cpp/numeric/math/copysign.html) | Copy sign of \(y\) to \(x\) | [copysignf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49copysignfff) | [copysign(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48copysigndd) | [\_\_nv\_fp128\_copysign(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv419__nv_fp128_copysigngg) |

### 5.5.7.9. Classification and Comparison

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for classification and comparison functions are available in both host and device code, except for `__float128`.

All the following functions have a maximum ULP error of zero.

Table 54 C++ Mathematical Standard Library Functions
 C Math API Mapping
 **Classification and Comparison Functions**

| `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128` |
| --- | --- | --- | --- | --- | --- | --- |
| [fpclassify(x)](https://en.cppreference.com/w/cpp/numeric/math/fpclassify.html) | Classify \(x\) | N/A | N/A | N/A | N/A | N/A |
| [isfinite(x)](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html) | Check if \(x\) is finite | N/A | N/A | [isfinite(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48isfinitef) | [isfinite(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48isfinited) | N/A |
| [isinf(x)](https://en.cppreference.com/w/cpp/numeric/math/isinf.html) | Check if \(x\) is infinite | [\_\_hisinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisinfK13__nv_bfloat16) | [\_\_hisinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisinfK6__half) | [isinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isinff) | [isinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isinfd) | N/A |
| [isnan(x)](https://en.cppreference.com/w/cpp/numeric/math/isnan.html) | Check if \(x\) is NaN | [\_\_hisnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisnanK13__nv_bfloat16) | [\_\_hisnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisnanK6__half) | [isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isnanf) | [isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isnand) | [\_\_nv\_fp128\_isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_isnang) |
| [isnormal(x)](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html) | Check if \(x\) is normal | N/A | N/A | N/A | N/A | N/A |
| [signbit(x)](https://en.cppreference.com/w/cpp/numeric/math/signbit.html) | Check if sign bit is set | N/A | N/A | [signbit(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47signbitf) | [signbit(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47signbitd) | N/A |
| [isgreater(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isgreater.html) | Check if \(x > y\) | [\_\_hgt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgtK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hgt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hgt6__half6__half) | N/A | N/A | N/A |
| [isgreaterequal(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal.html) | Check if \(x \geq y\) | [\_\_hge(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgeK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hge(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hge6__half6__half) | N/A | N/A | N/A |
| [isless(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isless.html) | Check if \(x < y\) | [\_\_hlt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hltK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hlt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hlt6__half6__half) | N/A | N/A | N/A |
| [islessequal(x, y)](https://en.cppreference.com/w/cpp/numeric/math/islessequal.html) | Check if \(x \leq y\) | [\_\_hle(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hleK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hle(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hle6__half6__half) | N/A | N/A | N/A |
| [islessgreater(x, y)](https://en.cppreference.com/w/cpp/numeric/math/islessgreater.html) | Check if \(x < y\) or \(x > y\) | [\_\_hne(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hneK13__nv_bfloat16K13__nv_bfloat16) | [\_\_hne(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hneK6__halfK6__half) | N/A | N/A | N/A |
| [isunordered(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isunordered.html) | Check if \(x\), \(y\), or both are NaN | N/A | N/A | N/A | N/A | [\_\_nv\_fp128\_isunordered(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv422__nv_fp128_isunorderedgg) |

**\*** Mathematical functions marked with "N/A" are not natively available for CUDA-extended floating-point types, such as \_\_half and \_\_nv\_bfloat16.

## 5.5.8. Non-Standard CUDA Mathematical Functions

CUDA provides mathematical functions that are not part of the C/C++ Standard Library and are instead offered as extensions. For single- and double-precision functions, host and device code availability is defined on a per-function basis.

This section specifies the error bounds of each function when executed on the device.

* The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the *round-to-nearest ties-to-even* rounding mode.
* The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.

Table 55 **Non-standard CUDA Mathematical functions**
 `float` and `double`
 Mapping and Accuracy (Maximum ULP)

| Meaning | `float` | `double` |
| --- | --- | --- |
| \(\dfrac{x}{y}\) | [fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff), device-only     0 ULP, same as `x / y` | N/A |
| \(10^x\) | [exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff)     2 ULP | [exp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45exp10d)     1 ULP |
| \(\sqrt{x^2 + y^2 + z^2}\) | [norm3df(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm3dffff), device-only     3 ULP | [norm3d(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm3dddd), device-only     2 ULP |
| \(\sqrt{x^2 + y^2 + z^2 + t^2}\) | [norm4df(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm4dfffff), device-only     3 ULP | [norm4d(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm4ddddd), device-only     2 ULP |
| \(\sqrt{\sum\_{i=0}^{\mathrm{dim}-1} p\_i^{2}}\) | [normf(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45normfiPKf), device-only     An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off | [norm(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44normiPKd), device-only     An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off |
| \(\dfrac{1}{\sqrt{x}}\) | [rsqrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rsqrtff)     2 ULP | [rsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rsqrtd)     1 ULP |
| \(\dfrac{1}{\sqrt[3]{x}}\) | [rcbrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rcbrtff)     1 ULP | [rcbrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rcbrtd)     1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2}}\) | [rhypotf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47rhypotfff), device-only     2 ULP | [rhypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46rhypotdd), device-only     1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2 + z^2}}\) | [rnorm3df(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm3dffff), device-only     2 ULP | [rnorm3d(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm3dddd), device-only     1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2 + z^2 + t^2}}\) | [rnorm4df(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm4dfffff), device-only     2 ULP | [rnorm4d(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm4ddddd), device-only     1 ULP |
| \(\dfrac{1}{\sqrt{\sum\_{i=0}^{\mathrm{dim}-1} p\_i^{2}}}\) | [rnormf(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rnormfiPKf), device-only     An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off | [rnorm(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rnormiPKd), device-only     An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off |
| \(\cos(\pi x)\) | [cospif(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46cospiff)     1 ULP | [cospi(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45cospid)     2 ULP |
| \(\sin(\pi x)\) | [sinpif(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46sinpiff)     1 ULP | [sinpi(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45sinpid)     2 ULP |
| \(\sin(\pi x), \cos(\pi x)\) | [sincospif(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49sincospiffPfPf)     1 ULP | [sincospi(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48sincospidPdPd)     2 ULP |
| \(\Phi(x)\) | [normcdff(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48normcdfff)     5 ULP | [normcdf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47normcdfd)     5 ULP |
| \(\Phi^{-1}(x)\) | [normcdfinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv411normcdfinvff)     5 ULP | [normcdfinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv410normcdfinvd)     8 ULP |
| \(\mathrm{erfc}^{-1}(x)\) | [erfcinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48erfcinvff)     4 ULP | [erfcinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47erfcinvd)     6 ULP |
| \(e^{x^2}\mathrm{erfc}(x)\) | [erfcxf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46erfcxff)     4 ULP | [erfcx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45erfcxd)     4 ULP |
| \(\mathrm{erf}^{-1}(x)\) | [erfinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47erfinvff)     2 ULP | [erfinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46erfinvd)     5 ULP |
| \(I\_0(x)\) | [cyl\_bessel\_i0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i0ff), device-only     6 ULP | [cyl\_bessel\_i0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i0d), device-only     6 ULP |
| \(I\_1(x)\) | [cyl\_bessel\_i1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i1ff), device-only     6 ULP | [cyl\_bessel\_i1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i1d), device-only     6 ULP |
| \(J\_0(x)\) | [j0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j0ff)     ≤ 9 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 2.2 \cdot 10^{-6}\), otherwise | [j0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j0d)     ≤ 7 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 5 \cdot 10^{-12}\), otherwise |
| \(J\_1(x)\) | [j1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j1ff)     ≤ 9 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 2.2 \cdot 10^{-6}\), otherwise | [j1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j1d)     ≤ 7 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 5 \cdot 10^{-12}\), otherwise |
| \(J\_n(x)\) | [jnf(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43jnfif)     For \(n = 128\), the maximum absolute error \(= 2.2 \cdot 10^{-6}\) | [jn(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42jnid)     For \(n = 128\), the maximum absolute error \(= 5 \cdot 10^{-12}\) |
| \(Y\_0(x)\) | [y0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y0ff)     ≤ 9 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 2.2 \cdot 10^{-6}\), otherwise | [y0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y0d)     ≤ 7 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 5 \cdot 10^{-12}\), otherwise |
| \(Y\_1(x)\) | [y1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y1ff)     ≤ 9 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 2.2 \cdot 10^{-6}\), otherwise | [y1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y1d)     ≤ 7 ULP for \(|x| < 8\)   ≤ the maximum absolute error \(= 5 \cdot 10^{-12}\), otherwise |
| \(Y\_n(x)\) | [ynf(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43ynfif)     ≤ \(\lceil 2 + 2.5n \rceil\) for \(|x| < n\)   ≤ the maximum absolute error \(= 2.2 \cdot 10^{-6}\), otherwise | [yn(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42ynid)     For \(|x| > 1.5n\), the maximum absolute error \(= 5 \cdot 10^{-12}\) |

Non-standard CUDA Mathematical functions for `__half`, `__nv_bfloat16`, and `__float128/_Float128` are only available in device code.

Table 56 **Non-standard CUDA Mathematical functions**
 `__nv_bfloat16`, `__half`, `__float128/_Float128`
 Mapping and Accuracy (Maximum ULP)

| Meaning | `__nv_bfloat16` | `__half` | `__float128/_Float128` |
| --- | --- | --- | --- |
| \(\dfrac{1}{x}\) | [hrcp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hrcpK13__nv_bfloat16)     0 ULP | [hrcp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hrcpK6__half)     0 ULP | N/A |
| \(10^x\) | [hexp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hexp10K13__nv_bfloat16)     0 ULP | [hexp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hexp10K6__half)     0 ULP | [\_\_nv\_fp128\_exp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_exp10g)     1 ULP |
| \(\dfrac{1}{\sqrt{x}}\) | [hrsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hrsqrtK13__nv_bfloat16)     0 ULP | [hrsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hrsqrtK6__half)     0 ULP | N/A |
| \(\tanh(x)\) (approximate) | [htanh\_approx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv412htanh_approxK13__nv_bfloat16)     1 ULP | [htanh\_approx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv412htanh_approxK6__half)     1 ULP | N/A |

## 5.5.9. Intrinsic Functions

Intrinsic mathematical functions are faster and less accurate versions of their corresponding [CUDA C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).

* They have the same name prefixed with `__`, such as `__sinf(x)`.
* They are only available in device code.
* They are faster because they map to fewer native instructions.
* The flag `--use_fast_math` automatically translates the corresponding [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) into intrinsic functions. See the [–use\_fast\_math Effect](#use-fast-math) section for the full list of affected functions.

### 5.5.9.1. Basic Intrinsic Functions

A subset of mathematical intrinsic functions allow specifying the rounding mode:

* Functions suffixed with `_rn` operate using the *round to nearest even* rounding mode.
* Functions suffixed with `_rz` operate using the *round towards zero* rounding mode.
* Functions suffixed with `_ru` operate using the *round up* (toward positive infinity) rounding mode.
* Functions suffixed with `_rd` operate using the *round down* (toward negative infinity) rounding mode.

The `__fadd_[rn,rz,ru,rd]()`, `__dadd_[rn,rz,ru,rd]()`, `__fmul_[rn,rz,ru,rd]()`, and `__dmul_[rn,rz,ru,rd]()` functions map to addition and multiplication operations that the compiler never merges into the `FFMA` or `DFMA` instructions. In contrast, additions and multiplications generated from the `*` and `+` operators are often combined into `FFMA` or `DFMA`.

The following table lists the single- and double-precision floating-point intrinsic functions. All of them have a maximum ULP error of 0 and are IEEE-compliant.

Table 57 Single- and Double-Precision Floating-Point Intrinsic Functions

| Meaning | `float` | `double` |
| --- | --- | --- |
| \(x + y\) | [\_\_fadd\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fadd_rnff) | [\_\_dadd\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dadd_rndd) |
| \(x - y\) | [\_\_fsub\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fsub_rnff) | [\_\_dsub\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dsub_rndd) |
| \(x \cdot y\) | [\_\_fmul\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmul_rnff) | [\_\_dmul\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dmul_rndd) |
| \(x \cdot y + z\) | [\_\_fmaf\_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff) | [\_\_fma\_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv48__fma_rnddd) |
| \(\dfrac{x}{y}\) | [\_\_fdiv\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fdiv_rnff) | [\_\_ddiv\_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__ddiv_rndd) |
| \(\dfrac{1}{x}\) | [\_\_frcp\_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__frcp_rnf) | [\_\_drcp\_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__drcp_rnd) |
| \(\sqrt{x}\) | [\_\_fsqrt\_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fsqrt_rnf) | [\_\_dsqrt\_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv410__dsqrt_rnd) |

### 5.5.9.2. Single-Precision-Only Intrinsic Functions

The following table lists the single-precision floating-point intrinsic functions with their maximum ULP error.

* The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the *round-to-nearest ties-to-even* rounding mode.
* The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.

Table 58 **Single-Precision Only Floating-Point Intrinsic Functions**
 Mapping and Accuracy (Maximum ULP)

| Function | Meaning | Maximum ULP Error |
| --- | --- | --- |
| [\_\_fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff) | \(\dfrac{x}{y}\) | \(2\) for \(|y| \in [2^{-126}, 2^{126}]\) |
| [\_\_frsqrt\_rn(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv411__frsqrt_rnf) | \(\dfrac{1}{\sqrt{x}}\) | 0 ULP |
| [\_\_expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff) | \(e^x\) | \(2 + \lfloor |1.173 \cdot x| \rfloor\) |
| [\_\_exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff) | \(10^x\) | \(2 + \lfloor |2.97 \cdot x| \rfloor\) |
| [\_\_powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff) | \(x^y\) | Derived from `exp2f(y * __log2f(x))` |
| [\_\_logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff) | \(\ln(x)\) | ≤ \(2^{-21.41}\) abs error for \(x \in [0.5, 2]\)   ≤ 3 ULP, otherwise |
| [\_\_log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff) | \(\log\_2(x)\) | ≤ \(2^{-22}\) abs error for \(x \in [0.5, 2]\)   ≤ 2 ULP, otherwise |
| [\_\_log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff) | \(\log\_{10}(x)\) | ≤ \(2^{-24}\) abs error for \(x \in [0.5, 2]\)   ≤ 3 ULP, otherwise |
| [\_\_sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff) | \(\sin(x)\) | ≤ \(2^{-21.41}\) abs error for \(x \in [-\pi, \pi]\)   ≤ larger otherwise |
| [\_\_cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff) | \(\cos(x)\) | ≤ \(2^{-21.41}\) abs error for \(x \in [-\pi, \pi]\)   ≤ larger otherwise |
| [\_\_sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf) | \(\sin(x), \cos(x)\) | Component-wise, the same as `__sinf(x)` and `__cosf(x)` |
| [\_\_tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff) | \(\tan(x)\) | Derived from `__sinf(x) * (1 / __cosf(x))` |
| [\_\_tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff) | \(\tanh(x)\) | ≤ Max relative error: \(2^{-11}\)   ≤ Subnormal results are not flushed to zero even under `-ftz=true` compiler flag. |

### 5.5.9.3. `--use_fast_math` Effect

The `nvcc` compiler flag `--use_fast_math` translates a subset of [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) called in device code into their intrinsic counterpart. Note that the [CUDA C++ Standard Library functions](#mathematical-functions-appendix-cxx-standard-functions) are also affected by this flag.
See the [Intrinsic Functions](#mathematical-functions-appendix-intrinsic-functions) section for more details on the implications of using intrinsic functions instead of CUDA Math API functions.

> A more robust approach is to selectively replace mathematical function calls with intrinsic versions only where the performance gains justify it and where the changed properties, such as reduced accuracy and different special-case handling, are acceptable.

Table 59 Functions Directly Affected by `--use_fast_math`

| Device Function | Intrinsic Function |
| --- | --- |
| [x/y, fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff) | [\_\_fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff) |
| [sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff) | [\_\_sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff) |
| [cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff) | [\_\_cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff) |
| [tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff) | [\_\_tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff) |
| [sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47sincosffPfPf) | [\_\_sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf) |
| [logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff) | [\_\_logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff) |
| [log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff) | [\_\_log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff) |
| [log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff) | [\_\_log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff) |
| [expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff) | [\_\_expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff) |
| [exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff) | [\_\_exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff) |
| [powf(x,y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff) | [\_\_powf(x,y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff) |
| [tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff) | [\_\_tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff) |

## 5.5.10. References

1. [IEEE 754-2019 Standard for Floating-Point Arithmetic](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229).
2. Jean-Michel Muller. [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf). INRIA/LIP research report, 2005.
3. Nathan Whitehead, Alex Fit-Florea. [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus). Nvidia Report, 2011.
4. David Goldberg. [What every computer scientist should know about floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). ACM Computing Surveys, March 1991.
5. David Monniaux. [The pitfalls of verifying floating-point computations](https://dl.acm.org/doi/pdf/10.1145/1353445.1353446). ACM Transactions on Programming Languages and Systems, May 2008.
6. Peter Dinda, Conor Hetland. [Do Developers Understand IEEE Floating Point?](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425212). IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2018.
