#### 9.7.3.16. [Floating Point Instructions: `rsqrt`](#floating-point-instructions-rsqrt)

`rsqrt`

Take the reciprocal of the square root of a value.

Syntax

```
rsqrt.approx{.ftz}.f32  d, a;
rsqrt.approx.f64        d, a;
```

Description

Compute `1/sqrt(a)` and store the result in `d`.

Semantics

```
d = 1/sqrt(a);
```

Notes

`rsqrt.approx` implements an approximation to the reciprocal square root.

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -normal | NaN |
| -0.0 | -Inf |
| +0.0 | +Inf |
| +Inf | +0.0 |
| NaN | NaN |

The maximum relative error for `rsqrt.f32` over the entire positive
finite floating-point range is 2-22.9.

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `rsqrt.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   `rsqrt.f64` supports subnormal numbers.

    `rsqrt.f32` flushes subnormal inputs and results to sign-preserving zero.

Note that `rsqrt.approx.f64` is emulated in software and are relatively slow.

PTX ISA Notes

`rsqrt.f32` and `rsqrt.f64` were introduced in PTX ISA version 1.0. Explicit modifiers
`.approx` and `.ftz` were introduced in PTX ISA version 1.4.

For PTX ISA version 1.4 and later, the `.approx` modifier is required.

For PTX ISA versions 1.0 through 1.3, `rsqrt.f32` defaults to `rsqrt.approx.ftz.f32`, and
`rsqrt.f64` defaults to `rsqrt.approx.f64`.

Target ISA Notes

`rsqrt.f32` supported on all target architectures.

`rsqrt.f64` requires `sm_13` or higher.

Examples

```
rsqrt.approx.ftz.f32  isr, x;
rsqrt.approx.f64      ISR, X;
```

---

#### 9.7.3.17. [Floating Point Instructions: `rsqrt.approx.ftz.f64`](#floating-point-instructions-rsqrt-approx-ftz-f64)

`rsqrt.approx.ftz.f64`

Compute an approximation of the square root reciprocal of a value.

Syntax

```
rsqrt.approx.ftz.f64 d, a;
```

Description

Compute a double-precision (`.f64`) approximation of the square root reciprocal of a value. The
least significant 32 bits of the double-precision (`.f64`) destination `d` are all zeros.

Semantics

```
tmp = a[63:32]; // upper word of a, 1.11.20 format
d[63:32] = 1.0 / sqrt(tmp);
d[31:0] = 0x00000000;
```

Notes

`rsqrt.approx.ftz.f64` implements a fast approximation of the square root reciprocal of a value.

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -subnormal | -Inf |
| -0.0 | -Inf |
| +0.0 | +Inf |
| +subnormal | +Inf |
| +Inf | +0.0 |
| NaN | NaN |

Input `NaN`s map to a canonical `NaN` with encoding `0x7fffffff00000000`.

Subnormal inputs and results are flushed to sign-preserving zero.

PTX ISA Notes

`rsqrt.approx.ftz.f64` introduced in PTX ISA version 4.0.

Target ISA Notes

`rsqrt.approx.ftz.f64` requires `sm_20` or higher.

Examples

```
rsqrt.approx.ftz.f64 xi,x;
```

---

#### 9.7.3.18. [Floating Point Instructions: `sin`](#floating-point-instructions-sin)

`sin`

Find the sine of a value.

Syntax

```
sin.approx{.ftz}.f32  d, a;
```

Description

Find the sine of the angle `a` (in radians).

Semantics

```
d = sin(a);
```

Notes

`sin.approx.f32` implements a fast approximation to sine.

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -0.0 | -0.0 |
| +0.0 | +0.0 |
| +Inf | NaN |
| NaN | NaN |

The maximum absolute error over input range is as follows:

|  |  |  |
| --- | --- | --- |
| Range | [-2pi .. 2pi] | [-100pi .. +100pi] |
| Error | 2-20.5 | 2-14.7 |

Outside of the range [-100pi .. +100pi], only best effort
is provided. There are no defined error guarantees.

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `sin.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   Subnormal inputs and results to sign-preserving zero.

PTX ISA Notes

`sin.f32` introduced in PTX ISA version 1.0. Explicit modifiers `.approx` and `.ftz`
introduced in PTX ISA version 1.4.

For PTX ISA version 1.4 and later, the .approx modifier is required.

For PTX ISA versions 1.0 through 1.3, `sin.f32` defaults to `sin.approx.ftz.f32`.

Target ISA Notes

Supported on all target architectures.

Examples

```
sin.approx.ftz.f32  sa, a;
```

---

#### 9.7.3.19. [Floating Point Instructions: `cos`](#floating-point-instructions-cos)

`cos`

Find the cosine of a value.

Syntax

```
cos.approx{.ftz}.f32  d, a;
```

Description

Find the cosine of the angle `a` (in radians).

Semantics

```
d = cos(a);
```

Notes

`cos.approx.f32` implements a fast approximation to cosine.

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -0.0 | +1.0 |
| +0.0 | +1.0 |
| +Inf | NaN |
| NaN | NaN |

The maximum absolute error over input range is as follows:

|  |  |  |
| --- | --- | --- |
| Range | [-2pi .. 2pi] | [-100pi .. +100pi] |
| Error | 2-20.5 | 2-14.7 |

Outside of the range [-100pi .. +100pi], only best effort
is provided. There are no defined error guarantees.

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `cos.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   Subnormal inputs and results to sign-preserving zero.

PTX ISA Notes

`cos.f32` introduced in PTX ISA version 1.0. Explicit modifiers `.approx` and `.ftz`
introduced in PTX ISA version 1.4.

For PTX ISA version 1.4 and later, the `.approx` modifier is required.

For PTX ISA versions 1.0 through 1.3, `cos.f32` defaults to `cos.approx.ftz.f32`.

Target ISA Notes

Supported on all target architectures.

Examples

```
cos.approx.ftz.f32  ca, a;
```

---

#### 9.7.3.20. [Floating Point Instructions: `lg2`](#floating-point-instructions-lg2)

`lg2`

Find the base-2 logarithm of a value.

Syntax

```
lg2.approx{.ftz}.f32  d, a;
```

Description

Determine the log2 of `a`.

Semantics

```
d = log(a) / log(2);
```

Notes

`lg2.approx.f32` implements a fast approximation to log2(a).

| Input | Result |
| --- | --- |
| -Inf | NaN |
| -normal | NaN |
| -0.0 | -Inf |
| +0.0 | -Inf |
| +Inf | +Inf |
| NaN | NaN |

The maximum absolute error is 2-22 when the input operand is in the
range (0.5, 2). For positive finite inputs outside of this interval, maximum
relative error is 2-22.

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `lg2.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   Subnormal inputs and results to sign-preserving zero.

PTX ISA Notes

`lg2.f32` introduced in PTX ISA version 1.0. Explicit modifiers `.approx` and `.ftz`
introduced in PTX ISA version 1.4.

For PTX ISA version 1.4 and later, the `.approx` modifier is required.

For PTX ISA versions 1.0 through 1.3, `lg2.f32` defaults to `lg2.approx.ftz.f32`.

Target ISA Notes

Supported on all target architectures.

Examples

```
lg2.approx.ftz.f32  la, a;
```

---

#### 9.7.3.21. [Floating Point Instructions: `ex2`](#floating-point-instructions-ex2)

`ex2`

Find the base-2 exponential of a value.

Syntax

```
ex2.approx{.ftz}.f32  d, a;
```

Description

Raise 2 to the power `a`.

Semantics

```
d = 2 ^ a;
```

Notes

`ex2.approx.f32` implements a fast approximation to 2a.

| Input | Result |
| --- | --- |
| -Inf | +0.0 |
| -0.0 | +1.0 |
| +0.0 | +1.0 |
| +Inf | +Inf |
| NaN | NaN |

The maximum ulp error is 2 ulp from correctly rounded result across the
full range of inputs.

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `ex2.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   Subnormal inputs and results to sign-preserving zero.

PTX ISA Notes

`ex2.f32` introduced in PTX ISA version 1.0. Explicit modifiers `.approx` and `.ftz`
introduced in PTX ISA version 1.4.

For PTX ISA version 1.4 and later, the `.approx` modifier is required.

For PTX ISA versions 1.0 through 1.3, `ex2.f32` defaults to `ex2.approx.ftz.f32`.

Target ISA Notes

Supported on all target architectures.

Examples

```
ex2.approx.ftz.f32  xa, a;
```

---

#### 9.7.3.22. [Floating Point Instructions: `tanh`](#floating-point-instructions-tanh)

`tanh`

Find the hyperbolic tangent of a value (in radians)

Syntax

```
tanh.approx.f32 d, a;
```

Description

Take hyperbolic tangent value of `a`.

The operands `d` and `a` are of type `.f32`.

Semantics

```
d = tanh(a);
```

Notes

`tanh.approx.f32` implements a fast approximation to FP32 hyperbolic-tangent.

Results of `tanh` for various corner-case inputs are as follows:

| Input | Result |
| --- | --- |
| -Inf | -1.0 |
| -0.0 | -0.0 |
| +0.0 | +0.0 |
| +Inf | 1.0 |
| NaN | NaN |

The maximum relative error over the entire floating point
range is 2-11.
The subnormal numbers are supported.

Note

The subnormal inputs gets passed through to the output since the value of `tanh(x)` for small
values of `x` is approximately the same as `x`.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_75` or higher.

Examples

```
tanh.approx.f32 ta, a;
```
