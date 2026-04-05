#### 9.7.3.9. [Floating Point Instructions: `abs`](#floating-point-instructions-abs)

`abs`

Absolute value.

Syntax

```
abs{.ftz}.f32  d, a;
abs.f64        d, a;
```

Description

Take the absolute value of `a` and store the result in `d`.

Semantics

```
d = |a|;
```

Notes

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `abs.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   `abs.f64` supports subnormal numbers.

    `abs.f32` flushes subnormal inputs and results to sign-preserving zero.

For `abs.f32`, `NaN` input yields unspecified `NaN`. For `abs.f64`, `NaN` input is passed
through unchanged. Future implementations may comply with the IEEE 754 standard by preserving
payload and modifying only the sign bit.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

`abs.f32` supported on all target architectures.

`abs.f64` requires `sm_13` or higher.

Examples

```
abs.ftz.f32  x,f0;
```

---

#### 9.7.3.10. [Floating Point Instructions: `neg`](#floating-point-instructions-neg)

`neg`

Arithmetic negate.

Syntax

```
neg{.ftz}.f32  d, a;
neg.f64        d, a;
```

Description

Negate the sign of `a` and store the result in `d`.

Semantics

```
d = -a;
```

Notes

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `neg.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   `neg.f64` supports subnormal numbers.

    `neg.f32` flushes subnormal inputs and results to sign-preserving zero.

`NaN` inputs yield an unspecified `NaN`. Future implementations may comply with the IEEE 754
standard by preserving payload and modifying only the sign bit.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

`neg.f32` supported on all target architectures.

`neg.f64` requires `sm_13` or higher.

Examples

```
neg.ftz.f32  x,f0;
```
