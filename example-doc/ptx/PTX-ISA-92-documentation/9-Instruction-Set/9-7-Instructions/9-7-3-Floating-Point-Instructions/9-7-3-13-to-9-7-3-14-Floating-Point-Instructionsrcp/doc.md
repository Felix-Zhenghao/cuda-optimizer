#### 9.7.3.13. [Floating Point Instructions: `rcp`](#floating-point-instructions-rcp)

`rcp`

Take the reciprocal of a value.

Syntax

```
rcp.approx{.ftz}.f32  d, a;  // fast, approximate reciprocal
rcp.rnd{.ftz}.f32     d, a;  // IEEE 754 compliant rounding
rcp.rnd.f64           d, a;  // IEEE 754 compliant rounding

.rnd = { .rn, .rz, .rm, .rp };
```

Description

Compute `1/a`, store result in `d`.

Semantics

```
d = 1 / a;
```

Notes

Fast, approximate single-precision reciprocal:

`rcp.approx.f32` implements a fast approximation to reciprocal.
The maximum ulp error is 1 across the full range of inputs.

| Input | Result |
| --- | --- |
| -Inf | -0.0 |
| -0.0 | -Inf |
| +0.0 | +Inf |
| +Inf | +0.0 |
| NaN | NaN |

Reciprocal with IEEE 754 compliant rounding:

Rounding modifiers (no default):

`.rn`
:   mantissa LSB rounds to nearest even

`.rz`
:   mantissa LSB rounds towards zero

`.rm`
:   mantissa LSB rounds towards negative infinity

`.rp`
:   mantissa LSB rounds towards positive infinity

Subnormal numbers:

`sm_20+`
:   By default, subnormal numbers are supported.

    `rcp.ftz.f32` flushes subnormal inputs and results to sign-preserving zero.

`sm_1x`
:   `rcp.f64` supports subnormal numbers.

    `rcp.f32` flushes subnormal inputs and results to sign-preserving zero.

PTX ISA Notes

`rcp.f32` and `rcp.f64` introduced in PTX ISA version 1.0. `rcp.rn.f64` and explicit modifiers
`.approx` and `.ftz` were introduced in PTX ISA version 1.4. General rounding modifiers were
added in PTX ISA version 2.0.

For PTX ISA version 1.4 and later, one of `.approx` or `.rnd` is required.

For PTX ISA versions 1.0 through 1.3, `rcp.f32` defaults to `rcp.approx.ftz.f32`, and
`rcp.f64` defaults to `rcp.rn.f64`.

Target ISA Notes

`rcp.approx.f32` supported on all target architectures.

`rcp.rnd.f32` requires `sm_20` or higher.

`rcp.rn.f64` requires `sm_13` or higher, or `.target map_f64_to_f32.`

`rcp.{rz,rm,rp}.f64` requires `sm_20` or higher.

Examples

```
rcp.approx.ftz.f32  ri,r;
rcp.rn.ftz.f32      xi,x;
rcp.rn.f64          xi,x;
```

---

#### 9.7.3.14. [Floating Point Instructions: `rcp.approx.ftz.f64`](#floating-point-instructions-rcp-approx-ftz-f64)

`rcp.approx.ftz.f64`

Compute a fast, gross approximation to the reciprocal of a value.

Syntax

```
rcp.approx.ftz.f64  d, a;
```

Description

Compute a fast, gross approximation to the reciprocal as follows:

1. extract the most-significant 32 bits of `.f64` operand `a` in 1.11.20 IEEE floating-point
   format (i.e., ignore the least-significant 32 bits of `a`),
2. compute an approximate `.f64` reciprocal of this value using the most-significant 20 bits of
   the mantissa of operand `a`,
3. place the resulting 32-bits in 1.11.20 IEEE floating-point format in the most-significant 32-bits
   of destination `d`,and
4. zero the least significant 32 mantissa bits of `.f64` destination `d`.

Semantics

```
tmp = a[63:32]; // upper word of a, 1.11.20 format
d[63:32] = 1.0 / tmp;
d[31:0] = 0x00000000;
```

Notes

`rcp.approx.ftz.f64` implements a fast, gross approximation to reciprocal.

| Input a[63:32] | Result d[63:32] |
| --- | --- |
| -Inf | -0.0 |
| -subnormal | -Inf |
| -0.0 | -Inf |
| +0.0 | +Inf |
| +subnormal | +Inf |
| +Inf | +0.0 |
| NaN | NaN |

Input `NaN`s map to a canonical `NaN` with encoding `0x7fffffff00000000`.

Subnormal inputs and results are flushed to sign-preserving zero.

PTX ISA Notes

`rcp.approx.ftz.f64` introduced in PTX ISA version 2.1.

Target ISA Notes

`rcp.approx.ftz.f64` requires `sm_20` or higher.

Examples

```
rcp.approx.ftz.f64  xi,x;
```
