##### 9.7.16.10.1. [Transpose and Negate operations](#tcgen05-transpose-and-negate-operations)

The matrices `A` and `B` can be transposed by specifying the Tranpose `A` Matrix
and Transpose `B` Matrix bits in the instruction descriptor respectively.

The elements of the matrices `A` and `B` can be negated by specifying the Negate
`A` Matrix and Negate `B` Matrix bits in the instruction descriptor respectively.

The support for Transpose and Negate operation for various MMA-Kind are shown in
[Table 51](#tcgen05-transpose-negate-mma-kind).

Table 51 Transpose and Negate operation for various MMA-Kind

| MMA-Kind | Is Transpose A/B supported | Is Negate A/B supported |
| --- | --- | --- |
| `.kind::tf32` | Yes | Yes |
| `.kind::f16` | Yes | Yes |
| `.kind::f8f6f4` | Yes | Yes |
| `.kind::mxf8f6f4` | Yes | Yes |
| `.kind::i8` | Yes | No |
| `.kind::mxf4` | No | Yes |
| `.kind::mxf4nvf4` | No | Yes |

For `.kind::tf32`, the transpose operations on matrices `A` and `B` are supported
only with 128B swizzling mode with 32B swizzle-atomicity.

For all other MMA-Kinds, the transpose operations on matrices `A` and `B` are not supported
on 128B swizzling mode with 32B swizzle-atomicity.

[Table 52](#tcgen05-kind-shapes-8b-transpose-b) shows the valid combinations of N shape with
`.cta_group` qualifier for 8bit transpose B.

Table 52 Various combinations of N shape with .cta\_group qualifier for 8bit transpose B

| .cta\_group | N shape |
| --- | --- |
| 1 | 16 <= N <= 256, step 16 |
| 2 | 32 <= N <= 256, step 32 |

---

##### 9.7.16.10.2. [Matrix Layout Organization](#tcgen05-matrix-layout-organization)

[Table 53](#tcgen05-matrices-majorness) describes the major-ness used for different matrices.

Table 53 Major-ness for different matrices

| Matrix | Residing in Memory | Default Major-ness |
| --- | --- | --- |
| D | Tensor Memory | Row-Major |
| A | Tensor Memory |
| Shared Memory | Depends on swizzling mode. Refer [Shared Memory Layout and Swizzling](#tcgen05-shared-memory-layout-swizzling) |
| B | Shared Memory |

---

##### 9.7.16.10.3. [Valid Combinations of Type-Size, Major-ness and Swizzling](#tcgen05-matrix-layout-organization-valid-comb-type-size-majorness-swizzle)

Table 54 Valid Combinations of Type-Size, Major-ness and Swizzling

| Type-Size | Major-ness | Matrix | Supported Swizzle |
| --- | --- | --- | --- |
| 4-bit, 6-bit, 8-bit, 16-bit, 32-bit | Row | A | All swizzling modes |
| Column | B |
| 8-bit  16-bit | Column (transpose) | A | All except 128B swizzling with 32B atomicity |
| Row (transpose) | B |
| 32-bit | Column (transpose) | A | Only 128B swizzling with 32B atomicity |
| Row (transpose) | B |
