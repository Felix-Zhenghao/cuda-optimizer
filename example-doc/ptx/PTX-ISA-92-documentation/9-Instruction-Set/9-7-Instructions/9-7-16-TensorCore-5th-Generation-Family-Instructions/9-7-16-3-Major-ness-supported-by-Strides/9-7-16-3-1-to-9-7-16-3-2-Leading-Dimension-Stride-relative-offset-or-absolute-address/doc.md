###### 9.7.16.3.1.1. [Relative offset mode](#tcgen05-leading-dimension-byte-offset-relative-offset)

In this mode, the leading dimension stride is specified as a relative byte offset between the
columns as explained in the below table.
The leading dimension stride can either be specified as a relative offset between the columns
or as an absolute byte address of next buffer. The leading dimension stride is defined
differently for transposed and non-transposed matrices. The leading dimension stride is defined
as follows for matrices whose element types are normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | * No-Swizzling: the stride from the first column to the second column   of the 8x2 tile in the 128-bit element type normalized matrix. * Swizzled layouts: not used, assumed to be 1. |
| MN-Major | * Interleave: stride from the first 8 columns to the next 8 columns. * Swizzled layouts: stride from the first (swizzle-byte-size/16) rows   to the next (swizzle-byte-size/16) rows. |

---

###### 9.7.16.3.1.2.1. [Restrictions on the Leading Dimension Absolute Address Stride](#tcgen05-leading-dimension-byte-offset-absolute-address-restriction)

Following are the restrictions on the absolute address stride mode:

1. Only 128B swizzle (with 16B atomicity) is supported.
2. Only K-Major mode is supported. That is, the transpose bits(bits #15 and #16) in
   [Instruction descriptor](#tcgen05-instruction-descriptor) must be 0.
3. The matrix base offset must be 0.

---

##### 9.7.16.3.2. [Stride Dimension Byte Offset](#tcgen05-stride-dimension-byte-offset)

The stride dimension byte offset is defined differently for transposed and non-transposed
matrices. The stride dimension byte offset is defined as follows for matrices whose element
types are normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | The offset from the first 8 rows to the next 8 rows. |
| MN-Major | * Interleave: offset from the first row to the next row. * Swizzled layout: offset from the first 8 columns to the next 8   columns |
