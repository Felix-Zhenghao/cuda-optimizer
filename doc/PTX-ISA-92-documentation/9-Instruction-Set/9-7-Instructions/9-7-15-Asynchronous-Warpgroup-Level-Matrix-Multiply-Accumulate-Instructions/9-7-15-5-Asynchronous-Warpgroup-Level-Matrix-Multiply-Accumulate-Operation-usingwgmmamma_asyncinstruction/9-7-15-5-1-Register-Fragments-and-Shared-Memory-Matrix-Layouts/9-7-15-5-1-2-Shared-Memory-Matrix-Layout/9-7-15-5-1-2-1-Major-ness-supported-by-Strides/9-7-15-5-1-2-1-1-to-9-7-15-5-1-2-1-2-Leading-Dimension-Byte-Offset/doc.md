###### 9.7.15.5.1.2.1.1. [Leading Dimension Byte Offset](#asynchronous-warpgroup-level-leading-dimension-byte-offset)

The leading dimension byte offset is defined differently for transposed and non-transposed
matrices. The leading byte offset is defined as follows for matrices whose element types are
normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | * No-Swizzling: the offset from the first column to the second columns   of the 8x2 tile in the 128-bit element type normalized matrix. * Swizzled layouts: not used, assumed to be 1. |
| MN-Major | * Interleave: offset from the first 8 columns to the next 8 columns. * Swizzled layouts: offset from the first (swizzle-byte-size/16) rows   to the next (swizzle-byte-size/16) rows. |

---

###### 9.7.15.5.1.2.1.2. [Stride Dimension Byte Offset](#asynchronous-warpgroup-level-stride-dimension-byte-offset)

The stride dimension byte offset is defined differently for transposed and non-transposed
matrices. The stride dimension byte offset is defined as follows for matrices whose element
types are normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | The offset from the first 8 rows to the next 8 rows. |
| MN-Major | * Interleave: offset from the first row to the next row. * Swizzled layout: offset from the first 8 columns to the next 8   columns |
