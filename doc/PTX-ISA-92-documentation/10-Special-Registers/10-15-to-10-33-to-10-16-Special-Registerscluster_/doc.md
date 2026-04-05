## 10.15. [Special Registers: `%cluster_nctaid`](#special-registers-cluster-nctaid)

`%cluster_nctaid`

Number of CTA identifiers per cluster.

Syntax (predefined)

```
.sreg .v4 .u32 %cluster_nctaid;
.sreg .u32 %cluster_nctaid.x, %cluster_nctaid.y, %cluster_nctaid.z;
```

Description

A predefined, read-only special register initialized with the number of CTAs in a cluster in each
dimension.

The `%cluster_nctaid` special register contains a 3D grid shape vector that holds the cluster
dimensions in terms of CTAs. The fourth element is unused and always returns zero.

Refer to the *Cuda Programming Guide* for details on the maximum values of
`%cluster_nctaid.{x,y,z}`.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r<2>;
.reg .v4 .b32 %rx;

mov.u32     %r0, %cluster_nctaid.x;
mov.u32     %r1, %cluster_nctaid.z;
mov.v4.u32  %rx, %cluster_nctaid;
```

---

## 10.16. [Special Registers: `%cluster_ctarank`](#special-registers-cluster-ctarank)

`%cluster_ctarank`

CTA identifier in a cluster across all dimensions.

Syntax (predefined)

```
.sreg .u32 %cluster_ctarank;
```

Description

A predefined, read-only special register initialized with the CTA rank within a cluster across all
dimensions.

It is guaranteed that:

```
0  <=  %cluster_ctarank <  %cluster_nctarank
```

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r;

mov.u32  %r, %cluster_ctarank;
```

---

## 10.17. [Special Registers: `%cluster_nctarank`](#special-registers-cluster-nctarank)

`%cluster_nctarank`

Number of CTA identifiers in a cluster across all dimensions.

Syntax (predefined)

```
.sreg .u32 %cluster_nctarank;
```

Description

A predefined, read-only special register initialized with the nunber of CTAs within a cluster across
all dimensions.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r;

mov.u32  %r, %cluster_nctarank;
```

---

## 10.18. [Special Registers: `%lanemask_eq`](#special-registers-lanemask-eq)

`%lanemask_eq`

32-bit mask with bit set in position equal to the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_eq;
```

Description

A predefined, read-only special register initialized with a 32-bit mask with a bit set in the
position equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_eq` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_eq;
```

---

## 10.19. [Special Registers: `%lanemask_le`](#special-registers-lanemask-le)

`%lanemask_le`

32-bit mask with bits set in positions less than or equal to the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_le;
```

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
less than or equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_le` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_le
```

---

## 10.20. [Special Registers: `%lanemask_lt`](#special-registers-lanemask-lt)

`%lanemask_lt`

32-bit mask with bits set in positions less than the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_lt;
```

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
less than the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_lt` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_lt;
```

---

## 10.21. [Special Registers: `%lanemask_ge`](#special-registers-lanemask-ge)

`%lanemask_ge`

32-bit mask with bits set in positions greater than or equal to the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_ge;
```

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
greater than or equal to the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_ge` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_ge;
```

---

## 10.22. [Special Registers: `%lanemask_gt`](#special-registers-lanemask-gt)

`%lanemask_gt`

32-bit mask with bits set in positions greater than the thread’s lane number in the warp.

Syntax (predefined)

```
.sreg .u32 %lanemask_gt;
```

Description

A predefined, read-only special register initialized with a 32-bit mask with bits set in positions
greater than the thread’s lane number in the warp.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%lanemask_gt` requires `sm_20` or higher.

Examples

```
mov.u32     %r, %lanemask_gt;
```

---

## 10.23. [Special Registers: `%clock`, `%clock_hi`](#special-registers-clock)

`%clock`, `%clock_hi`

`%clock`
:   A predefined, read-only 32-bit unsigned cycle counter.

`%clock_hi`
:   The upper 32-bits of `%clock64` special register.

Syntax (predefined)

```
.sreg .u32 %clock;
.sreg .u32 %clock_hi;
```

Description

Special register `%clock` and `%clock_hi` are unsigned 32-bit read-only cycle counters that wrap
silently.

PTX ISA Notes

`%clock` introduced in PTX ISA version 1.0.

`%clock_hi` introduced in PTX ISA version 5.0.

Target ISA Notes

`%clock` supported on all target architectures.

`%clock_hi` requires `sm_20` or higher.

Examples

```
mov.u32 r1,%clock;
mov.u32 r2, %clock_hi;
```

---

## 10.24. [Special Registers: `%clock64`](#special-registers-clock64)

`%clock64`

A predefined, read-only 64-bit unsigned cycle counter.

Syntax (predefined)

```
.sreg .u64 %clock64;
```

Description

Special register `%clock64` is an unsigned 64-bit read-only cycle counter that wraps silently.

Notes

The lower 32-bits of `%clock64` are identical to `%clock`.

The upper 32-bits of `%clock64` are identical to `%clock_hi`.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%clock64` requires `sm_20` or higher.

Examples

```
mov.u64  r1,%clock64;
```

---

## 10.25. [Special Registers: `%pm0` … `%pm7`](#special-registers-pm0-pm7)

`%pm0` … `%pm7`

Performance monitoring counters.

Syntax (predefined)

```
.sreg .u32 %pm<8>;
```

Description

Special registers `%pm0` … `%pm7` are unsigned 32-bit read-only performance monitor counters. Their
behavior is currently undefined.

PTX ISA Notes

`%pm0` … `%pm3` introduced in PTX ISA version 1.3.

`%pm4` … `%pm7` introduced in PTX ISA version 3.0.

Target ISA Notes

`%pm0` … `%pm3` supported on all target architectures.

`%pm4` … `%pm7` require `sm_20` or higher.

Examples

```
mov.u32  r1,%pm0;
mov.u32  r1,%pm7;
```

---

## 10.26. [Special Registers: `%pm0_64` … `%pm7_64`](#special-registers-pm0-64-pm7-64)

`%pm0_64` … `%pm7_64`

64 bit Performance monitoring counters.

Syntax (predefined)

```
.sreg .u64 %pm0_64;
.sreg .u64 %pm1_64;
.sreg .u64 %pm2_64;
.sreg .u64 %pm3_64;
.sreg .u64 %pm4_64;
.sreg .u64 %pm5_64;
.sreg .u64 %pm6_64;
.sreg .u64 %pm7_64;
```

Description

Special registers `%pm0_64` … `%pm7_64` are unsigned 64-bit read-only performance monitor
counters. Their behavior is currently undefined.

Notes

The lower 32bits of `%pm0_64` … `%pm7_64` are identical to `%pm0` … `%pm7`.

PTX ISA Notes

`%pm0_64` … `%pm7_64` introduced in PTX ISA version 4.0.

Target ISA Notes

`%pm0_64` … `%pm7_64` require `sm_50` or higher.

Examples

```
mov.u32  r1,%pm0_64;
mov.u32  r1,%pm7_64;
```

---

## 10.27. [Special Registers: `%envreg<32>`](#special-registers-envreg-32)

`%envreg<32>`

Driver-defined read-only registers.

Syntax (predefined)

```
.sreg .b32 %envreg<32>;
```

Description

A set of 32 pre-defined read-only registers used to capture execution environment of PTX program
outside of PTX virtual machine. These registers are initialized by the driver prior to kernel launch
and can contain cta-wide or grid-wide values.

Precise semantics of these registers is defined in the driver documentation.

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.b32      %r1,%envreg0;  // move envreg0 to %r1
```

---

## 10.28. [Special Registers: `%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`](#special-registers-globaltimer)

`%globaltimer`, `%globaltimer_lo`, `%globaltimer_hi`

`%globaltimer`
:   A predefined, 64-bit global nanosecond timer.

`%globaltimer_lo`
:   The lower 32-bits of %globaltimer.

`%globaltimer_hi`
:   The upper 32-bits of %globaltimer.

Syntax (predefined)

```
.sreg .u64 %globaltimer;
.sreg .u32 %globaltimer_lo, %globaltimer_hi;
```

Description

Special registers intended for use by NVIDIA tools. The behavior is target-specific and may change
or be removed in future GPUs. When JIT-compiled to other targets, the value of these registers is
unspecified.

PTX ISA Notes

Introduced in PTX ISA version 3.1.

Target ISA Notes

Requires target `sm_30` or higher.

Examples

```
mov.u64  r1,%globaltimer;
```

---

## 10.29. [Special Registers: `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset_<2>`](#special-registers-reserved-smem)

`%reserved_smem_offset_begin`, `%reserved_smem_offset_end`, `%reserved_smem_offset_cap`, `%reserved_smem_offset_<2>`

`%reserved_smem_offset_begin`
:   Start of the reserved shared memory region.

`%reserved_smem_offset_end`
:   End of the reserved shared memory region.

`%reserved_smem_offset_cap`
:   Total size of the reserved shared memory region.

`%reserved_smem_offset_<2>`
:   Offsets in the reserved shared memory region.

Syntax (predefined)

```
.sreg .b32 %reserved_smem_offset_begin;
.sreg .b32 %reserved_smem_offset_end;
.sreg .b32 %reserved_smem_offset_cap;
.sreg .b32 %reserved_smem_offset_<2>;
```

Description

These are predefined, read-only special registers containing information about the shared memory
region which is reserved for the NVIDIA system software use. This region of shared memory is not
available to users, and accessing this region from user code results in undefined behavior. Refer to
*CUDA Programming Guide* for details.

PTX ISA Notes

Introduced in PTX ISA version 7.6.

Target ISA Notes

Require `sm_80` or higher.

Examples

```
.reg .b32 %reg_begin, %reg_end, %reg_cap, %reg_offset0, %reg_offset1;

mov.b32 %reg_begin,   %reserved_smem_offset_begin;
mov.b32 %reg_end,     %reserved_smem_offset_end;
mov.b32 %reg_cap,     %reserved_smem_offset_cap;
mov.b32 %reg_offset0, %reserved_smem_offset_0;
mov.b32 %reg_offset1, %reserved_smem_offset_1;
```

---

## 10.30. [Special Registers: `%total_smem_size`](#special-registers-total-smem-size)

`%total_smem_size`

Total size of shared memory used by a CTA of a kernel.

Syntax (predefined)

```
.sreg .u32 %total_smem_size;
```

Description

A predefined, read-only special register initialized with total size of shared memory allocated
(statically and dynamically, excluding the shared memory reserved for the NVIDIA system software
use) for the CTA of a kernel at launch time.

Size is returned in multiples of shared memory allocation unit size supported by target
architecture.

Allocation unit values are as follows:

| Target architecture | Shared memory allocation unit size |
| --- | --- |
| `sm_2x` | 128 bytes |
| `sm_3x`, `sm_5x`, `sm_6x`, `sm_7x` | 256 bytes |
| `sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 128 bytes |

PTX ISA Notes

Introduced in PTX ISA version 4.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples

```
mov.u32  %r, %total_smem_size;
```

---

## 10.31. [Special Registers: `%aggr_smem_size`](#special-registers-aggr-smem-size)

`%aggr_smem_size`

Total size of shared memory used by a CTA of a kernel.

Syntax (predefined)

```
.sreg .u32 %aggr_smem_size;
```

Description

A predefined, read-only special register initialized with total aggregated size of shared memory
consisting of the size of user shared memory allocated (statically and dynamically) at launch time
and the size of shared memory region which is reserved for the NVIDIA system software use.

PTX ISA Notes

Introduced in PTX ISA version 8.1.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mov.u32  %r, %aggr_smem_size;
```

---

## 10.32. [Special Registers: `%dynamic_smem_size`](#special-registers-dynamic-smem-size)

`%dynamic_smem_size`

Size of shared memory allocated dynamically at kernel launch.

Syntax (predefined)

```
.sreg .u32 %dynamic_smem_size;
```

Description

Size of shared memory allocated dynamically at kernel launch.

A predefined, read-only special register initialized with size of shared memory allocated dynamically for the CTA of a kernel at launch time.

PTX ISA Notes

Introduced in PTX ISA version 4.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples

```
mov.u32  %r, %dynamic_smem_size;
```

---

## 10.33. [Special Registers: `%current_graph_exec`](#special-registers-current-graph-exec)

`%current_graph_exec`

An Identifier for currently executing CUDA device graph.

Syntax (predefined)

```
.sreg .u64 %current_graph_exec;
```

Description

A predefined, read-only special register initialized with the identifier referring to the CUDA
device graph being currently executed. This register is 0 if the executing kernel is not part of a
CUDA device graph.

Refer to the *CUDA Programming Guide* for more details on CUDA device graphs.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_50` or higher.

Examples

```
mov.u64  r1, %current_graph_exec;
```
