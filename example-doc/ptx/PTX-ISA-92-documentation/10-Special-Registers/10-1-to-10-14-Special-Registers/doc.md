## 10.1. [Special Registers: `%tid`](#special-registers-tid)

`%tid`

Thread identifier within a CTA.

Syntax (predefined)

```
.sreg .v4 .u32 %tid;                  // thread id vector
.sreg .u32 %tid.x, %tid.y, %tid.z;    // thread id components
```

Description

A predefined, read-only, per-thread special register initialized with the thread identifier within
the CTA. The `%tid` special register contains a 1D, 2D, or 3D vector to match the CTA shape; the
`%tid` value in unused dimensions is `0`. The fourth element is unused and always returns
zero. The number of threads in each dimension are specified by the predefined special register
`%ntid`.

Every thread in the CTA has a unique `%tid`.

`%tid` component values range from `0` through `%ntid-1` in each CTA dimension.

`%tid.y == %tid.z == 0` in 1D CTAs. `%tid.z == 0` in 2D CTAs.

It is guaranteed that:

```
0  <=  %tid.x <  %ntid.x
0  <=  %tid.y <  %ntid.y
0  <=  %tid.z <  %ntid.z
```

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit
`mov` and `cvt` instructions may be used to read the lower 16-bits of each component of
`%tid`.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32      %r1,%tid.x;  // move tid.x to %rh

// legacy code accessing 16-bit components of %tid
mov.u16      %rh,%tid.x;
cvt.u32.u16  %r2,%tid.z;  // zero-extend tid.z to %r2
```

---

## 10.2. [Special Registers: `%ntid`](#special-registers-ntid)

`%ntid`

Number of thread IDs per CTA.

Syntax (predefined)

```
.sreg .v4 .u32 %ntid;                   // CTA shape vector
.sreg .u32 %ntid.x, %ntid.y, %ntid.z;   // CTA dimensions
```

Description

A predefined, read-only special register initialized with the number of thread ids in each CTA
dimension. The `%ntid` special register contains a 3D CTA shape vector that holds the CTA
dimensions. CTA dimensions are non-zero; the fourth element is unused and always returns zero. The
total number of threads in a CTA is `(%ntid.x * %ntid.y * %ntid.z)`.

```
%ntid.y == %ntid.z == 1 in 1D CTAs.
%ntid.z ==1 in 2D CTAs.
```

Maximum values of %ntid.{x,y,z} are as follows:

| .target architecture | %ntid.x | %ntid.y | %ntid.z |
| --- | --- | --- | --- |
| `sm_1x` | 512 | 512 | 64 |
| `sm_20`, `sm_3x`, `sm_5x`, `sm_6x`, `sm_7x`, `sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 1024 | 1024 | 64 |

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit
`mov` and `cvt` instructions may be used to read the lower 16-bits of each component of
`%ntid`.

Target ISA Notes

Supported on all target architectures.

Examples

```
// compute unified thread id for 2D CTA
mov.u32  %r0,%tid.x;
mov.u32  %h1,%tid.y;
mov.u32  %h2,%ntid.x;
mad.u32  %r0,%h1,%h2,%r0;

mov.u16  %rh,%ntid.x;      // legacy code
```

---

## 10.3. [Special Registers: `%laneid`](#special-registers-laneid)

`%laneid`

Lane Identifier.

Syntax (predefined)

```
.sreg .u32 %laneid;
```

Description

A predefined, read-only special register that returns the thread's lane within the warp. The lane
identifier ranges from zero to `WARP_SZ-1`.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r, %laneid;
```

---

## 10.4. [Special Registers: `%warpid`](#special-registers-warpid)

`%warpid`

Warp identifier.

Syntax (predefined)

```
.sreg .u32 %warpid;
```

Description

A predefined, read-only special register that returns the thread's warp identifier. The warp
identifier provides a unique warp number within a CTA but not across CTAs within a grid. The warp
identifier will be the same for all threads within a single warp.

Note that `%warpid` returns the location of a thread at the moment when read, but
its value may change during execution, e.g., due to rescheduling of threads following
preemption. For this reason, `%ctaid` and `%tid` should be used to compute a virtual warp index
if such a value is needed in kernel code; `%warpid` is intended mainly to enable profiling and
diagnostic code to sample and log information such as work place mapping and load distribution.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r, %warpid;
```

---

## 10.5. [Special Registers: `%nwarpid`](#special-registers-nwarpid)

`%nwarpid`

Number of warp identifiers.

Syntax (predefined)

```
.sreg .u32 %nwarpid;
```

Description

A predefined, read-only special register that returns the maximum number of warp identifiers.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%nwarpid` requires `sm_20` or higher.

Examples

```
mov.u32  %r, %nwarpid;
```

---

## 10.6. [Special Registers: `%ctaid`](#special-registers-ctaid)

`%ctaid`

CTA identifier within a grid.

Syntax (predefined)

```
.sreg .v4 .u32 %ctaid;                      // CTA id vector
.sreg .u32 %ctaid.x, %ctaid.y, %ctaid.z;    // CTA id components
```

Description

A predefined, read-only special register initialized with the CTA identifier within the CTA
grid. The `%ctaid` special register contains a 1D, 2D, or 3D vector, depending on the shape and
rank of the CTA grid. The fourth element is unused and always returns zero.

It is guaranteed that:

```
0  <=  %ctaid.x <  %nctaid.x
0  <=  %ctaid.y <  %nctaid.y
0  <=  %ctaid.z <  %nctaid.z
```

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit
`mov` and `cvt` instructions may be used to read the lower 16-bits of each component of
`%ctaid`.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r0,%ctaid.x;
mov.u16  %rh,%ctaid.y;   // legacy code
```

---

## 10.7. [Special Registers: `%nctaid`](#special-registers-nctaid)

`%nctaid`

Number of CTA ids per grid.

Syntax (predefined)

```
.sreg .v4 .u32 %nctaid                      // Grid shape vector
.sreg .u32 %nctaid.x,%nctaid.y,%nctaid.z;   // Grid dimensions
```

Description

A predefined, read-only special register initialized with the number of CTAs in each grid
dimension. The `%nctaid` special register contains a 3D grid shape vector, with each element
having a value of at least `1`. The fourth element is unused and always returns zero.

Maximum values of %nctaid.{x,y,z} are as follows:

| .target architecture | %nctaid.x | %nctaid.y | %nctaid.z |
| --- | --- | --- | --- |
| `sm_1x`, `sm_20` | 65535 | 65535 | 65535 |
| `sm_3x`, `sm_5x`, `sm_6x`, `sm_7x`, `sm_8x`, `sm_9x`, `sm_10x`, `sm_12x` | 231 -1 | 65535 | 65535 |

PTX ISA Notes

Introduced in PTX ISA version 1.0 with type `.v4.u16`.

Redefined as type `.v4.u32` in PTX ISA version 2.0. For compatibility with legacy PTX code, 16-bit
`mov` and `cvt` instructions may be used to read the lower 16-bits of each component of
`%nctaid`.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r0,%nctaid.x;
mov.u16  %rh,%nctaid.x;     // legacy code
```

---

## 10.8. [Special Registers: `%smid`](#special-registers-smid)

`%smid`

SM identifier.

Syntax (predefined)

```
.sreg .u32 %smid;
```

Description

A predefined, read-only special register that returns the processor (SM) identifier on which a
particular thread is executing. The SM identifier ranges from `0` to `%nsmid-1`. The SM
identifier numbering is not guaranteed to be contiguous.

Notes

Note that `%smid` returns the location of a thread at the moment when read, but
its value may change during execution, e.g. due to rescheduling of threads following
preemption. `%smid` is intended mainly to enable profiling and diagnostic code to sample and log
information such as work place mapping and load distribution.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u32  %r, %smid;
```

---

## 10.9. [Special Registers: `%nsmid`](#special-registers-nsmid)

`%nsmid`

Number of SM identifiers.

Syntax (predefined)

```
.sreg .u32 %nsmid;
```

Description

A predefined, read-only special register that returns the maximum number of SM identifiers. The SM
identifier numbering is not guaranteed to be contiguous, so `%nsmid` may be larger than the
physical number of SMs in the device.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`%nsmid` requires `sm_20` or higher.

Examples

```
mov.u32  %r, %nsmid;
```

---

## 10.10. [Special Registers: `%gridid`](#special-registers-gridid)

`%gridid`

Grid identifier.

Syntax (predefined)

```
.sreg .u64 %gridid;
```

Description

A predefined, read-only special register initialized with the per-grid temporal grid identifier. The
`%gridid` is used by debuggers to distinguish CTAs and clusters within concurrent (small) grids.

During execution, repeated launches of programs may occur, where each launch starts a
grid-of-CTAs. This variable provides the temporal grid launch number for this context.

For `sm_1x` targets, `%gridid` is limited to the range [0..216-1]. For `sm_20`,
`%gridid` is limited to the range [0..232-1]. `sm_30` supports the entire 64-bit range.

PTX ISA Notes

Introduced in PTX ISA version 1.0 as type `.u16`.

Redefined as type `.u32` in PTX ISA version 1.3.

Redefined as type `.u64` in PTX ISA version 3.0.

For compatibility with legacy PTX code, 16-bit and 32-bit `mov` and `cvt` instructions may be
used to read the lower 16-bits or 32-bits of each component of `%gridid`.

Target ISA Notes

Supported on all target architectures.

Examples

```
mov.u64  %s, %gridid;  // 64-bit read of %gridid
mov.u32  %r, %gridid;  // legacy code with 32-bit %gridid
```

---

## 10.11. [Special Registers: `%is_explicit_cluster`](#special-registers-is-explicit-cluster)

`%is_explicit_cluster`

Checks if user has explicitly specified cluster launch.

Syntax (predefined)

```
.sreg .pred %is_explicit_cluster;
```

Description

A predefined, read-only special register initialized with the predicate value of whether the cluster
launch is explicitly specified by user.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .pred p;

mov.pred  p, %is_explicit_cluster;
```

---

## 10.12. [Special Registers: `%clusterid`](#special-registers-clusterid)

`%clusterid`

Cluster identifier within a grid.

Syntax (predefined)

```
.sreg .v4 .u32 %clusterid;
.sreg .u32 %clusterid.x, %clusterid.y, %clusterid.z;
```

Description

A predefined, read-only special register initialized with the cluster identifier in a grid in each
dimension. Each cluster in a grid has a unique identifier.

The `%clusterid` special register contains a 1D, 2D, or 3D vector, depending upon the shape and
rank of the cluster. The fourth element is unused and always returns zero.

It is guaranteed that:

```
0  <=  %clusterid.x <  %nclusterid.x
0  <=  %clusterid.y <  %nclusterid.y
0  <=  %clusterid.z <  %nclusterid.z
```

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r<2>;
.reg .v4 .b32 %rx;

mov.u32     %r0, %clusterid.x;
mov.u32     %r1, %clusterid.z;
mov.v4.u32  %rx, %clusterid;
```

---

## 10.13. [Special Registers: `%nclusterid`](#special-registers-nclusterid)

`%nclusterid`

Number of cluster identifiers per grid.

Syntax (predefined)

```
.sreg .v4 .u32 %nclusterid;
.sreg .u32 %nclusterid.x, %nclusterid.y, %nclusterid.z;
```

Description

A predefined, read-only special register initialized with the number of clusters in each grid
dimension.

The `%nclusterid` special register contains a 3D grid shape vector that holds the grid dimensions
in terms of clusters. The fourth element is unused and always returns zero.

Refer to the *Cuda Programming Guide* for details on the maximum values of `%nclusterid.{x,y,z}`.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r<2>;
.reg .v4 .b32 %rx;

mov.u32     %r0, %nclusterid.x;
mov.u32     %r1, %nclusterid.z;
mov.v4.u32  %rx, %nclusterid;
```

---

## 10.14. [Special Registers: `%cluster_ctaid`](#special-registers-cluster-ctaid)

`%cluster_ctaid`

CTA identifier within a cluster.

Syntax (predefined)

```
.sreg .v4 .u32 %cluster_ctaid;
.sreg .u32 %cluster_ctaid.x, %cluster_ctaid.y, %cluster_ctaid.z;
```

Description

A predefined, read-only special register initialized with the CTA identifier in a cluster in each
dimension. Each CTA in a cluster has a unique CTA identifier.

The `%cluster_ctaid` special register contains a 1D, 2D, or 3D vector, depending upon the shape of
the cluster. The fourth element is unused and always returns zero.

It is guaranteed that:

```
0  <=  %cluster_ctaid.x <  %cluster_nctaid.x
0  <=  %cluster_ctaid.y <  %cluster_nctaid.y
0  <=  %cluster_ctaid.z <  %cluster_nctaid.z
```

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
.reg .b32 %r<2>;
.reg .v4 .b32 %rx;

mov.u32     %r0, %cluster_ctaid.x;
mov.u32     %r1, %cluster_ctaid.z;
mov.v4.u32  %rx, %cluster_ctaid;
```
