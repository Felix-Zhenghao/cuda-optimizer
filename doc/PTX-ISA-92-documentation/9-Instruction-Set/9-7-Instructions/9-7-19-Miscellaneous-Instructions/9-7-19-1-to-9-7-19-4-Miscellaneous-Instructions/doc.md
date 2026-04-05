#### 9.7.19.1. [Miscellaneous Instructions: `brkpt`](#miscellaneous-instructions-brkpt)

`brkpt`

Breakpoint.

Syntax

```
brkpt;
```

Description

Suspends execution.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

`brkpt` requires `sm_11` or higher.

Examples

```
    brkpt;
@p  brkpt;
```

---

#### 9.7.19.2. [Miscellaneous Instructions: `nanosleep`](#miscellaneous-instructions-nanosleep)

`nanosleep`

Suspend the thread for an approximate delay given in nanoseconds.

Syntax

```
nanosleep.u32 t;
```

Description

Suspends the thread for a sleep duration approximately close to the delay `t`, specified in
nanoseconds. `t` may be a register or an immediate value.

The sleep duration is approximated, but guaranteed to be in the interval `[0, 2*t]`. The maximum
sleep duration is 1 millisecond. The implementation may reduce the sleep duration for individual
threads within a warp such that all sleeping threads in the warp wake up together.

PTX ISA Notes

`nanosleep` introduced in PTX ISA 6.3.

Target ISA Notes

`nanosleep` requires `sm_70` or higher.

Examples

```
.reg .b32 r;
.reg .pred p;

nanosleep.u32 r;
nanosleep.u32 42;
@p nanosleep.u32 r;
```

---

#### 9.7.19.3. [Miscellaneous Instructions: `pmevent`](#miscellaneous-instructions-pmevent)

`pmevent`

Trigger one or more Performance Monitor events.

Syntax

```
pmevent       a;    // trigger a single performance monitor event
pmevent.mask  a;    // trigger one or more performance monitor events
```

Description

Triggers one or more of a fixed number of performance monitor events, with event index or mask
specified by immediate operand `a`.

`pmevent` (without modifier `.mask`) triggers a single performance monitor event indexed by
immediate operand `a`, in the range `0..15`.

`pmevent.mask` triggers one or more of the performance monitor events. Each bit in the 16-bit
immediate operand `a` controls an event.

Programmatic performance moniter events may be combined with other hardware events using Boolean
functions to increment one of the four performance counters. The relationship between events and
counters is programmed via API calls from the host.

Notes

Currently, there are sixteen performance monitor events, numbered 0 through 15.

PTX ISA Notes

`pmevent` introduced in PTX ISA version 1.4.

`pmevent.mask` introduced in PTX ISA version 3.0.

Target ISA Notes

pmevent supported on all target architectures.

`pmevent.mask` requires `sm_20` or higher.

Examples

```
    pmevent      1;
@p  pmevent      7;
@q  pmevent.mask 0xff;
```

---

#### 9.7.19.4. [Miscellaneous Instructions: `trap`](#miscellaneous-instructions-trap)

`trap`

Perform trap operation.

Syntax

```
trap;
```

Description

Abort execution and generate an interrupt to the host CPU.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
    trap;
@p  trap;
```
