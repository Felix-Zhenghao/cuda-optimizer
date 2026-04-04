##### 9.7.13.15.1. [Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment)

An mbarrier object is an opaque object with the following type and alignment requirements :

| Type | Alignment (bytes) | Memory space |
| --- | --- | --- |
| `.b64` | 8 | `.shared` |

---

##### 9.7.13.15.2. [Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents)

An opaque *mbarrier object* keeps track of the following information :

* Current phase of the *mbarrier object*
* Count of pending arrivals for the current phase of the *mbarrier object*
* Count of expected arrivals for the next phase of the *mbarrier object*
* Count of pending asynchronous memory operations (or transactions) tracked by the current phase of
  the *mbarrier object*. This is also referred to as *tx-count*.

An *mbarrier object* progresses through a sequence of phases where each phase is defined by threads
performing an expected number of
[arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
operations.

The valid range of each of the counts is as shown below:

| Count name | Minimum value | Maximum value |
| --- | --- | --- |
| Expected arrival count | 1 | 220 - 1 |
| Pending arrival count | 0 | 220 - 1 |
| tx-count | -(220 - 1) | 220 - 1 |

---

##### 9.7.13.15.3. [Lifecycle of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-lifecycle)

The *mbarrier object* must be initialized prior to use.

An *mbarrier object* is used to synchronize threads and asynchronous memory operations.

An *mbarrier object* may be used to perform a sequence of such synchronizations.

An *mbarrier object* must be invalidated to repurpose its memory for any purpose,
including repurposing it for another mbarrier object.

---

##### 9.7.13.15.4. [Phase of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase)

The phase of an *mbarrier object* is the number of times the *mbarrier object* has been used to
synchronize threads and [asynchronous](#program-order-async-operations)
operations. In each phase {0, 1, 2, ...}, threads perform in program order :

* [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
  operations to complete the current phase and
* *test\_wait* / *try\_wait* operations to check for the completion of the current phase.

An *mbarrier object* is automatically reinitialized upon completion of the current phase for
immediate use in the next phase. The current phase is incomplete and all prior phases are complete.

For each phase of the mbarrier object, at least one *test\_wait* or *try\_wait* operation must be
performed which returns `True` for `waitComplete` before an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
in the subsequent phase.

---

###### 9.7.13.15.5.1. [expect-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)

The *expect-tx* operation, with an `expectCount` argument, increases the *tx-count* of an
*mbarrier object* by the value specified by `expectCount`. This sets the current phase of the
*mbarrier object* to expect and track the completion of additional asynchronous transactions.

---

###### 9.7.13.15.5.2. [complete-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)

The *complete-tx* operation, with an `completeCount` argument, on an *mbarrier object* consists of the following:

mbarrier signaling
:   Signals the completion of asynchronous transactions that were tracked by the current phase. As a
    result of this, *tx-count* is decremented by `completeCount`.

mbarrier potentially completing the current phase
:   If the current phase has been completed then the mbarrier transitions to the next phase. Refer to
    [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
    for details on phase completion requirements and phase transition process.

---

##### 9.7.13.15.6. [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)

The requirements for completion of the current phase are described below. Upon completion of the
current phase, the phase transitions to the subsequent phase as described below.

Current phase completion requirements
:   An *mbarrier object* completes the current phase when all of the following conditions are met:

    * The count of the pending arrivals has reached zero.
    * The *tx-count* has reached zero.

Phase transition
:   When an *mbarrier* object completes the current phase, the following actions are performed
    atomically:

    * The *mbarrier object* transitions to the next phase.
    * The pending arrival count is reinitialized to the expected arrival count.

---

##### 9.7.13.15.7. [Arrive-on operation on mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)

An *arrive-on* operation, with an optional *count* argument, on an *mbarrier object* consists of the
following 2 steps :

* mbarrier signalling:

  Signals the arrival of the executing thread OR completion of the asynchronous instruction which
  signals the arrive-on operation initiated by the executing thread on the *mbarrier object*. As a
  result of this, the pending arrival count is decremented by *count*. If the *count* argument is
  not specified, then it defaults to 1.
* mbarrier potentially completing the current phase:

  If the current phase has been completed then the mbarrier transitions to the next phase. Refer to
  [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
  for details on phase completion requirements and phase transition process.

---

##### 9.7.13.15.8. [mbarrier support with shared memory](#parallel-synchronization-and-communication-instructions-mbarrier-smem)

The following table summarizes the support of various mbarrier operations on *mbarrier objects*
located at different shared memory locations:

| mbarrier operations | `.shared::cta` | `.shared::cluster` |
| --- | --- | --- |
| `mbarrier.arrive`, `mbarrier.arrive_drop` | Supported | Supported, cannot return result |
| `mbarrier.expect_tx` | Supported | Supported |
| `mbarrier.complete_tx` | Supported | Supported |
| Other mbarrier operations | Supported | Not supported |

---

##### 9.7.13.15.9. [Parallel Synchronization and Communication Instructions: `mbarrier.init`](#parallel-synchronization-and-communication-instructions-mbarrier-init)

`mbarrier.init`

Initialize the *mbarrier object*.

Syntax

```
mbarrier.init{.shared{::cta}}.b64 [addr], count;
```

Description

`mbarrier.init` initializes the *mbarrier object* at the location specified by the address operand
`addr` with the unsigned 32-bit integer `count`.
The value of operand `count` must be in the range
as specified in [Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents).

Initialization of the *mbarrier object* involves :

* Initializing the current phase to 0.
* Initializing the expected arrival count to `count`.
* Initializing the pending arrival count to `count`.
* Initializing the *tx-count* to 0.

The valid range of values for the operand `count` is [1, ..., 220 - 1].
Refer [Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents) for the
valid range of values for the various constituents of the mbarrier.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The behavior of performing an `mbarrier.init` operation on a memory location containing a
valid *mbarrier object* is undefined; invalidate the *mbarrier object* using `mbarrier.inval`
first, before repurposing the memory location for any other purpose, including another *mbarrier object*.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
.shared .b64 shMem, shMem2;
.reg    .b64 addr;
.reg    .b32 %r1;

cvta.shared.u64          addr, shMem2;
mbarrier.init.b64        [addr],   %r1;
bar.cta.sync             0;
// ... other mbarrier operations on addr

mbarrier.init.shared::cta.b64 [shMem], 12;
bar.sync                 0;
// ... other mbarrier operations on shMem
```

---

##### 9.7.13.15.10. [Parallel Synchronization and Communication Instructions: `mbarrier.inval`](#parallel-synchronization-and-communication-instructions-mbarrier-inval)

`mbarrier.inval`

Invalidates the *mbarrier object*.

Syntax

```
mbarrier.inval{.shared{::cta}}.b64 [addr];
```

Description

`mbarrier.inval` invalidates the *mbarrier object* at the location specified by the address
operand `addr`.

An *mbarrier object* must be invalidated before using its memory location for any other purpose.

Performing any *mbarrier* operation except `mbarrier.init` on a memory location that does not
contain a valid *mbarrier object*, results in undefined behaviour.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
.shared .b64 shmem;
.reg    .b64 addr;
.reg    .b32 %r1;
.reg    .pred t0;

// Example 1 :
bar.sync                      0;
@t0 mbarrier.init.b64     [addr], %r1;
// ... other mbarrier operations on addr
bar.sync                      0;
@t0 mbarrier.inval.b64    [addr];

// Example 2 :
bar.cta.sync                  0;
mbarrier.init.shared.b64           [shmem], 12;
// ... other mbarrier operations on shmem
bar.cta.sync                  0;
@t0 mbarrier.inval.shared.b64      [shmem];

// shmem can be reused here for unrelated use :
bar.cta.sync                  0;
st.shared.b64                      [shmem], ...;

// shmem can be re-initialized as mbarrier object :
bar.cta.sync                  0;
@t0 mbarrier.init.shared.b64       [shmem], 24;
// ... other mbarrier operations on shmem
bar.cta.sync                  0;
@t0 mbarrier.inval.shared::cta.b64 [shmem];
```

---

##### 9.7.13.15.11. [Parallel Synchronization and Communication Instructions: `mbarrier.expect_tx`](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)

`mbarrier.expect_tx`

Perfoms
[expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object*.

Syntax

```
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta}, .shared::cluster }
```

Description

A thread executing `mbarrier.expect_tx` performs an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object* at the location specified by the address operand `addr`. The
32-bit unsigned integer operand `txCount` specifies the `expectCount` argument to the
*expect-tx* operation.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` or `.shared::cluster` state space then the behavior is undefined.

Supported addressing modes for operand `addr` are as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](#memory-consistency-model).
The `.relaxed` qualifier does not provide any memory ordering semantics and visibility
guarantees.

The optional `.scope` qualifier indicates the set of threads that directly observe the memory
synchronizing effect of this operation, as described in the [Memory Consistency Model](#memory-consistency-model).

Qualifiers `.sem` and `.scope` must be specified together.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mbarrier.expect_tx.b64                       [addr], 32;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj1], 512;
mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj2], 512;
```

---

##### 9.7.13.15.12. [Parallel Synchronization and Communication Instructions: `mbarrier.complete_tx`](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx)

`mbarrier.complete_tx`

Perfoms
[complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the *mbarrier object*.

Syntax

```
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta}, .shared::cluster }
```

Description

A thread executing `mbarrier.complete_tx` performs a [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the *mbarrier object* at the location specified by the address operand `addr`. The
32-bit unsigned integer operand `txCount` specifies the `completeCount` argument to the
*complete-tx* operation.

`mbarrier.complete_tx` does not involve any asynchronous memory operations and only simulates the
completion of an asynchronous memory operation and its side effect of signaling to the *mbarrier
object*.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` or `.shared::cluster` state space then the behavior is undefined.

Supported addressing modes for operand `addr` are as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](#memory-consistency-model).
The `.relaxed` qualifier does not provide any memory ordering semantics and visibility
guarantees.

The optional `.scope` qualifier indicates the set of threads that directly observe the memory
synchronizing effect of this operation, as described in the [Memory Consistency Model](#memory-consistency-model).

Qualifiers `.sem` and `.scope` must be specified together.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mbarrier.complete_tx.b64             [addr],     32;
mbarrier.complete_tx.shared.b64      [mbarObj1], 512;
mbarrier.complete_tx.relaxed.cta.b64 [addr2],    32;
```
