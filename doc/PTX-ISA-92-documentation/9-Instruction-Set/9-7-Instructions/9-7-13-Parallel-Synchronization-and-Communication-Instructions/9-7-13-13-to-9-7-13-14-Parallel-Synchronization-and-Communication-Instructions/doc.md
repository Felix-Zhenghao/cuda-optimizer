#### 9.7.13.13. [Parallel Synchronization and Communication Instructions: `griddepcontrol`](#parallel-synchronization-and-communication-instructions-griddepcontrol)

`griddepcontrol`

Control execution of dependent grids.

Syntax

```
griddepcontrol.action;

.action   = { .launch_dependents, .wait }
```

Description

The `griddepcontrol` instruction allows the dependent grids and prerequisite grids as defined by
the runtime, to control execution in the following way:

`.launch_dependents` modifier signals that specific dependents the runtime system designated to
react to this instruction can be scheduled as soon as all other CTAs in the grid issue the same
instruction or have completed. The dependent may launch before the completion of the current
grid. There is no guarantee that the dependent will launch before the completion of the current
grid. Repeated invocations of this instruction by threads in the current CTA will have no additional
side effects past that of the first invocation. A *release fence* preceeding a `griddepcontrol.launch_dependents`
in program-order *synchronizes* with the start of a dependent grid if either both grids have the same
[memory synchronization domain](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)
and the fence scope is `gpu`, or if the fence scope is `sys`.

`.wait` modifier causes the executing thread to wait until all prerequisite grids in flight have
completed and all the memory operations from the prerequisite grids are performed and made visible
to the current grid.

Note

If the prerequisite grid is using `griddepcontrol.launch_dependents`, then the dependent grid
must use `griddepcontrol.wait` to ensure correct functional execution.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
griddepcontrol.launch_dependents;
griddepcontrol.wait;
```

---

#### 9.7.13.14. [Parallel Synchronization and Communication Instructions: `elect.sync`](#parallel-synchronization-and-communication-instructions-elect-sync)

`elect.sync`

Elect a leader thread from a set of threads.

Syntax

```
elect.sync d|p, membermask;
```

Description

`elect.sync` elects one predicated active leader thread from among a set of threads specified by
`membermask`. `laneid` of the elected thread is returned in the 32-bit destination operand
`d`. The sink symbol ‘\_’ can be used for destination operand `d`. The predicate destination
`p` is set to `True` for the leader thread, and `False` for all other threads.

Operand `membermask` specifies a 32-bit integer indicating the set of threads from which a leader
is to be elected. The behavior is undefined if the executing thread is not in `membermask`.

Election of a leader thread happens deterministically, i.e. the same leader thread is elected for
the same `membermask` every time.

The mandatory `.sync` qualifier indicates that `elect` causes the executing thread to wait until
all threads in the `membermask` execute the `elect` instruction before resuming execution.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
elect.sync    %r0|%p0, 0xffffffff;
```
