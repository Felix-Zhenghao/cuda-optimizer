## 8.5. [Scope](#scope)

Each *strong* operation must specify a *scope*, which is the set of threads that may interact
directly with that operation and establish any of the relations described in the memory consistency
model. There are four scopes:

Table 21 Scopes

| Scope | Description |
| --- | --- |
| `.cta` | The set of all threads executing in the same CTA as the current thread. |
| `.cluster` | The set of all threads executing in the same cluster as the current thread. |
| `.gpu` | The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device. |
| `.sys` | The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself. |

Note that the warp is not a *scope*; the CTA is the smallest collection of threads that qualifies as
a *scope* in the memory consistency model.

---

## 8.6. [Proxies](#proxies)

A *memory proxy*, or a *proxy* is an abstract label applied to a method of memory access. When two
memory operations use distinct methods of memory access, they are said to be different *proxies*.

Memory operations as defined in [Operation types](#operation-types) use *generic*
method of memory access, i.e. a *generic proxy*. Other operations such as textures and surfaces all
use distinct methods of memory access, also distinct from the *generic* method.

A *proxy fence* is required to synchronize memory operations across different *proxies*. Although
virtual aliases use the *generic* method of memory access, since using distinct virtual addresses
behaves as if using different *proxies*, they require a *proxy fence* to establish memory ordering.

---

### 8.7.1. [Conflict and Data-races](#conflict-and-data-races)

Two *overlapping* memory operations are said to *conflict* when at least one of them is a *write*.

Two *conflicting* memory operations are said to be in a *data-race* if they are not related in
*causality order* and they are not *morally strong*.

---

### 8.7.2. [Limitations on Mixed-size Data-races](#mixed-size-limitations)

A *data-race* between operations that *overlap* completely is called a *uniform-size data-race*,
while a *data-race* between operations that *overlap* partially is called a *mixed-size data-race*.

The axioms in the memory consistency model do not apply if a PTX program contains one or more
*mixed-size data-races*. But these axioms are sufficient to describe the behavior of a PTX program
with only *uniform-size data-races*.

Atomicity of mixed-size RMW operations

In any program with or without *mixed-size data-races*, the following property holds for every pair
of *overlapping atomic* operations A1 and A2 such that each specifies a *scope* that includes the
other: Either the *read-modify-write* operation specified by A1 is performed completely before A2 is
initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2
overlap partially or completely.
