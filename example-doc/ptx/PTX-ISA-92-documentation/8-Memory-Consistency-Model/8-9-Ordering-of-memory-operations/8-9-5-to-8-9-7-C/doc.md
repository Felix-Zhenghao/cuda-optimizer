### 8.9.5. [Causality Order](#causality-order)

*Causality order* captures how memory operations become visible across threads through synchronizing
operations. The axiom "Causality" uses this order to constrain the set of write operations from
which a read operation may read a value.

Relations in the *causality order* primarily consist of relations in *Base causality order*1 , which is a transitive order, determined at runtime.

Base causality order

An operation X precedes an operation Y in *base causality order* if:

1. X precedes Y in *program order*, or
2. X *synchronizes* with Y, or
3. For some operation Z,

   1. X precedes Z in *program order* and Z precedes Y in *base causality order*, or
   2. X precedes Z in *base causality order* and Z precedes Y in *program order*, or
   3. X precedes Z in *base causality order* and Z precedes Y in *base causality order*.

Proxy-preserved base causality order

A memory operation X precedes a memory operation Y in *proxy-preserved base causality order* if X
precedes Y in *base causality order*, and:

1. X and Y are performed to the same address, using the *generic proxy*, or
2. X and Y are performed to the same address, using the same *proxy*, and by the same thread block,
   or
3. X and Y are aliases and there is an alias *proxy fence* along the base causality path from X
   to Y.

Causality order

*Causality order* combines *base causality order* with some non-transitive relations as follows:

An operation X precedes an operation Y in *causality order* if:

1. X precedes Y in *proxy-preserved base causality order*, or
2. For some operation Z, X precedes Z in observation order, and Z precedes Y in *proxy-preserved
   base causality order*.

1 The transitivity of *base causality order* accounts for the "cumulativity" of synchronizing
operations.

---

### 8.9.6. [Coherence Order](#coherence-order)

There exists a partial transitive order that relates *overlapping* write operations, determined at
runtime, called the *coherence order*1. Two *overlapping* write operations are related in
*coherence order* if they are *morally strong* or if they are related in *causality order*. Two
*overlapping* writes are unrelated in *coherence order* if they are in a *data-race*, which gives
rise to the partial nature of *coherence order*.

1 *Coherence order* cannot be observed directly since it consists entirely of write
operations. It may be observed indirectly by its use in constraining the set of candidate
writes that a read operation may read from.

---

### 8.9.7. [Communication Order](#communication-order)

The *communication order* is a non-transitive order, determined at runtime, that relates write
operations to other *overlapping* memory operations.

1. A write W precedes an *overlapping* read R in *communication order* if R returns the value of any
   byte that was written by W.
2. A write W precedes a write W' in *communication order* if W precedes W' in *coherence order*.
3. A read R precedes an *overlapping* write W in *communication order* if, for any byte accessed by
   both R and W, R returns the value written by a write W' that precedes W in *coherence order*.

*Communication order* captures the visibility of memory operations --- when a memory operation X1
precedes a memory operation X2 in *communication order*, X1 is said to be visible to X2.
