###### 9.7.9.25.3.2. [Data Movement and Conversion Instructions: `cp.async.commit_group`](#data-movement-and-conversion-instructions-cp-async-commit-group)

`cp.async.commit_group`

Commits all prior initiated but uncommitted `cp.async` instructions into a *cp.async-group*.

Syntax

```
cp.async.commit_group ;
```

Description

`cp.async.commit_group` instruction creates a new *cp.async-group* per thread and batches all
prior `cp.async` instructions initiated by the executing thread but not committed to any
*cp.async-group* into the new *cp.async-group*. If there are no uncommitted `cp.async`
instructions then `cp.async.commit_group` results in an empty *cp.async-group.*

An executing thread can wait for the completion of all `cp.async` operations in a *cp.async-group*
using `cp.async.wait_group`.

There is no memory ordering guarantee provided between any two `cp.async` operations within the
same *cp.async-group*. So two or more `cp.async` operations within a *cp.async-group* copying data
to the same location results in undefined behavior.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example 1:
cp.async.ca.shared.global [shrd], [gbl], 4;
cp.async.commit_group ; // Marks the end of a cp.async group

// Example 2:
cp.async.ca.shared.global [shrd1],   [gbl1],   8;
cp.async.ca.shared.global [shrd1+8], [gbl1+8], 8;
cp.async.commit_group ; // Marks the end of cp.async group 1

cp.async.ca.shared.global [shrd2],    [gbl2],    16;
cp.async.cg.shared.global [shrd2+16], [gbl2+16], 16;
cp.async.commit_group ; // Marks the end of cp.async group 2
```

---

###### 9.7.9.25.3.3. [Data Movement and Conversion Instructions: `cp.async.wait_group` / `cp.async.wait_all`](#data-movement-and-conversion-instructions-cp-async-wait-group)

`cp.async.wait_group`, `cp.async.wait_all`

Wait for completion of prior asynchronous copy operations.

Syntax

```
cp.async.wait_group N;
cp.async.wait_all ;
```

Description

`cp.async.wait_group` instruction will cause executing thread to wait till only `N` or fewer of
the most recent *cp.async-group*s are pending and all the prior *cp.async-group*s committed by
the executing threads are complete. For example, when `N` is 0, the executing thread waits on all
the prior *cp.async-group*s to complete. Operand `N` is an integer constant.

`cp.async.wait_all` is equivalent to :

```
cp.async.commit_group;
cp.async.wait_group 0;
```

An empty *cp.async-group* is considered to be trivially complete.

Writes performed by `cp.async` operations are made visible to the executing thread only after:

1. The completion of `cp.async.wait_all` or
2. The completion of `cp.async.wait_group` on the *cp.async-group* in which the `cp.async`
   belongs to or
3. [mbarrier.test\_wait](#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
   returns `True` on an *mbarrier object* which is tracking the completion of the `cp.async`
   operation.

There is no ordering between two `cp.async` operations that are not synchronized with
`cp.async.wait_all` or `cp.async.wait_group` or [mbarrier objects](#parallel-synchronization-and-communication-instructions-mbarrier).

`cp.async.wait_group` and `cp.async.wait_all` does not provide any ordering and visibility
guarantees for any other memory operation apart from `cp.async`.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example of .wait_all:
cp.async.ca.shared.global [shrd1], [gbl1], 4;
cp.async.cg.shared.global [shrd2], [gbl2], 16;
cp.async.wait_all;  // waits for all prior cp.async to complete

// Example of .wait_group :
cp.async.ca.shared.global [shrd3], [gbl3], 8;
cp.async.commit_group;  // End of group 1

cp.async.cg.shared.global [shrd4], [gbl4], 16;
cp.async.commit_group;  // End of group 2

cp.async.cg.shared.global [shrd5], [gbl5], 16;
cp.async.commit_group;  // End of group 3

cp.async.wait_group 1;  // waits for group 1 and group 2 to complete
```
