##### 9.7.16.6.1. [Asynchronous Operations](#tcgen05-memory-consistency-model-async-operations)

The tcgen05 family of instructions are divided into 2 categories:

1. Asynchronous instructions:

   These `tcgen05` operations are not inherently ordered with respect to
   other `tcgen05` operations in the same thread (unless pipelined as mentioned below).
2. Synchronous instructions:

   These `tcgen05` operations are inherently ordered with respect to other `tcgen05`
   operations in the same order.

   The Tensor Memory allocation related instructions that access shared memory maintain
   same-address ordering with respect to non-`tcgen05` instructions.

The following table lists the category of each of the `tcgen05` instruction:

| tcgen05.\* operation | Category |
| --- | --- |
| `.alloc` | Synchronous  instructions |
| `.dealloc` |
| `.relinquish_alloc_permit` |
| `.fence::*` |
| `.wait::*` |
| `.commit` |
| `.mma` | Asynchronous  instructions |
| `.cp` |
| `.shift` |
| `.ld` |
| `.st` |

---

##### 9.7.16.6.3. [Specialized Inter-thread Synchronization for tcgen05 instructions](#tcgen05-memory-consistency-model-inter-thread-sync)

The `tcgen05` instructions support a specialized inter-thread synchronization which are
optimized for `tcgen05` family of instructions. The standard memory consistency model
synchronization mechanisms also apply to the `tcgen05` family of instructions.

The [TensorCore 5th Generation Specialized Synchronization Operations](#tcgen05-special-sync-operations) section contains the specialized inter-thread
synchronization for tcgen05 instructions.

The `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` composes
with execution ordering instructions, like morally strong `ld`/`st`/`atom` instructions,
`mbarrier` instruction, `barrier` instructions and so on, to establish an ordering between
the `tcgen05` operations across threads. The asynchronous `tcgen05` instructions that are
ordered across threads also form a `tcgen05` pipeline.

An asynchronous `tcgen05` operation prior to a `tcgen05.fence::before_thread_sync` is ordered
before all subsequent `tcgen05` and the execution ordering operations.

An asynchronous `tcgen05` operation subsequent to a `tcgen05.fence::after_thread_sync` is
ordered after all the prior `tcgen05` and the execution ordering operations.

---

###### 9.7.16.6.2.1.1. [mbarrier based completion mechanism](#tcgen05-memory-consistency-model-mbarrier-completion)

Completion of the following instruction's asynchronous operations is observed
through the mbarrier based waiting mechanism:

1. `tcgen05.mma`
2. `tcgen05.cp`
3. `tcgen05.shift`

`tcgen05.commit` is used to track the completion of the above asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses mbarrier
based completion mechanism:

* `tcgen05.mma.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.cp.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.shift.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)

---

###### 9.7.16.6.2.1.2. [`tcgen05.wait` instruction based completion mechanism](#tcgen05-memory-consistency-model-wait-completion)

Completion of the following instruction's asynchronous operations is observed through
`tcgen05.wait` based waiting mechanism:

1. `tcgen05.ld`
2. `tcgen05.st`

`tcgen05.wait::ld` and `tcgen05.wait::st` is used to track the completion of the
`tcgen05.ld` and `tcgen05.st` asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses
`tcgen05.wait` based completion mechanism:

* `tcgen05.ld` -> `tcgen05.wait::ld`
* `tcgen05.st` -> `tcgen05.wait::st`
