#### 9.7.9.16. [Data Movement and Conversion Instructions: `applypriority`](#data-movement-and-conversion-instructions-applypriority)

`applypriority`

Apply the cache eviction priority to the specified address in the specified cache level.

Syntax

```
applypriority{.global}.level::eviction_priority  [a], size;

.level::eviction_priority = { .L2::evict_normal };
```

Description

The `applypriority` instruction applies the cache eviction priority specified by the
`.level::eviction_priority` qualifier to the address range `[a..a+size)` in the specified cache
level.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the specified address does not fall within the address window of `.global` state space
then the behavior is undefined.

The operand `size` is an integer constant that specifies the amount of data, in bytes, in the
specified cache level on which the priority is to be applied. The only supported value for the
`size` operand is 128.

Supported addressing modes for operand `a` are described in [Addresses as Operands](#addresses-as-operands).
`a` must be aligned to 128 bytes.

PTX ISA Notes

Introduced in PTX ISA version 7.4.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
applypriority.global.L2::evict_normal [ptr], 128;
```

---

#### 9.7.9.17. [Data Movement and Conversion Instructions: `discard`](#data-movement-and-conversion-instructions-discard)

`discard`

Discard the data at the specified address range and cache level.

Syntax

```
discard{.global}.level  [a], size;

.level = { .L2 };
```

Description

Semantically, this behaves like a weak write of an *unstable indeterminate value*:
reads of memory locations with *unstable indeterminate values* may return different
bit patterns each time until the memory is overwritten.
This operation *hints* to the implementation that data in the specified cache `.level`
can be destructively discarded without writing it back to memory.

The operand `size` is an integer constant that specifies the length in bytes of the
address range `[a, a + size)` to write *unstable indeterminate values* into.
The only supported value for the `size` operand is `128`.

If no state space is specified then [Generic Addressing](#generic-addressing) is used.
If the specified address does not fall within the address window of `.global` state space
then the behavior is undefined.

Supported addressing modes for address operand `a` are described in [Addresses as Operands](#addresses-as-operands).
`a` must be aligned to 128 bytes.

PTX ISA Notes

Introduced in PTX ISA version 7.4.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
discard.global.L2 [ptr], 128;
ld.weak.u32 r0, [ptr];
ld.weak.u32 r1, [ptr];
// The values in r0 and r1 may differ!
```
