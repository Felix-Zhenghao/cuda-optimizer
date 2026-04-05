#### 9.7.17.1. [Stack Manipulation Instructions: `stacksave`](#stack-manipulation-instructions-stacksave)

`stacksave`

Save the value of stack pointer into a register.

Syntax

```
stacksave.type  d;

.type = { .u32, .u64 };
```

Description

Copies the current value of stack pointer into the destination register `d`. Pointer returned by
`stacksave` can be used in a subsequent `stackrestore` instruction to restore the stack
pointer. If `d` is modified prior to use in `stackrestore` instruction, it may corrupt data in
the stack.

Destination operand `d` has the same type as the instruction type.

Semantics

```
d = stackptr;
```

PTX ISA Notes

Introduced in PTX ISA version 7.3.

Target ISA Notes

`stacksave` requires `sm_52` or higher.

Examples

```
.reg .u32 rd;
stacksave.u32 rd;

.reg .u64 rd1;
stacksave.u64 rd1;
```

---

#### 9.7.17.2. [Stack Manipulation Instructions: `stackrestore`](#stack-manipulation-instructions-stackrestore)

`stackrestore`

Update the stack pointer with a new value.

Syntax

```
stackrestore.type  a;

.type = { .u32, .u64 };
```

Description

Sets the current stack pointer to source register `a`.

When `stackrestore` is used with operand `a` written by a prior `stacksave` instruction, it
will effectively restore the state of stack as it was before `stacksave` was executed. Note that
if `stackrestore` is used with an arbitrary value of `a`, it may cause corruption of stack
pointer. This implies that the correct use of this feature requires that `stackrestore.type a` is
used after `stacksave.type a` without redefining the value of `a` between them.

Operand `a` has the same type as the instruction type.

Semantics

```
stackptr = a;
```

PTX ISA Notes

Introduced in PTX ISA version 7.3.

Target ISA Notes

`stackrestore` requires `sm_52` or higher.

Examples

```
.reg .u32 ra;
stacksave.u32 ra;
// Code that may modify stack pointer
...
stackrestore.u32 ra;
```
