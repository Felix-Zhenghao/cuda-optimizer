#### 9.7.9.23. [Data Movement and Conversion Instructions: `mapa`](#data-movement-and-conversion-instructions-mapa)

`mapa`

Map the address of the shared variable in the target CTA.

Syntax

```
mapa{.space}.type          d, a, b;

// Maps shared memory address in register a into CTA b.
mapa.shared::cluster.type  d, a, b;

// Maps shared memory variable into CTA b.
mapa.shared::cluster.type  d, sh, b;

// Maps shared memory variable into CTA b.
mapa.shared::cluster.type  d, sh + imm, b;

// Maps generic address in register a into CTA b.
mapa.type                  d, a, b;

.space = { .shared::cluster }
.type  = { .u32, .u64 }
```

Description

Get address in the CTA specified by operand `b` which corresponds to the address specified by
operand `a`.

Instruction type `.type` indicates the type of the destination operand `d` and the source
operand `a`.

When space is `.shared::cluster`, source `a` is either a shared memory variable or a register
containing a valid shared memory address and register `d` contains a shared memory address. When
the optional qualifier `.space` is not specified, both `a` and `d` are registers containing
generic addresses pointing to shared memory.

`b` is a 32-bit integer operand representing the rank of the target CTA.

Destination register `d` will hold an address in CTA `b` corresponding to operand `a`.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mapa.shared::cluster.u64 d1, %reg1, cta;
mapa.shared::cluster.u32 d2, sh, 3;
mapa.u64                 d3, %reg2, cta;
```

---

#### 9.7.9.24. [Data Movement and Conversion Instructions: `getctarank`](#data-movement-and-conversion-instructions-getctarank)

`getctarank`

Generate the CTA rank of the address.

Syntax

```
getctarank{.space}.type d, a;

// Get cta rank from source shared memory address in register a.
getctarank.shared::cluster.type d, a;

// Get cta rank from shared memory variable.
getctarank.shared::cluster.type d, var;

// Get cta rank from shared memory variable+offset.
getctarank.shared::cluster.type d, var + imm;

// Get cta rank from generic address of shared memory variable in register a.
getctarank.type d, a;

.space = { .shared::cluster }
.type  = { .u32, .u64 }
```

Description

Write the destination register `d` with the rank of the CTA which contains the address specified
in operand `a`.

Instruction type `.type` indicates the type of source operand `a`.

When space is `.shared::cluster`, source `a` is either a shared memory variable or a register
containing a valid shared memory address. When the optional qualifier `.space` is not specified,
`a` is a register containing a generic addresses pointing to shared memory. Destination `d` is
always a 32-bit register which holds the rank of the CTA.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
getctarank.shared::cluster.u32 d1, addr;
getctarank.shared::cluster.u64 d2, sh + 4;
getctarank.u64                 d3, src;
```
