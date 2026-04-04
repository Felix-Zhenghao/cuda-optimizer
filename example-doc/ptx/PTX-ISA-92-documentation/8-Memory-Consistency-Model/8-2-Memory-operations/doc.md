### 8.2.1. [Overlap](#overlap)

Two memory locations are said to overlap when the starting address of one location is within the
range of bytes constituting the other location. Two memory operations are said to overlap when they
specify the same virtual address and the corresponding memory locations overlap. The overlap is said
to be complete when both memory locations are identical, and it is said to be partial otherwise.

---

### 8.2.2. [Aliases](#aliases)

Two distinct virtual addresses are said to be aliases if they map to the same memory location.

---

### 8.2.3. [Multimem Addresses](#multimem-addresses)

A multimem address is a virtual address which points to multiple distinct memory locations across
devices.

Only *multimem.*\* operations are valid on multimem addresses. That is, the behavior of accessing
a multimem address in any other memory operation is undefined.

---

### 8.2.4. [Memory Operations on Vector Data Types](#memory-operations-on-vector-data-types)

The memory consistency model relates operations executed on memory locations with scalar data types,
which have a maximum size and alignment of 64 bits. Memory operations with a vector data type are
modelled as a set of equivalent memory operations with a scalar data type, executed in an
unspecified order on the elements in the vector.

---

### 8.2.5. [Memory Operations on Packed Data Types](#memory-operations-on-packed-data-types)

A packed data type consists of two values of the same scalar data type, as described in
[Packed Data Types](#packed-data-types). These values are accessed in adjacent memory locations. A
memory operation on a packed data type is modelled as a pair of equivalent memory operations on the
scalar data type, executed in an unspecified order on each element of the packed data.

---

### 8.2.6. [Initialization](#initialization)

Each byte in memory is initialized by a hypothetical write *W0* executed before starting any thread
in the program. If the byte is included in a program variable, and that variable has an initial
value, then *W0* writes the corresponding initial value for that byte; else *W0* is assumed to have
written an unknown but constant value to the byte.
