# 6.4.1. Addresses as Operands

PTX supports generic addressing where memory instructions without an explicit state space operate through a unified address space. The .const, .param, .local, and .shared spaces are mapped as windows within the generic address space; all other addresses map to global memory.
