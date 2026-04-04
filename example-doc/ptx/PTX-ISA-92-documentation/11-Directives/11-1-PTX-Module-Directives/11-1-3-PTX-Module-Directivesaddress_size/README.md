# 11.1.3: PTX Module Directives: .address_size

Specifies the address size (32 or 64 bits) used throughout a PTX module, including binary DWARF information. Optional but must follow the .target directive if present. Defaults to 32-bit if omitted. All modules in separate compilation must use the same address size. Introduced in PTX ISA version 2.3.
