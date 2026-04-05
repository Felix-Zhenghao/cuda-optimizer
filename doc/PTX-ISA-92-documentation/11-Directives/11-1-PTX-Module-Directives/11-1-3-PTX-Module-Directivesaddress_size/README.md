# 11.1.3. PTX Module Directives:.address_size

Defines `.address_size` directive (32 or 64) setting the pointer width for the entire PTX module and its DWARF debug information. Optional and defaults to 32-bit; must immediately follow `.target` if specified. All modules in a separate compilation must use the same address size.
