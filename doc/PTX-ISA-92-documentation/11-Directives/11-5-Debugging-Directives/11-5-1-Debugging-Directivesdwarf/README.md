# 11.5.1. Debugging Directives:@@dwarf

Defines `@@DWARF` directive for embedding DWARF-format debug information directly in PTX. Supports raw byte, 4-byte integer, and 64-bit quad value lists, as well as labels. Used by compilers to associate PTX instructions with source-level debug information for debuggers.
