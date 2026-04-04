# 11.5.4: Debugging Directives: .loc

Declares source file location (file index, line, column) for subsequent PTX instructions. Supports .inlined_at attribute for representing inlined function information, referencing function names via offsets in .debug_str DWARF sections. Labels inherit the location of their following instruction. Introduced in PTX ISA 1.0, inline support added in 7.2.
