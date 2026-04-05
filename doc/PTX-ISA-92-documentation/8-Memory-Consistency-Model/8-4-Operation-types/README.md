# Operation-types

Defines two special operation types: mmio operations (for memory-mapped I/O registers, equivalent to strong operations with strict hardware constraints preventing caching or combining within scope) and volatile operations (relaxed at system scope, preserving instruction count but not operation count; not suitable for MMIO).
