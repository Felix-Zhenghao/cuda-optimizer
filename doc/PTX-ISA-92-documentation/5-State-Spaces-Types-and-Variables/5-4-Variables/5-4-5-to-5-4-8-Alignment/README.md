# Alignment

Merged from: 5-4-5-Alignment to 5-4-8-Variable-and-Function-Attribute-Directiveattribute

Covers PTX alignment via `.align byte-count` specifier (power-of-two, applies to array start address). Parameterized variable names use `%prefix<N>` shorthand to declare N variables at once. The `.attribute` directive supports `.managed` (unified virtual memory, sm_30+) and `.unified` (same host/device address with UUID, sm_90+) variable and function attributes.
