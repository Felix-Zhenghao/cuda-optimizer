# 5.1.6.4. Device Function Parameters

Explains device function parameters in `.param` space (PTX ISA 2.0+) for passing large structs by value as flattened byte arrays. Callers write via `st.param`; callees read via `ld.param`. Taking a parameter's address forces it into `.local` space. `mov` can obtain a return parameter's address from PTX ISA 6.0 onward.
