# 9.7.17.3. Stack Manipulation Instructions: alloca

Documents the `alloca` instruction for dynamic stack memory allocation. Allocates memory on the current function's stack frame with configurable alignment (power of 2, up to 2^23). Returns a local-memory pointer usable with `ld.local`/`st.local`. Memory is deallocated on function exit or via `stacksave`/`stackrestore`. Requires `sm_52`+, PTX ISA 7.3.
