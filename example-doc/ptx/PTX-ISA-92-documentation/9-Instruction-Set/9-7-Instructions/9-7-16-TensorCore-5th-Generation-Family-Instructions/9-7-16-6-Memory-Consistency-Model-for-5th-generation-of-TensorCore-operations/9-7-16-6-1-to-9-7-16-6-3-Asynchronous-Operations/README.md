# 9.7.16.6.1-3. Asynchronous Operations

Classifies tcgen05 instructions into asynchronous (mma, cp, shift) and synchronous (ld, st, alloc, dealloc) categories. Asynchronous operations execute in the async proxy and require explicit ordering via fence, commit, and wait mechanisms. Defines the memory consistency model and ordering rules.
