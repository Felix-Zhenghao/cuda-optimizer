# 9.7.9.12. Data Movement and Conversion Instructions: st.async

The `st.async` instruction initiates a non-blocking asynchronous store. In weak/cluster mode, stores to shared::cluster memory with mbarrier completion signaling. In release mode, performs a release store to global memory with .gpu or .sys scope. Supports .mmio qualifier for memory-mapped I/O. Requires sm_90+; release/global variants require sm_100+. Introduced in PTX ISA 8.1.
