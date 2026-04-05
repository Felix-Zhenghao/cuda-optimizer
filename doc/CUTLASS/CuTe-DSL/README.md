# CuTe-DSL

- **Introduction**: Overview of CuTe DSL with @jit/@kernel decorators and calling conventions.
- **Control-Flow**: Python control flow handling: for loops, while, if/else with compile-time vs runtime.
- **Code-Generation**: Hybrid AST-rewrite and tracing code generation pipeline.
- **Compile-with-TVM-FFI**: Apache TVM FFI integration for faster invocation and framework interop.
- **Debugging-with-the-DSL**: Debugging tools: source correlation, logging, IR dumping.
- **Integration-with-Frameworks**: DLPack-based integration with PyTorch, JAX, and other frameworks.
- **JIT-Argument-Generation**: JIT argument tracing, static/dynamic arguments, and type safety.
- **JIT-Argument-Layouts**: Static and dynamic layout handling for framework tensor conversion.
- **JIT-Caching**: Zero Compile and JIT Executor caching to avoid recompilation.
- **JIT-Compilation-Options**: Compilation options for optimization, debug flags, and cute.compile.
- **Ahead-of-Time-AOT-Compilation**: Pre-compile kernels for production with CuTe ABI AOT workflow.
- **Autotuning-with-the-DSL**: Auto-tuning guidance for GEMM kernel parameter selection.
- **None-None-Deprecation-Policy**: Deprecation policy, educational notebooks, and conference talks.
