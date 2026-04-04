# 11.2.2: Kernel and Function Directives: .func

Defines device functions with input/return parameters and optional body. Supports .noreturn, .attribute, .abi_preserve, and .abi_preserve_control directives. Parameters can be in register or .param state space. Supports unsized array parameters for variadic-style calls. ABI details including stack and recursion support for sm_20+.
