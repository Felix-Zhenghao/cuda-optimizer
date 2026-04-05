# 8.1.1. Limitations on atomicity at system scope

Covers the limitation that strong operations at system scope may not be performed atomically when communicating with the host CPU on certain systems. The CTA is the minimum scope unit; the warp is not a scope. Refers to CUDA Atomicity Requirements for details on host-memory guarantees.
