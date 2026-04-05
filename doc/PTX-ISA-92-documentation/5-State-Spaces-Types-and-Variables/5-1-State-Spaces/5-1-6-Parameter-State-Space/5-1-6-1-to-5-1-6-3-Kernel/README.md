# Kernel

Merged from: 5-1-6-1-Kernel-Function-Parameters to 5-1-6-3-Kernel-Parameter-Attributeptr

Covers kernel function parameters declared in `.param` space as addressable read-only variables accessed via `ld.param`. The `.ptr` attribute (PTX ISA 2.2+) annotates pointer parameters with target state space (const/global/local/shared) and byte alignment, enabling the compiler and runtime to correctly manage pointer provenance and generic addressing across kernel invocations.
