# State-Spaces

- **5-1-1-to-5-1-7-Register-State-Space** — Core PTX state spaces: registers, special registers, global, local, and shared memory with cluster sub-qualifiers.
- **5-1-3-Constant-State-Space** — Deprecated banked constant memory (eleven 64 KB banks) from pre-PTX-2.2, replaced by pointer kernel parameters.
- **5-1-6-Parameter-State-Space** — Kernel and device function parameters in `.param` space, including the `.ptr` attribute for pointer typing.
- **5-1-8-Texture-State-Space-deprecated** — Deprecated `.tex` state space; equivalent to `.global .texref` and retained for backward compatibility.
