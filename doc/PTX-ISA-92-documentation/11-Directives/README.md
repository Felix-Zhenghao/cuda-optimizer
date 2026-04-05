# Directives

- **11-1-PTX-Module-Directives** — `.version`, `.target`, and `.address_size` directives for module-level ISA version, architecture, and pointer width.
- **11-2-Specifying-Kernel-Entry-Points-and-Functions** — `.entry`, `.func`, and `.alias` directives for defining kernels, device functions, and function aliases.
- **11-3-Control-Flow-Directives** — `.branchtargets`, `.calltargets`, and `.callprototype` for safe indirect branches and calls.
- **11-4-Performance-Tuning-Directives** — Register limits, CTA size requirements, occupancy hints, ABI preservation, and compiler pragma directives.
- **11-5-Debugging-Directives** — `@@DWARF`, `.section`, `.file`, and `.loc` for embedding source-level debug information.
- **11-6-Linking-Directives** — `.extern`, `.visible`, `.weak` for cross-module symbol visibility and separate compilation.
- **11-7-Cluster-Dimension-Directives** — `.reqnctapercluster` for specifying required CTA count per cluster (sm_90+).
- **11-8-Miscellaneous-Directives** — `.blocksareclusters` reinterprets grid launch config as cluster count instead of block count.
