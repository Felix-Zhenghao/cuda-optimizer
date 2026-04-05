# Register-State-Space

Merged from: 5-1-1-Register-State-Space to 5-1-7-Shared-State-Space

Covers five core PTX state spaces: `.reg` (fast non-addressable registers, 1–128 bits wide), `.sreg` (predefined special registers for grid/CTA/thread parameters), `.global` (context-wide shared memory for cross-CTA communication), `.local` (per-thread private stack memory), and `.shared` (CTA-cluster-owned memory with `::cta`/`::cluster` sub-qualifiers enabling intra-cluster address sharing via `mapa`).
