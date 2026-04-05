# 11.1.2. PTX Module Directives:.target

Specifies target GPU architecture and platform options. Standard architectures are forward-compatible (onion-layer model). Suffix `a` targets (e.g., sm_90a) include architecture-specific features not forward-compatible; suffix `f` targets (e.g., sm_100f) are family-specific. Also controls texture mode (unified/independent) and f64-to-f32 mapping options.
