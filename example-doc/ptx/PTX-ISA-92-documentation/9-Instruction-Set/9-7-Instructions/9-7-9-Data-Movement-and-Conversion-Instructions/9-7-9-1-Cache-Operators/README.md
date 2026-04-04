# 9.7.9.1. Cache Operators

Describes optional cache operators for load and store instructions introduced in PTX ISA 2.0 (requires sm_20+). Load operators: .ca (cache all levels), .cg (cache global/L2 only), .cs (cache streaming/evict-first), .lu (last use), .cv (don't cache, fetch again). Store operators: .wb (write-back), .cg (global cache only), .cs (streaming), .wt (write-through). All are performance hints only.
