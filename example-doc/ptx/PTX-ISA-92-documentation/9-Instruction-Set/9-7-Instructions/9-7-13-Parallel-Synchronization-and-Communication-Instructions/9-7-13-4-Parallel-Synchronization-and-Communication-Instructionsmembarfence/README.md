# 9.7.13.4. membar / fence

Memory ordering instructions. membar enforces ordering at cta/gl/sys levels. fence provides acq_rel/sc semantics with cta/cluster/gpu/sys scopes. Includes proxy fence variants for alias, async, and tensormap proxy ordering. fence.sc restores sequential consistency. Supports sync_restrict and op_restrict qualifiers.
