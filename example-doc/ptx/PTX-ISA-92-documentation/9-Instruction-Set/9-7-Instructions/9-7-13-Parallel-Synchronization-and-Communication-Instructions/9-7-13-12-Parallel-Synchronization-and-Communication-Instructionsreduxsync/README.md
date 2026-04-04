# 9.7.13.12. redux.sync

Warp-level synchronized reduction across predicated active threads. Supports arithmetic operations (add/min/max on u32/s32) and bitwise operations (and/or/xor on b32). Also supports f32 min/max with optional abs and NaN qualifiers. Requires sm_80+; f32 requires sm_100a.
