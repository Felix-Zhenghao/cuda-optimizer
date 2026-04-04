# 9.7.15.1. Warpgroup

Defines a warpgroup as four contiguous warps where the first warp's rank is a multiple of 4. Warp rank is computed from thread/block dimensions. All four warps must execute warpgroup-level instructions; missing warps cause undefined behavior.
