# 5.1.8. Texture State Space (deprecated)

Documents the deprecated `.tex` state space for global read-only cached texture memory, sequentially bound to hardware texture identifiers (up to 128 per kernel). The `.tex` directive is retained for backward compatibility; programs should use `.texref` variables in `.global` space instead. A `.tex .u32 tex_a` declaration is equivalent to `.global .texref tex_a`.
