# 9.7.3.12 Floating Point Instructions: max

Documents PTX `max` for f32 and f64 with two- and three-input forms. Supports `.NaN` (propagate NaN), `.xorsign.abs` (XOR sign bits with absolute values) modifiers. Three-input max requires sm_100+. Treats +0.0 as greater than -0.0.
