# 9.7.3.11 Floating Point Instructions: min

Documents PTX `min` for f32 and f64 with two- and three-input forms. Supports `.NaN` (propagate NaN), `.xorsign.abs` (XOR sign bits with absolute values) modifiers. Three-input min requires sm_100+. Treats -0.0 as less than +0.0.
