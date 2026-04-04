# 9.7.3.9--9.7.3.10 Floating Point Instructions: abs, neg

Documents `abs` (absolute value) and `neg` (arithmetic negate) for f32 and f64 floating-point types. Both support flush-to-zero modifier for subnormals on f32. NaN handling varies: abs.f64 passes NaN through unchanged; abs.f32 and neg yield unspecified NaN.
