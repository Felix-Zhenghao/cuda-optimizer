# 5.2.4. Fixed-point Data format

Defines the `s2f6` fixed-point format: an 8-bit signed two's complement value with 2 integer bits and 6 fractional bits (form xx.xxxxxx). The effective value equals the s8 integer times 2^(-6), ranging from -2.0 to approximately +1.984375. Does not support infinity or NaN.
