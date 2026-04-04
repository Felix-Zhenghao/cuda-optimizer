# 9.7.16.10.8.5. Alignment Restriction

Specifies alignment restrictions for sparse tcgen05 MMA operations. Layouts using only half the datapath lanes (Layout F and Layout C) must have even-aligned Tensor Memory column addresses for matrices A and D. The column index (bits 15:0) must be even.
