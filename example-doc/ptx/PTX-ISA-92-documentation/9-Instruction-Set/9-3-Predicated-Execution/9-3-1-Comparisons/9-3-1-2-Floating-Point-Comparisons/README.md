# 9.3.1.2. Floating-Point Comparisons

Defines ordered (eq, ne, lt, le, gt, ge) and unordered (equ, neu, ltu, leu, gtu, geu) floating-point comparison operators. Ordered comparisons return False if either operand is NaN; unordered comparisons return True if either is NaN. Also provides `num` and `nan` operators for testing whether operands are NaN values.
