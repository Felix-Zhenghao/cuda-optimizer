# 4.5.5. Integer Constant Expression Evaluation

Integer constant expressions are evaluated at compile time as 64-bit signed (.s64) or unsigned (.u64) values. Type propagation rules (based on C's usual arithmetic conversions) determine result types. Behavior for all operators, including remainder and shifts, is fully defined.
