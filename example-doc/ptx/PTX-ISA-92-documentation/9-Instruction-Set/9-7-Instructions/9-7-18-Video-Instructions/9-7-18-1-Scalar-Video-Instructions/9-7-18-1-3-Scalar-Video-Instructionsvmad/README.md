# 9.7.18.1.3. Scalar Video Instructions: vmad

Documents the `vmad` instruction for integer byte/half-word/word multiply-accumulate. Computes `(a*b) + c` with optional operand negation, plus-one mode for averages, and right-shift scaling (`.shr7`, `.shr15`). Supports signed/unsigned combinations with optional saturation. Requires `sm_20` or higher, introduced in PTX ISA 2.0.
