# Scalar Video Instructions: vadd, vsub, vabsdiff, vmin, vmax, vshl, vshr

Merged from: 9-7-18-1-1-Scalar-Video-Instructionsvaddvsubvabsdiffvminvmax to 9-7-18-1-2-Scalar-Video-Instructionsvshlvshr

Documents scalar video arithmetic and shift instructions. `vadd`, `vsub`, `vabsdiff`, `vmin`, `vmax` perform 32-bit integer byte/half-word/word operations with optional saturation, secondary operations, or subword merge. `vshl`/`vshr` perform shifts with clamp or wrap modes. All support `.u32`/`.s32` types, require `sm_20`+, PTX ISA 2.0.
