# 9.1--9.2. Format and Semantics of Instruction Descriptions / PTX Instructions

Describes the general format of PTX instruction documentation and the structure of PTX instructions. Instructions have zero to four operands plus an optional guard predicate (`@p`). The destination operand `d` receives results, while `a`, `b`, `c` are sources. The `setp` instruction supports dual destinations separated by `|`, and an underscore `_` serves as a bit-bucket discard destination.
