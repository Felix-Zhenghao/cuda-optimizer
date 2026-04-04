# 9.7.18.1.4. Scalar Video Instructions: vset

Documents the `vset` instruction for integer byte/half-word/word comparison. Compares two operands using `.eq`, `.ne`, `.lt`, `.le`, `.gt`, or `.ge`, producing an unsigned result. Supports optional secondary arithmetic operations (`.add`, `.min`, `.max`) or subword data merge. Requires `sm_20`+, introduced in PTX ISA 2.0.
