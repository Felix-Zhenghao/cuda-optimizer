# 9.7.9.7. Data Movement and Conversion Instructions: prmt

The `prmt` instruction picks four arbitrary bytes from two 32-bit source registers and reassembles them into a 32-bit destination. In generic mode, a 16-bit permute control selects source bytes with optional sign extension. Specialized modes include f4e (forward extract), b4e (backward extract), rc8 (replicate 8), ecl/ecr (edge clamp), and rc16 (replicate 16). Requires sm_20+. Introduced in PTX ISA 2.0.
