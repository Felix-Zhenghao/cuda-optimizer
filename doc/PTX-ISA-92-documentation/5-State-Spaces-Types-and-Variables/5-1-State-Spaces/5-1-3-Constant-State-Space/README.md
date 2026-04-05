# 5.1.3.1. Banked Constant State Space (deprecated)

Describes the deprecated banked constant state space from PTX ISA prior to version 2.2, which organized constant memory into eleven fixed 64 KB banks referenced via `.const[bank]` syntax. Bank 0 held statically-sized constants; banks 1–10 held incomplete arrays. PTX ISA 2.2 replaced explicit banks with kernel parameter pointer attributes.
