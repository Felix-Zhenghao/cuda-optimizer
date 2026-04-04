# 9.7.16.8.4. Tensorcore 5th Generation Instructions: tcgen05.st

Documents `tcgen05.st` for asynchronous collective store from registers to Tensor Memory. Supports multiple shapes (32x32b, 16x32bx2, 16x128b, 16x256b). Optional `.unpack` qualifier unpacks packed elements. Requires `tcgen05.wait::st` for completion. Includes syntax, semantics, and PTX ISA requirements.
