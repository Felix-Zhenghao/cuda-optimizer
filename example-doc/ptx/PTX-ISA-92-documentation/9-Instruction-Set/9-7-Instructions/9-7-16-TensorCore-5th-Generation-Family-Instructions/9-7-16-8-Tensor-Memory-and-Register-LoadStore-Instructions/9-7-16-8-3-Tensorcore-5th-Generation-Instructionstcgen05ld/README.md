# 9.7.16.8.3. Tensorcore 5th Generation Instructions: tcgen05.ld

Documents `tcgen05.ld` for asynchronous collective load from Tensor Memory into registers. Supports multiple shapes (32x32b, 16x32bx2, 16x128b, 16x256b). Optional `.pack` qualifier packs sub-word elements. Requires `tcgen05.wait::ld` for completion. Includes syntax, semantics, and PTX ISA requirements.
