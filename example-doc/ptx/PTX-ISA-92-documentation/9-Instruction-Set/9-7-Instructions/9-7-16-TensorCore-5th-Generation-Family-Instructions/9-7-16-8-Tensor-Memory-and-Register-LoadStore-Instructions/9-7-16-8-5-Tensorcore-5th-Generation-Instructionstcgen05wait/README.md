# 9.7.16.8.5. Tensorcore 5th Generation Instructions: tcgen05.wait

Documents `tcgen05.wait` which waits for completion of prior async `tcgen05.ld`/`tcgen05.st` instructions. Has two variants: `tcgen05.wait::ld` for loads and `tcgen05.wait::st` for stores. Must be issued by all threads in the warp. Requires PTX ISA 8.6+.
