# 9.7.16.9.2. Tensorcore 5th Generation Instructions: tcgen05.cp

Documents `tcgen05.cp` for asynchronous copy from shared memory to Tensor Memory. Supports multiple shapes (128x256b, 32x128b, 128x128b). Optional multicast copies to multiple CTAs. Optional format conversion (decompression) during copy. Completion tracked via mbarrier. Requires PTX ISA 8.6+.
