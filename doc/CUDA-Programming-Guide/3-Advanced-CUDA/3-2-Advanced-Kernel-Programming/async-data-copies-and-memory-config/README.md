# Asynchronous Data Copies and Memory Configuration

Explains hardware-accelerated asynchronous data copy mechanisms (LDGSTS, TMA, STAS) that overlap computation with data movement within a kernel, and covers per-kernel configuration of the L1/shared memory carveout balance using `cudaFuncSetAttribute`.
