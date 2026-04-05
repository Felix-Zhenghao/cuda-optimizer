# 5.5.5.4.wOffset

Explains the `wOffset` argument in tensor copy/prefetch instructions for `im2col::w` mode: adjusts the W-dimension Bounding Box Lower/Upper Corners and the W coordinate by `wOffset`, allowing the same filter footprint to load into multiple separate shared memory buffers. Enables efficient multi-buffer convolution loading by offsetting each buffer's source pixel location along W.
