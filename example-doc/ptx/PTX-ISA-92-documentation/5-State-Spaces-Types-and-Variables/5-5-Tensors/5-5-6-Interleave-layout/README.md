# 5.5.6. Interleave Layout

Describes supported tensor interleave layouts: no interleave (NDHWC), 8-byte interleave (NC/8DHWC8, 16B per slice), and 16-byte interleave (NC/16HWC16, 32B per slice). Channel data is organized in slices with zero-padding for incomplete slices. Supported for 3D/4D/5D tensors only; not supported for im2col::w modes.
