# 5.5.4.1. Bounding Box

Defines the im2col mode Bounding Box in DHW space (two dimensions fewer than tensor dimensionality). Access properties include Lower/Upper Corner coordinates (16-bit signed), Pixels-per-Column (total elements in NDHW space), and Channels-per-Pixel. Tensor coordinates specify convolution filter base location; im2col offsets added to filter base determine the actual data access start point.
