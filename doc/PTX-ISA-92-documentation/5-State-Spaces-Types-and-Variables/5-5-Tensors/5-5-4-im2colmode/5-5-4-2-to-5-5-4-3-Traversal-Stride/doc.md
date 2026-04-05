#### 5.5.4.2. [Traversal Stride](#tensor-im2col-mode-traversal-stride)

The traversal stride, in im2col mode, does not impact the total number of elements (or pixels) being
accessed unlike the tiled mode. Pixels-per-Column determines the total number of elements being
accessed, in im2col mode.

The number of elements traversed along the D, H and W dimensions is strided by the traversal stride
for that dimension.

The following example with [Figure 15](#tensor-im2col-mode-example3) illustrates accesse with traversal-strides:

```
Tensor Size[0] = 64
Tensor Size[1] = 8
Tensor Size[2] = 14
Tensor Size[3] = 64
Traversal Stride = 2
Pixels-per-Column = 32
channels-per-pixel = 16
Bounding-Box Lower-Corner W = -1
Bounding-Box Lower-Corner H = -1
Bounding-Box Upper-Corner W = -1
Bounding-Box Upper-Corner H = -1.
Tensor coordinates in the instruction = (7, 7, 5, 0)
Im2col offsets in the instruction : (1, 1)
```

![_images/tensor-im2col-mode-example3.png](img/Figure-15im2col-mode-traversal-stride-example.png)

*Figure 15im2col mode traversal stride example*

---

#### 5.5.4.3. [Out of Boundary Access](#tensor-im2col-mode-oob-access)

In im2col mode, when the number of requested pixels in NDHW space specified by *Pixels-per-Column*
exceeds the number of available pixels in the image batch then out-of-bounds access is performed.

Similar to tiled mode, zero fill or `OOB-NaN` fill can be performed based on the Fill-Mode
specified.
