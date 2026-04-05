##### 5.5.3.4.1. [Bounding Box](#tensor-tiled-scatter4-gather4-modes-bounding-box)

For `Tile::scatter4` and `Tile::gather4` modes, four request coordinates will form four Bounding
Boxes in the tensor space.

[Figure 10](#tiled-scatter4-gather4-bounding-box) shows an example of the same with start
coordinates (1, 2), (1, 5), (1, 0) and (1, 9).

The size of the bounding box in the dimension 0 represents the length of the rows.
The size of the bounding box in the dimension 1 must be one.

![_images/tiled-scatter4-gather4-bounding-box.png](img/Figure-10tiledscatter4tiledgather4-mode-bounding-box-example.png)

*Figure 10tiled::scatter4/tiled::gather4 mode bounding box example*
