# 5.5.3.4.1. Bounding Box

Describes the Bounding Box for `Tile::scatter4` and `Tile::gather4` modes: four request coordinates each define an independent Bounding Box in tensor space. The bounding box dimension-1 size must be 1; dimension-0 size represents the row length. Illustrates four simultaneous row-level accesses with different W starting coordinates.
