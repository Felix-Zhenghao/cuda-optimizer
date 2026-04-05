# Bounding-Box

Merged from: 5-5-5-1-Bounding-Box to 5-5-5-3-wHalo

In `im2col::w` and `im2col::w::128` modes, D and H bounding-box dimensions are fixed at 1; only W dimension uses Lower/Upper Corner bounds. `im2col::w` loads Pixels-per-Column pixels plus `wHalo` filter-halo elements at row end. `im2col::w::128` always loads 128 elements inserting `wHalo` halo elements after every 32 main elements.
