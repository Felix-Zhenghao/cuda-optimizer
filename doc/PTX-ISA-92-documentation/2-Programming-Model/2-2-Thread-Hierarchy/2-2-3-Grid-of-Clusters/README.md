# 2.2.3. Grid of Clusters

Clusters are batched into a grid to scale kernel execution across very large thread counts. Threads in different clusters cannot communicate. Grids have unique identifiers and may be launched with dependencies between them, enabling CUDA Graph-based execution ordering.
