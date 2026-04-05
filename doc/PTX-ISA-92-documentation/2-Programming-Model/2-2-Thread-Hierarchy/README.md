# Thread Hierarchy

- **2-2-1-Cooperative-Thread-Arrays** — Arrays of concurrent threads sharing a kernel; execute in SIMT warps of 32 threads with barrier synchronization.
- **2-2-2-Cluster-of-Cooperative-Thread-Arrays** — Groups of CTAs that can synchronize and share memory; requires sm_90 or higher.
- **2-2-3-Grid-of-Clusters** — Clusters batched into large grids for massive parallelism; threads across clusters cannot communicate.
