# 2.1. A Highly Multithreaded Coprocessor

The GPU acts as a massively parallel coprocessor to the CPU, executing compute-intensive, data-parallel kernel functions off-loaded from the host. Kernel code compiled to PTX runs as many independent threads and is translated at install time to the native GPU instruction set.
