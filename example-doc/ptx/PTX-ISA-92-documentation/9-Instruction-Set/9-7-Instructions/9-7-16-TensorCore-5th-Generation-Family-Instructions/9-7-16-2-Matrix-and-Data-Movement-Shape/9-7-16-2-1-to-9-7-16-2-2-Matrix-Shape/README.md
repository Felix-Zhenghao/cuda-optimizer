# 9.7.16.2.1-2. Matrix Shape

Defines supported MxNxK shapes for tcgen05 MMA operations. M can be 64 or 256 (set in instruction descriptor). N varies by kind. K depends on element type. K=96 requires sm_103a. Shape can be specified in the instruction descriptor or via the `.shape` qualifier.
