# 9.7.16.8.1-2. Access Restrictions

Defines Tensor Memory access restrictions for `tcgen05.ld` and `tcgen05.st`. Memory is divided into 4 chunks, one per warp in a warpgroup. Each warp can only access its own chunk. Also describes CTA group restrictions: cta_group::1 uses one warpgroup, cta_group::2 spans two.
