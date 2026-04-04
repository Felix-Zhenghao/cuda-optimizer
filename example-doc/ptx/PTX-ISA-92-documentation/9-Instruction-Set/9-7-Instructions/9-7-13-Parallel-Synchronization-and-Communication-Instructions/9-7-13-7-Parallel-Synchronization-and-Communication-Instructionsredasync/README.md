# 9.7.13.7. red.async

Asynchronous non-blocking reduction operations. Supports cluster-scoped relaxed reductions on shared::cluster memory with mbarrier completion tracking (inc/dec/min/max/and/or/xor/add). Also supports release-semantic reductions on global memory for GPU/system scopes. Requires sm_90+.
