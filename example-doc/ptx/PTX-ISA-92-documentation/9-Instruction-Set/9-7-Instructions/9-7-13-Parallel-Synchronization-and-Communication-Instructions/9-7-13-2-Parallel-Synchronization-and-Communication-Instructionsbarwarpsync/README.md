# 9.7.13.2. bar.warp.sync

Warp-level barrier synchronization. Threads specified in a 32-bit membermask wait until all corresponding threads execute bar.warp.sync with the same mask. Guarantees memory ordering among participating threads, enabling safe intra-warp shared memory communication. Requires sm_30+.
