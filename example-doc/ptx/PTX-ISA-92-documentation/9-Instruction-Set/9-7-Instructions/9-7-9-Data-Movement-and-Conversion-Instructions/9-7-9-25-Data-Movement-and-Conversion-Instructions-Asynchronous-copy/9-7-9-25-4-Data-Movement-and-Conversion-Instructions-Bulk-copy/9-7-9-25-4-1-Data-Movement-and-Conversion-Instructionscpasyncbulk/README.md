# 9.7.9.25.4.1. Data Movement and Conversion Instructions: cp.async.bulk

The `cp.async.bulk` instruction initiates asynchronous bulk copy between state spaces: global-to-shared::cta, global-to-shared::cluster (with optional multicast), shared::cta-to-shared::cluster, and shared::cta-to-global. Size must be a multiple of 16 bytes with 16-byte alignment. Supports mbarrier and bulk_group completion mechanisms, cache hints, multicast, cp_mask, and ignore_oob. Requires sm_90+. Introduced in PTX ISA 8.0.
