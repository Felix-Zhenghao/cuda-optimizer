# 9.7.13.15.17. mbarrier.pending_count

Queries the pending arrival count from an opaque mbarrier state register. The state must originate from a prior mbarrier.arrive.noComplete or mbarrier.arrive_drop.noComplete instruction. Returns a 32-bit unsigned integer count. Requires sm_80+, introduced in PTX ISA 7.0.
