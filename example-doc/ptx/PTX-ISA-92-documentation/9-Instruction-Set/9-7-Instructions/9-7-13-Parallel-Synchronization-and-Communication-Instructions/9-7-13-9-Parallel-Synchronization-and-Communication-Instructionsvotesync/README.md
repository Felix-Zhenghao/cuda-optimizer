# 9.7.13.9. vote.sync

Synchronized warp-level predicate reduction. Threads in membermask synchronize then perform all/any/uni reduction or ballot across non-exited threads. Replaces deprecated vote instruction with explicit synchronization via membermask. Requires sm_30+, introduced in PTX ISA 6.0.
