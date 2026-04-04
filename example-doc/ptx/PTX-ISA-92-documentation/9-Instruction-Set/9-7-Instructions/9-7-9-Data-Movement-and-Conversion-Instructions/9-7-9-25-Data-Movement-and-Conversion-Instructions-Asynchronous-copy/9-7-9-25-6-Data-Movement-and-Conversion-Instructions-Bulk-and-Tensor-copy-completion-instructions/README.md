# 9.7.9.25.6. Bulk and Tensor Copy Completion Instructions

Covers two bulk async-group management instructions. `cp.async.bulk.commit_group` creates a new per-thread bulk async-group batching prior uncommitted cp{.reduce}.async.bulk operations. `cp.async.bulk.wait_group N` waits until only N most recent bulk async-groups remain pending. Optional .read modifier waits only for source reads to complete. Requires sm_90+. Introduced in PTX ISA 8.0.
