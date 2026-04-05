Documents the synchronization proxy instructions required for correct warpgroup MMA execution ordering.

- **9-7-15-7-1-wgmma.fence**: `wgmma.fence.sync.aligned` enforces ordering between thread register accesses and `wgmma.mma_async` instruction register access
- **9-7-15-7-2-to-9-7-15-7-3**: `wgmma.commit_group` batches async operations into groups; `wgmma.wait_group N` blocks until at most N groups remain pending
