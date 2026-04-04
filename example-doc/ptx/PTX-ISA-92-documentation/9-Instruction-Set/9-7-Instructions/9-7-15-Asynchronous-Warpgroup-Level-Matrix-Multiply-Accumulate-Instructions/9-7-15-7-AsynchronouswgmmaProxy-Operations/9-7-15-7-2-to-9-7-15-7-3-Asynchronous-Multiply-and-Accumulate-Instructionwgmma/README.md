# 9.7.15.7.2-3. wgmma.commit_group and wgmma.wait_group

Documents `wgmma.commit_group` which groups prior uncommitted `wgmma.mma_async` operations into a wgmma-group, and `wgmma.wait_group` which blocks until at most N wgmma-groups remain pending. Both require all warpgroup threads to participate.
