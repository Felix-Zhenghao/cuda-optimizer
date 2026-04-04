# 9.7.9.25. Asynchronous Copy Instructions

- **9-7-9-25-1 Completion Mechanisms** -- Async-group and mbarrier-based completion tracking mechanisms.
- **9-7-9-25-2 Async Proxy** -- Async proxy concept and cross-proxy fence requirements.
- **9-7-9-25-3 Non-bulk Copy** -- cp.async for small async copies with commit/wait group management.
- **9-7-9-25-4 Bulk Copy** -- cp.async.bulk, cp.reduce.async.bulk, prefetch, and multimem variants.
- **9-7-9-25-5 Tensor Copy** -- cp.async.bulk.tensor operations using tensor-map objects.
- **9-7-9-25-6 Completion Instructions** -- cp.async.bulk.commit_group and wait_group for bulk operations.
