# 8.10.4. No Thin Air

Forbids values appearing "out of thin air" via self-satisfying speculative execution cycles. Demonstrated by the LB+deps litmus test (load buffering with dependencies), which cannot produce nonzero final values. The LB litmus test without dependencies (unconditional stores) is explicitly permitted by the PTX memory model.
