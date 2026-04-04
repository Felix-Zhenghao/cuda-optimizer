# 8.10.4 No Thin Air

Forbids self-fulfilling speculative value cycles in memory operations. Demonstrated via Load Buffering litmus tests showing that dependency-based cycles must resolve to initial values, while independent stores may freely reorder.
