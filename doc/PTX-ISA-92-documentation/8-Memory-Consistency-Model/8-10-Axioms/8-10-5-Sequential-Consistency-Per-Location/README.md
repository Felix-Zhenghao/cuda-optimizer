# 8.10.5. Sequential Consistency Per Location

Guarantees that pairwise morally strong overlapping operations are strictly sequentially consistent: communication order cannot contradict program order. Demonstrated by the CoRR (Coherent Read-Read) litmus test, where if a read observes a write, all subsequent reads by the same thread must also observe that write or later.
