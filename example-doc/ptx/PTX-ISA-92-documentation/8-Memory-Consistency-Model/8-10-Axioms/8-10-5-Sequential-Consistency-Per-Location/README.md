# 8.10.5 Sequential Consistency Per Location

Ensures that communication order cannot contradict program order for overlapping morally strong operations. Demonstrated with the CoRR litmus test: once a write is observed by a read, subsequent reads in the same thread must also observe it.
