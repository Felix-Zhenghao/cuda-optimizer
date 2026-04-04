# 9.7.13.3. barrier.cluster

Cluster-level barrier synchronization with separate arrive and wait phases. All non-exited cluster threads must arrive before the barrier completes. Supports release/relaxed semantics on arrive and acquire on wait. Provides memory ordering guarantees for cross-CTA communication within a cluster.
