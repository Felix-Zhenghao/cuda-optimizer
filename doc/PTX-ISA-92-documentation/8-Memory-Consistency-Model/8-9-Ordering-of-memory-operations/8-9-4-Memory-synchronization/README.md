# 8.9.4. Memory synchronization

Defines how synchronizing operations across threads establish causality order: fence.sc pairs, bar.sync/red/arrive barriers, barrier.cluster operations, and release/acquire pattern pairs. Also covers CUDA API synchronization (streams, events, graphs, kernel start/end) that establish synchronizes-with relations and participate in proxy-preserved base causality order.
