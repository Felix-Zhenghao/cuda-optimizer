# 8.10.6. Causality

Constrains reads using causality order: a read cannot read from a write that follows it in causality order, and if a write precedes a read in causality, the read must observe that write or a later one in coherence order. Illustrated by the MP (Message Passing), CoWR (alias proxy fence), and SB (Store Buffering with fence.sc) litmus tests.
