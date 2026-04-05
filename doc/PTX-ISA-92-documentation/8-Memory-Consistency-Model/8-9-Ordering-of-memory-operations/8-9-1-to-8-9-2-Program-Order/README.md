# Program-Order

Merged from: 8-9-1-Program-Order to 8-9-2-to-8-9-3-Observation-Order

Defines program order (sequential instruction ordering within a thread), asynchronous operations (cp.async, cp.async.bulk, wgmma.mma_async) that are ordered after prior instructions but not in program order, observation order (write-to-read relation through atomic RMW chains), and Fence-SC order (acyclic partial order over morally strong fence.sc operations).
