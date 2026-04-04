# 9.7.9.25.2. Async Proxy

Explains that cp{.reduce}.async.bulk operations execute in the asynchronous proxy. Cross-proxy memory access requires fence.proxy.async for synchronization between generic and async proxies. Completion of async bulk operations includes an implicit generic-async proxy fence, making results visible to the generic proxy upon observed completion via async-group or mbarrier mechanisms.
