# 9.7.13.15.14. mbarrier.arrive_drop

Decrements the expected arrival count of an mbarrier object (permanently for all subsequent phases) then performs an arrive-on operation. Used by threads opting out of future barrier participation. Supports expect_tx, noComplete, shared::cluster addressing, and configurable release/relaxed semantics.
