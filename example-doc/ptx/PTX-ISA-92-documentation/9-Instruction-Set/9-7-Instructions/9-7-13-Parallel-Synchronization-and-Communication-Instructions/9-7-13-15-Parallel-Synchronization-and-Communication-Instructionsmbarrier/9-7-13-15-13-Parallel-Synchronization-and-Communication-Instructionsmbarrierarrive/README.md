# 9.7.13.15.13. mbarrier.arrive

Performs arrive-on operation on an mbarrier object, decrementing pending arrival count. Supports optional count argument, expect_tx qualifier for tracking async transactions, noComplete modifier, and shared::cta or shared::cluster addressing. Returns opaque phase state for subsequent wait operations. Configurable release/relaxed semantics.
