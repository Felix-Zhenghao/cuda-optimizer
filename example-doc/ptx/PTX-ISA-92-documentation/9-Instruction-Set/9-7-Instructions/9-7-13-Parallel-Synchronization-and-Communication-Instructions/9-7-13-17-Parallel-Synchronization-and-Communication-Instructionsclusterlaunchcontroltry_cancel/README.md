# 9.7.13.17. clusterlaunchcontrol.try_cancel

Asynchronously requests cancellation of a not-yet-launched cluster. Writes an opaque 16-byte response to shared memory, tracked via mbarrier completion mechanism at cluster scope. On success, response contains ctaid of the canceled cluster's first CTA. Supports multicast::cluster::all. Requires sm_100+.
