# 9.7.13.18. clusterlaunchcontrol.query_cancel

Decodes the opaque 16-byte response from clusterlaunchcontrol.try_cancel. The is_canceled variant checks success via predicate. The get_first_ctaid variant extracts the x/y/z coordinates of the first CTA in the canceled cluster, either as a vector or individual dimensions. Requires sm_100+.
