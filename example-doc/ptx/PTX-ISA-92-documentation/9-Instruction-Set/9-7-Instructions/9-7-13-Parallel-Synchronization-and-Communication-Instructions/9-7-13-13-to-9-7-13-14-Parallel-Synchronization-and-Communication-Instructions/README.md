# 9.7.13.13-14. griddepcontrol, elect.sync

griddepcontrol manages dependent grid execution: launch_dependents signals dependents can be scheduled, wait blocks until prerequisites complete. elect.sync deterministically elects a leader thread from a membermask, returning its laneid and setting a predicate. Both require sm_90+.
