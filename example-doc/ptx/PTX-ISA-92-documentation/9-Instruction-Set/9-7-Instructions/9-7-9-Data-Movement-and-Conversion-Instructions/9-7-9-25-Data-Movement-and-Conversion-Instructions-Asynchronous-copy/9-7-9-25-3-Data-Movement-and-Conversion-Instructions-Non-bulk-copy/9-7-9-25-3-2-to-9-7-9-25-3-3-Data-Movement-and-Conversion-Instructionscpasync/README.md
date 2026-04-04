# 9.7.9.25.3.2-3. Data Movement and Conversion Instructions: cp.async.commit_group, cp.async.wait_group/wait_all

Covers async copy group management. `cp.async.commit_group` batches prior uncommitted cp.async operations into a per-thread cp.async-group. `cp.async.wait_group N` waits until only N most recent groups are pending. `cp.async.wait_all` commits and waits for all groups. No ordering within a group. Requires sm_80+. Introduced in PTX ISA 7.0.
