# 9.7.12.6-7. Control Flow Instructions: ret, exit

The ret instruction returns execution to the caller, with divergent returns suspending threads until all are ready. The exit instruction terminates a thread. Barriers exclusively waiting on arrivals from exited threads are always released. Both support predicated execution.
