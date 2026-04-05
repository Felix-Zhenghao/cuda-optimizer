# 2.3. Asynchronous Execution

Explains CUDA Streams (in-order work queues), Events (timing and dependency markers), callback functions via `cudaLaunchHostFunc`, blocking vs. non-blocking streams, the legacy default stream synchronization semantics, per-thread default streams, explicit and implicit synchronization APIs, stream prioritization, and an introduction to CUDA Graphs via stream capture for reducing repeated-launch CPU overhead.
