# 5.8. CUDA C++ Execution model

Specifies CUDA C++ forward progress guarantees for host and device threads relative to the C++ standard. Device threads provide parallel forward progress; all threads in a thread-block cluster eventually make progress once one starts. Includes examples of API forward progress and stream dependency behavior.
