# 4.5. Programmatic Dependent Launch and Synchronization

Programmatic Dependent Launch lets a secondary kernel start executing before its primary kernel completes, unlocking GPU pipeline parallelism on compute capability 9.0+ devices. Covers the trigger-at-block-start mechanism, CUDA event-based launch, synchronization semantics, programming guidelines, and stream capture integration with CUDA Graphs.
