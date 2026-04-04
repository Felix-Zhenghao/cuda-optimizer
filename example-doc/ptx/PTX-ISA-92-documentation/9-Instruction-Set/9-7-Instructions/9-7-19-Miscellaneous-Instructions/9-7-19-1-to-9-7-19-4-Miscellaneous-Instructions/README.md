# Miscellaneous Instructions: brkpt, nanosleep, pmevent, trap

Merged from: 9-7-19-1-Miscellaneous-Instructionsbrkpt to 9-7-19-4-Miscellaneous-Instructionstrap

Documents four miscellaneous PTX instructions. `brkpt` suspends execution as a breakpoint (sm_11+). `nanosleep` suspends a thread for an approximate nanosecond delay in the range [0, 2*t] (sm_70+). `pmevent` triggers performance monitor events with index or mask mode (PTX 1.4+). `trap` aborts execution and signals the host CPU, supported on all architectures.
