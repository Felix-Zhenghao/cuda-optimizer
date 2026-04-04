# Stack Manipulation Instructions: stacksave, stackrestore

Merged from: 9-7-17-1-Stack-Manipulation-Instructionsstacksave to 9-7-17-2-Stack-Manipulation-Instructionsstackrestore

Documents `stacksave` and `stackrestore` instructions for saving and restoring the stack pointer. `stacksave` copies the current stack pointer into a register; `stackrestore` sets the stack pointer from a register. Both support `.u32` and `.u64` types and require `sm_52` or higher. Introduced in PTX ISA 7.3.
