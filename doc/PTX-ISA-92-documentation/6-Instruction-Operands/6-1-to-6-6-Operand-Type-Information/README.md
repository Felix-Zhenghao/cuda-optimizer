# 6.1–6.6. Operand Type Information and Costs

All PTX operand types must be compatible with the instruction type; no automatic conversion occurs. Source operands must reside in .reg space for ALU instructions. Operand access cost varies by state space: registers and shared memory are fastest; global, local, texture, and surface are slowest (>100 clocks).
