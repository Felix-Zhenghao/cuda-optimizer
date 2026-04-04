# 6.6 Operand Costs

Provides cost estimates for accessing different PTX state spaces. Registers, shared, constant, and parameter accesses are effectively free (0 cost), while global, local, texture, and surface memory accesses cost over 100 clock cycles. Discusses latency-hiding strategies.
