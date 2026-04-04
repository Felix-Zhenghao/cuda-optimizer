# 9.7.14.5.1. Matrix Fragments for mma.m8n8k4 with .f16 Floating Point Type

Describes fragment distribution for `mma.m8n8k4` with f16. A warp computes 4 MMA operations of shape m8n8k4. Each thread holds specific elements of matrices A (f16 pairs), B (f16 pairs), C and D (f16 or f32), distributed via groupID and threadIdx within the group.
