# 9.7.11.3. Surface Instructions: sured

The sured instruction performs atomic reduction operations (add, min, max, and, or) on surface memory. Supports byte-addressed (sured.b) and sample-addressed (sured.p) modes for 1d/2d/3d surfaces. Handles u32/s32/b32/u64/s64 types with trap/clamp/zero out-of-bounds behavior.
