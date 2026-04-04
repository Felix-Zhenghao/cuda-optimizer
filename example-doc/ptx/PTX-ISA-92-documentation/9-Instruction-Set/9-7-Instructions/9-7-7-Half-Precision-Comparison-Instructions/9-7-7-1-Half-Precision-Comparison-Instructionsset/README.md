# 9.7.7.1. Half Precision Comparison Instructions: set

The half-precision `set` instruction compares two values with a relational operator and optionally combines the result with a predicate via Boolean operation. Supports f16, bf16, f16x2, and bf16x2 source types with integer or half-precision destination types. Packed f16x2/bf16x2 comparisons produce per-element results. Requires sm_53+; bf16 variants require sm_90+. Introduced in PTX ISA 4.2.
