# 9.7.7.2. Half Precision Comparison Instructions: setp

The half-precision `setp` instruction compares two values with a relational operator and optionally combines the result with a predicate via Boolean operation, writing to predicate destinations. For f16x2/bf16x2 types, element-wise comparisons produce separate predicate outputs p and q. Supports ordered/unordered comparisons and NaN handling. Requires sm_53+; bf16 variants require sm_90+. Introduced in PTX ISA 4.2.
