# 9.7.6.2. Comparison and Selection Instructions: setp

The `setp` instruction compares two numeric values with a relational operator and optionally combines the result with a predicate via a Boolean operator, writing the result to predicate destination `p` and its complement to optional destination `q`. Supports all numeric types, ordered/unordered floating-point comparisons, and NaN handling. Introduced in PTX ISA 1.0.
