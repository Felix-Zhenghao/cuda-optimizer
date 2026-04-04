# 9.7.9.20. Data Movement and Conversion Instructions: cvta

The `cvta` instruction converts addresses between state-space-specific (.const, .global, .local, .shared, .param) and generic forms. Can convert state-space addresses to generic, take generic address of declared variables, or convert generic back to state-space. Supports ::cta/::cluster sub-qualifiers for shared and ::entry for param. Requires sm_20+. Introduced in PTX ISA 2.0.
