# 9.7.9.19. Data Movement and Conversion Instructions: isspacep

The `isspacep` instruction queries whether a generic address falls within a specified state space window (.const, .global, .local, .shared, .param), writing a predicate result. Supports sub-qualifiers ::cta and ::cluster for shared memory and ::entry for param space. Note: isspacep.global returns true for kernel parameter addresses. Requires sm_20+. Introduced in PTX ISA 2.0.
