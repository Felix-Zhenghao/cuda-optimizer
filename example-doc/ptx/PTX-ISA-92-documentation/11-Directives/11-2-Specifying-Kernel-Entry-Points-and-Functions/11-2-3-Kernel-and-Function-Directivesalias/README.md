# 11.2.3: Kernel and Function Directives: .alias

Defines an alias from one function symbol to another within the same module. Both alias and aliasee must be non-entry functions with matching prototypes. The aliasee cannot have .weak linkage. Programs can use either identifier to call the function. Introduced in PTX ISA 6.3, requires sm_30+.
