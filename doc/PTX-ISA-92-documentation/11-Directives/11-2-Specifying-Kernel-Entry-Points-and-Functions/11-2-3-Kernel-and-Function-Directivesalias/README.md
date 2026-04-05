# 11.2.3. Kernel and Function Directives:.alias

Defines `.alias` module-scope directive creating an alternate name for an existing non-entry function symbol within the same module. The aliasee cannot have `.weak` linkage, and both alias and aliasee must have matching prototypes. Either name can be used to reference the function.
