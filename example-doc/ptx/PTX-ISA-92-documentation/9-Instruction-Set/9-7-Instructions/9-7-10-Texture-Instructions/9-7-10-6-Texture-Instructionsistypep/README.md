# 9.7.10.6. Texture Instructions: istypep

The istypep instruction queries whether a .u64 register points to an opaque variable of a specified type (.texref, .samplerref, or .surfref). Writes 1 to a predicate register if the type matches, 0 otherwise. Requires sm_30 or higher.
