# 11.2.1. Kernel and Function Directives:.entry

Defines `.entry` directive declaring a kernel entry point with its name, optional parameter list (passed via .param space), and body. Parameters include regular typed parameters and opaque texref/samplerref/surfref references. CTA shape and size are available via special registers within the kernel body.
