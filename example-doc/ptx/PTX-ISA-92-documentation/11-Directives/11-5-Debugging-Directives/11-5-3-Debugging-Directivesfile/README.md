# 11.5.3: Debugging Directives: .file

Associates a source filename with an integer index for use by .loc directives. Optionally specifies file timestamp (time_t format) and file size. Only allowed at outermost scope alongside kernel and function declarations. Introduced in PTX ISA 1.0, timestamp/size support added in version 3.2.
