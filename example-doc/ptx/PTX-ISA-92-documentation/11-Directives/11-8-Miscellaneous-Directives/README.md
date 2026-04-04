# 11.8: Miscellaneous Directives: .blocksareclusters

Documents .blocksareclusters directive that remaps CUDA grid launch configuration to specify cluster counts instead of thread block counts. Requires .reqntid and .reqnctapercluster to also be specified. Only valid for .entry functions. Introduced in PTX ISA 9.0, requires sm_90 or higher.
