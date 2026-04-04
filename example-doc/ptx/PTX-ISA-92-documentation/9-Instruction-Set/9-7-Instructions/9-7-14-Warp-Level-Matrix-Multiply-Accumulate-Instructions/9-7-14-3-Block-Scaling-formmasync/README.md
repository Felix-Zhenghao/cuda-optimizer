# 9.7.14.3. Block Scaling for mma.sync

Describes block scaling for `mma` instructions with `.kind::mxf8f6f4`, `.kind::mxf4`, and `.kind::mxf4nvf4` qualifiers. The operation computes `D = (A * scale_A) * (B * scale_B) + C`. Defines scale factor data types (ue8m0, ue4m3), per-thread scale register layouts, and supported type/shape combinations per PTX ISA version.
