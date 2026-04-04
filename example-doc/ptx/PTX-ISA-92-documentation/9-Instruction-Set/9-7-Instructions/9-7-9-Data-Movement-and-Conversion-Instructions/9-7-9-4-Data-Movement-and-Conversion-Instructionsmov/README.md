# 9.7.9.4. Data Movement and Conversion Instructions: mov (vector pack/unpack)

The vector form of `mov` packs vector register elements into a scalar register or unpacks a scalar register into vector elements. Supports .b16, .b32, .b64, and .b128 types. Can pack/unpack 2- or 4-element vectors of various sub-widths. The sink symbol '_' may be used for unused vector elements. Introduced in PTX ISA 1.0.
