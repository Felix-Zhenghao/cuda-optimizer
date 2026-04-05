# warpgroup submodule

*class* cutlass.cute.nvgpu.warpgroup.OperandMajorMode(*value*)
:   Bases: `Enum`

    An enumeration for the majorness of the input operands of the MMA.

*class* cutlass.cute.nvgpu.warpgroup.OperandSource(*value*)
:   Bases: `Enum`

    An enumeration for the source memory location of the A input operand of the MMA.

*class* cutlass.cute.nvgpu.warpgroup.Field(*value*)
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    ACCUMULATE *= 'accum\_c'*

*class* cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    F16/BF16 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

    descriptive\_name *= 'warpgroup F16/BF16 MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.warpgroup.MmaF8Op( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    F8 warpgroup MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
    This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

    descriptive\_name *= 'warpgroup F8 MMA Operation'*

    \_\_init\_\_( : *a\_dtype: Type[cutlass.cute.typing.Numeric]*, : *b\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource "cutlass.cute.nvgpu.warpgroup.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode "cutlass.cute.nvgpu.warpgroup.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind(*value*)
:   Bases: `Enum`

    Enum class for the kinds of SMEM layout atoms for SM90.

    Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
    be used to construct an SMEM layout using blocked product for operand A or B such that the
    resulting layout is legal for both TMA and UMMA.

    Note that there are other ways of creating legal layouts for operand A and B.

    MN\_INTER *= 1*

    MN\_SW32 *= 2*

    MN\_SW64 *= 3*

    MN\_SW128 *= 4*

    K\_INTER *= 5*

    K\_SW32 *= 6*

    K\_SW64 *= 7*

    K\_SW128 *= 8*

cutlass.cute.nvgpu.warpgroup.make\_smem\_layout\_atom( : *kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.mma.SmemLayoutAtomKind")*, : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.ComposedLayout
:   Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    Parameters:
    :   * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind "cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind")) -- The kind of layout Atom
        * **element\_type** (*Type**[**Numeric**]*) -- The element data type to construct the layout for

    Returns:
    :   The SMEM layout atom

    Return type:
    :   ComposedLayout

cutlass.cute.nvgpu.warpgroup.fence(*\**, *loc=None*, *ip=None*) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence).

cutlass.cute.nvgpu.warpgroup.commit\_group(*\**, *loc=None*, *ip=None*) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).

cutlass.cute.nvgpu.warpgroup.wait\_group(*group*, *\**, *loc=None*, *ip=None*) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group).
