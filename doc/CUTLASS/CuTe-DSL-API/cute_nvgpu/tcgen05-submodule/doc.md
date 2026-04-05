# tcgen05 submodule

*class* cutlass.cute.nvgpu.tcgen05.Repetition(*value*)
:   Bases: `Enum`

    An enumeration for the number of repetitions of a given TMEM copy within the instruction.

    x1 *= 1*

    x2 *= 2*

    x4 *= 4*

    x8 *= 8*

    x16 *= 16*

    x32 *= 32*

    x64 *= 64*

    x128 *= 128*

*class* cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp(*value*)
:   Bases: `Enum`

    An enumeration for the possible reduce operations for TMEM load operations.

*class* cutlass.cute.nvgpu.tcgen05.Pack(*value*)
:   Bases: `Enum`

    An enumeration for the possible packing patterns for TMEM to RMEM copies.

    NONE *= 1*

    PACK\_16b\_IN\_32b *= 2*

*class* cutlass.cute.nvgpu.tcgen05.Unpack(*value*)
:   Bases: `Enum`

    An enumeration for the possible unpacking patterns for RMEM to TMEM copies.

    NONE *= 1*

    UNPACK\_32b\_IN\_16b *= 2*

*class* cutlass.cute.nvgpu.tcgen05.Ld16x64bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )
:   Bases: `_LdBase`

    16x64b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x64b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.Ld16x128bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )
:   Bases: `_LdBase`

    16x128b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x128b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.Ld16x256bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )
:   Bases: `_LdBase`

    16x256b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x256b` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )
:   Bases: `_LdBase`

    16x32bx2 TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.16x32bx2` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.Ld32x32bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, )
:   Bases: `_LdBase`

    32x32b TMEM load Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
    This Operation corresponds to the `.32x32` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition = <Repetition.x1>*, : *pack: ~cutlass.cute.nvgpu.tcgen05.copy.Pack = <Pack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.St16x64bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )
:   Bases: `_StBase`

    16x64b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x64` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.St16x128bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )
:   Bases: `_StBase`

    16x128b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x128` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.St16x256bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )
:   Bases: `_StBase`

    16x256b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x256` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.St16x32bx2Op( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )
:   Bases: `_StBase`

    16x32x2b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.16x32x2` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.St32x32bOp( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, )
:   Bases: `_StBase`

    32x32b TMEM store Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
    This Operation corresponds to the `.32x32` qualifier.

    \_\_init\_\_( : *repeat: ~cutlass.cute.nvgpu.tcgen05.copy.Repetition*, : *unpack: ~cutlass.cute.nvgpu.tcgen05.copy.Unpack = <Unpack.NONE>*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.OperandMajorMode(*value*)
:   Bases: `Enum`

    An enumeration for the majorness of the input operands of the MMA.

*class* cutlass.cute.nvgpu.tcgen05.OperandSource(*value*)
:   Bases: `Enum`

    An enumeration for the source memory location of the A input operand of the MMA.

*class* cutlass.cute.nvgpu.tcgen05.CtaGroup(*value*)
:   Bases: `Enum`

    An enumeration for the `cta_group` qualifier of the MMA.

    ONE *= 1*

    TWO *= 2*

*class* cutlass.cute.nvgpu.tcgen05.Field(*value*)
:   Bases: `Enum`

    An enumeration for the fields of the MMA Atom that can be modified at runtime.

    NEGATE\_A *= 'neg\_a'*

    NEGATE\_B *= 'neg\_b'*

    ACCUMULATE *= 'accum\_c'*

    SFA *= 'sf\_a'*

    SFB *= 'sf\_b'*

*class* cutlass.cute.nvgpu.tcgen05.MmaTF32Op( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    TF32 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::tf32` qualifier.

    descriptive\_name *= 'tcgen05 TF32 MMA Operation'*

    \_\_init\_\_( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    F16/BF16 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::f16` qualifier.

    descriptive\_name *= 'tcgen05 F16/BF16 MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaI8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    I8 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::i8` qualifier.

    descriptive\_name *= 'tcgen05 I8 MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaFP8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )
:   Bases: `MmaOp`

    F8 tcgen05 MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).

    descriptive\_name *= 'tcgen05 F8 MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *acc\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF8Op( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, )
:   Bases: `BlockScaledMmaOp`

    MXF8 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

    descriptive\_name *= 'tcgen05 MXF8 BlockScaled MMA Operation'*

    \_\_init\_\_( : *ab\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, : *a\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_major\_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF4Op( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, )
:   Bases: `BlockScaledMmaOp`

    MXF4 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf4` qualifier.

    descriptive\_name *= 'tcgen05 MXF4 BlockScaled MMA Operation'*

    \_\_init\_\_( : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op( : *sf\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, )
:   Bases: `BlockScaledMmaOp`

    MXF4NVF4 tcgen05 BlockScaled MMA Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
    This Operation corresponds to the `.kind::mxf4nvf4` qualifier.

    descriptive\_name *= 'tcgen05 MXF4NVF4 BlockScaled MMA Operation'*

    \_\_init\_\_( : *sf\_dtype: Type[cutlass.cute.typing.Numeric]*, : *instruction\_shape: cutlass.cute.typing.Shape*, : *cta\_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *a\_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource")*, ) → None

*class* cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind(*value*)
:   Bases: `Enum`

    Enum class for the kinds of SMEM layout atoms for SM100.

    Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be
    used to construct an SMEM layout using blocked product for operand A or B such that the
    resulting layout is legal for both TMA and UMMA.

    Note that there are other ways of creating legal layouts for operand A and B.

    MN\_INTER *= 1*

    MN\_SW32 *= 2*

    MN\_SW64 *= 3*

    MN\_SW128 *= 4*

    MN\_SW128\_32B *= 5*

    K\_INTER *= 6*

    K\_SW32 *= 7*

    K\_SW64 *= 8*

    K\_SW128 *= 9*

cutlass.cute.nvgpu.tcgen05.make\_smem\_layout\_atom( : *kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.mma.SmemLayoutAtomKind")*, : *element\_type: Type[cutlass.cute.typing.Numeric]*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.ComposedLayout
:   Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    Parameters:
    :   * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind "cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind")) -- The kind of layout Atom
        * **element\_type** (*Type**[**Numeric**]*) -- The element data type to construct the layout for

    Returns:
    :   The SMEM layout atom

    Return type:
    :   ComposedLayout

cutlass.cute.nvgpu.tcgen05.tile\_to\_mma\_shape( : *atom*, : *mma\_tile\_shape: cutlass.cute.typing.Shape*, : *order: cutlass.cute.typing.IntTuple | None = None*, : *\**, : *loc=None*, : *ip=None*, )
:   Tiles a layout to an MMA shape.

cutlass.cute.nvgpu.tcgen05.commit( : *mbar\_ptr: cutlass.cute.typing.Pointer*, : *mask=None*, : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

    Parameters:
    :   * **mbar\_ptr** (*Pointer*) -- A pointer to the mbarrier in SMEM
        * **mask** (*Int*) -- An optional multicast mask for the CTAs in the cluster to signal arrival to

cutlass.cute.nvgpu.tcgen05.is\_tmem\_load(*atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*) → bool
:   Returns whether a CopyAtom instance is a TMEM load.

cutlass.cute.nvgpu.tcgen05.is\_tmem\_store(*atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*) → bool
:   Returns whether a CopyAtom instance is a TMEM store.

cutlass.cute.nvgpu.tcgen05.get\_tmem\_copy\_properties( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, ) → Tuple[int, int, int, [Pack](#cutlass.cute.nvgpu.tcgen05.Pack "cutlass.cute.nvgpu.tcgen05.copy.Pack") | [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack "cutlass.cute.nvgpu.tcgen05.copy.Unpack")]
:   Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
    and whether packing/unpacking is used).

cutlass.cute.nvgpu.tcgen05.find\_tmem\_tensor\_col\_offset( : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int
:   Computes the TMEM column offset given a TMEM tensor.

    Parameters:
    :   **tmem\_tensor** (*Tensor*) -- The TMEM tensor to use to compute the columns offset

    Returns:
    :   The columns offset

    Return type:
    :   Int

cutlass.cute.nvgpu.tcgen05.make\_tmem\_copy( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")
:   Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

cutlass.cute.nvgpu.tcgen05.make\_s2t\_copy( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")
:   Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

cutlass.cute.nvgpu.tcgen05.get\_s2t\_smem\_desc\_tensor( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *smem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tensor
:   Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

cutlass.cute.nvgpu.tcgen05.make\_umma\_smem\_desc( : *src: cutlass.cute.typing.Pointer*, : *layout: cutlass.cute.typing.Layout*, : *major: str*, : *next\_src: cutlass.cute.typing.Pointer | None = None*, : *\**, : *loc=None*, : *ip=None*, )
:   Construct shared memory descriptor for UMMA.

    The make\_umma\_smem\_desc operation accepts an input cute.ptr (optionally a nextSrc
    pointer for the second buffer in a circular buffer scheme), alongside a cute.layout
    and a major attr, then constructs the shared memory descriptor and returns it.
    The layout must be describing the buffer pointed to by the input pointer and the
    iterator must carry valid swizzle information.

    There are 5 supported swizzle variants:
    - S<0, 4, 3> | SWIZZLE\_NONE
    - S<1, 4, 3> | SWIZZLE\_32B
    - S<2, 4, 3> | SWIZZLE\_64B
    - S<3, 4, 3> | SWIZZLE\_128B
    - S<2, 5, 2> | SWIZZLE\_128B\_BASE32B

    The cute.ptr must carry shared address space and must be aligned to 16B.

    Parameters:
    :   * **src** (*Pointer*) -- The source pointer to shared memory
        * **layout** (*Layout*) -- The layout describing the buffer
        * **major** (*str*) -- The major mode attribute
        * **next\_src** (*Optional**[**Pointer**]*) -- Optional next source pointer for circular buffer scheme

    Returns:
    :   The shared memory descriptor

    Return type:
    :   SmemDescType
