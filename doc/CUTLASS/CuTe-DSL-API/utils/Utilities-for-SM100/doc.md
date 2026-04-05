# Utilities for SM100

cutlass.utils.sm100.compute\_epilogue\_tile\_shape( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *use\_2cta\_instrs: bool*, : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *\**, : *layout\_c: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum") | None = None*, : *elem\_ty\_c: Type[cutlass.cutlass\_dsl.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Tile
:   Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) -- A tuple or list representing the dimensions of the CTA tile, where
          cta\_tile\_shape[0] corresponds to the height (M) and cta\_tile\_shape[1]
          corresponds to the width (N) of the tile.
        * **use\_2cta\_instrs** (*bool*) -- A flag indicating whether the configuration is for a 2SM setup.
        * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) -- The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) -- The element type of output tensor D.
        * **layout\_c** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")*,* *optional*) -- The layout enum of the input tensor C. Defaults to None.
        * **elem\_ty\_c** (*Union**[**Type**[**Numeric**]**,* *None**]**,* *optional*) -- The element type for input tensor C. Defaults to None.

    Returns:
    :   Returns epilog tiler, which is used in subsequent epilog partitions.

    Return type:
    :   cute.Tile

    Raises:
    :   **ValueError** -- If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

cutlass.utils.sm100.get\_smem\_store\_op( : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *tiled\_tmem\_load: [TiledCopy](cute.html#cutlass.cute.TiledCopy "cutlass.cute.atom.TiledCopy")*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM\_LOAD's thread-value ownership.

    Parameters:
    :   * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) -- The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) -- The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) -- The element type for accumulator.
        * **tiled\_tmem\_load** ([*cute.TiledCopy*](cute.html#cutlass.cute.TiledCopy "cutlass.cute.TiledCopy")) -- An instance of TiledCopy that represents the tmem load operation.

    Returns:
    :   Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

cutlass.utils.sm100.get\_tmem\_load\_op( : *cta\_tile\_shape: cutlass.cute.typing.Shape*, : *layout\_d: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *elem\_ty\_d: Type[cutlass.cutlass\_dsl.Numeric]*, : *elem\_ty\_acc: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_tile: cutlass.cute.typing.Tile*, : *use\_2cta\_instrs: bool*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")
:   Finds a performant TMEM\_LOAD copy op for the selected epilogue
    tile (epi\_tile), element types, and tcgen05.mma instruction used.

    Parameters:
    :   * **cta\_tile\_shape** (*cute.Shape*) -- A tuple or list representing the dimensions of the CTA tile.
        * **layout\_d** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) -- The layout enum of the output tensor D.
        * **elem\_ty\_d** (*Type**[**Numeric**]*) -- The element type for output tensor D.
        * **elem\_ty\_acc** (*Type**[**Numeric**]*) -- The element type for accumulation.
        * **epi\_tile** (*cute.Tile*) -- The epilogue tile configuration.
        * **use\_2cta\_instrs** (*bool*) -- A flag indicating whether the configuration is for 2 SMs.

    Returns:
    :   An instance of Sm100TmemLoad with the computed configuration.

    Return type:
    :   [cute.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")

    Raises:
    :   **ValueError** -- If the function cannot handle the given combination of accumulation
        and dimension types, or if it cannot determine the appropriate configuration based on
        the input parameters.

cutlass.utils.sm100.make\_smem\_layout\_a( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *a\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps with:

    1. Get the partitioned shape of the A tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) -- The tiled MMA used to partition tensor A
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) -- The MMA tile shape
        * **a\_dtype** (*Type**[**Numeric**]*) -- The element type for tensor A
        * **num\_stages** (*int*) -- The number of pipeline stages for tensor A

    Returns:
    :   SMEM layout for tensor A

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_smem\_layout\_b( : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *mma\_tiler\_mnk: cutlass.cute.typing.Tile*, : *b\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *num\_stages: int*, : *\**, : *is\_k\_major=None*, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps:

    1. Get the partitioned shape of the B tensor based on the tiled\_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **tiled\_mma** ([*cute.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")) -- The tiled MMA which is used to partition the B tensor.
        * **mma\_tiler\_mnk** (*cute.cute.Tile*) -- The MMA tile shape.
        * **b\_dtype** (*Type**[**Numeric**]*) -- The element type for the B tensor.
        * **num\_stages** (*int*) -- The stage of the B tensor.

    Returns:
    :   SMEM layout for the B tensor.

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_smem\_layout\_epi( : *epi\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *epi\_layout: [LayoutEnum](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.layout.LayoutEnum")*, : *epi\_tile: cutlass.cute.typing.Tile*, : *epi\_stage: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout
:   This function helps:

    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensor's majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    Parameters:
    :   * **epi\_dtype** (*Type**[**Numeric**]*) -- The element type for the epilog tensor.
        * **epi\_layout** ([*LayoutEnum*](utils.html#cutlass.utils.LayoutEnum "cutlass.utils.LayoutEnum")) -- The layout enum for the epilog tensor.
        * **epi\_tile** (*cute.cute.Tile*) -- The epilogue tile shape.
        * **epi\_stage** (*int*) -- The stage of the epilog tensor.

    Returns:
    :   SMEM layout for epilog tensors (usually C & D which are processed in the epilog)

    Return type:
    :   Union[cute.Layout, cute.ComposedLayout]

cutlass.utils.sm100.make\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *acc\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
:   Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) -- Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) -- Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) -- Leading dimension of operand B (1 for K, 0 for M/N).
        * **acc\_dtype** (*type**[**Numeric**]*) -- Data type of the accumulator.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) -- The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) -- The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) -- The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** -- If the data type is not supported.

cutlass.utils.sm100.make\_blockscaled\_trivial\_tiled\_mma( : *ab\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *a\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *b\_leading\_mode: [OperandMajorMode](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode")*, : *sf\_dtype: Type[cutlass.cutlass\_dsl.Numeric]*, : *sf\_vec\_size: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *mma\_tiler\_mn: Tuple[int, int]*, : *a\_source: [OperandSource](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.mma.OperandSource") = cutlass.\_mlir.dialects.cute.MmaFragKind.smem\_desc*, : *\**, : *loc=None*, : *ip=None*, ) → [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")
:   Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    Parameters:
    :   * **ab\_dtype** (*type**[**Numeric**]*) -- Data type of operands A and B.
        * **a\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) -- Leading dimension of operand A (1 for K, 0 for M/N).
        * **b\_leading\_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandMajorMode "cutlass.cute.nvgpu.tcgen05.OperandMajorMode")) -- Leading dimension of operand B (1 for K, 0 for M/N).
        * **sf\_dtype** (*type**[**Numeric**]*) -- Data type of the Scale Factor.
        * **sf\_vec\_size** (*int*) -- The vector size of the Scale Factor.
        * **cta\_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.CtaGroup")) -- The CTA group to use.
        * **mma\_tiler\_mn** (*Tuple**[**int**,* *int**]*) -- The shape (M, N, K) of the MMA tiler.
        * **a\_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.OperandSource "cutlass.cute.nvgpu.tcgen05.OperandSource")) -- The source of operand A (SMEM by default or TMEM).

    Returns:
    :   A tiled MMA atom.

    Return type:
    :   [cute.TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.TiledMma")

    Raises:
    :   **TypeError** -- If the data type is not supported.

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_A( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) -- The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) -- The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** -- If the atom\_sm\_cnt is invalid
        * **ValueError** -- If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_B( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) -- The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) -- The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** -- If the atom\_sm\_cnt is invalid
        * **ValueError** -- If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.cluster\_shape\_to\_tma\_atom\_SFB( : *cluster\_shape\_mnk: cutlass.cute.typing.Shape*, : *atom\_thr\_id: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, ) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp")
:   Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

    Parameters:
    :   * **cluster\_shape\_mnk** (*cute.Shape*) -- The shape of the cluster
        * **atom\_thr\_id** (*cute.Layout*) -- The thread ID of the atom

    Returns:
    :   The appropriate TMA copy atom kind

    Return type:
    :   [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp") or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")

    Raises:
    :   * **ValueError** -- If the atom\_sm\_cnt is invalid
        * **ValueError** -- If the cluster shape is not divisible by the atom SM count

cutlass.utils.sm100.get\_permutation\_mnk( : *tile\_shape\_mnk: cutlass.cute.typing.Shape*, : *sf\_vec\_size: int*, : *use\_mxf8f6f4: bool*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[int, int, int]
:   Get the permutation of M, N, K for the tiled MMA.

    Parameters:
    :   * **tile\_shape\_mnk** (*cute.Shape*) -- The shape of the tile
        * **sf\_vec\_size** (*int*) -- The vector size of the Scale Factor.
        * **use\_mxf8f6f4** (*bool*) -- Whether to use MXF8F6F4 or MXF4NVF4.

    Returns:
    :   The permutation of M, N, K

    Return type:
    :   Tuple[int, int, int]

    Raises:
    :   **ValueError** -- If the tile shape is not divisible by the sf\_vec\_size

cutlass.utils.sm100.get\_num\_tmem\_alloc\_cols( : *tmem\_tensors: cutlass.cute.typing.Tensor | List[cutlass.cute.typing.Tensor]*, : *rounding=True*, : *\**, : *loc=None*, : *ip=None*, ) → int
