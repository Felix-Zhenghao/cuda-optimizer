# cpasync submodule

*class* cutlass.cute.nvgpu.cpasync.LoadCacheMode(*value*)
:   Bases: `Enum`

    An enumeration for the possible cache modes of a non-bulk `cp.async` instruction.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

*class* cutlass.cute.nvgpu.cpasync.CopyG2SOp( : *cache\_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode "cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode") = cutlass.\_mlir.dialects.cute\_nvgpu.LoadCacheMode.always*, )
:   Bases: `CopyOp`

    Non-bulk asynchronous GMEM to SMEM Copy Operation.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy).

    \_\_init\_\_( : *cache\_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode "cutlass.cute.nvgpu.cpasync.copy.LoadCacheMode") = cutlass.\_mlir.dialects.cute\_nvgpu.LoadCacheMode.always*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )
:   Bases: `TmaCopyOp`

    Bulk tensor asynchrnous GMEM to SMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")* *= 1*

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, )
:   Bases: `TmaCopyOp`

    Bulk tensor asynchrnous multicast GMEM to SMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")* *= 1*

    \_\_init\_\_( : *cta\_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup = <CtaGroup.ONE>*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp
:   Bases: `TmaCopyOp`

    Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
    This Operation uses TMA in the `.tile` mode.

    \_\_init\_\_() → None

*class* cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp( : *reduction\_kind: cutlass.\_mlir.dialects.cute.ReductionOp = cutlass.\_mlir.dialects.cute.ReductionOp.ADD*, )
:   Bases: `TmaCopyOp`

    Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk).
    This Operation uses TMA in the `.tile` mode.

    \_\_init\_\_( : *reduction\_kind: cutlass.\_mlir.dialects.cute.ReductionOp = cutlass.\_mlir.dialects.cute.ReductionOp.ADD*, ) → None

*class* cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp
:   Bases: `CopyOp`

    Asynchronous Store operation to DSMEM with explicit synchronization.

    See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async).

    \_\_init\_\_() → None

cutlass.cute.nvgpu.cpasync.make\_tiled\_tma\_atom( : *op: [CopyBulkTensorTileG2SOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp") | [CopyBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileS2GOp") | [CopyReduceBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp "cutlass.cute.nvgpu.cpasync.copy.CopyReduceBulkTensorTileS2GOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout\_: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *cta\_tiler: cutlass.cute.typing.Tiler*, : *num\_multicast: int = 1*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor]
:   Makes a TMA Copy Atom in the `.tile` mode to copy tiles of a GMEM tensor to/from SMEM
    buffer with the given Layout.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a CTA-level Tiler

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
    layout while maintaining consistency with the provided Tiler.

    This function returns two results:

    1. the Copy Atom
    2. a TMA tensor that maps logical coordinates of the GMEM tensor to coordinates consumed by the TMA unit. TMA tensors contain basis stride elements that enable their associated layout to compute coordinates. Like other CuTe tensors, TMA tensors can be partitioned.

    Parameters:
    :   * **op** (*TMAOp*) -- The TMA Copy Operation to construct an Atom
        * **gmem\_tensor** (*Tensor*) -- The GMEM tensor involved in the Copy
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) -- The SMEM layout to construct the Copy Atom, either w/ or w/o the stage mode
        * **cta\_tiler** (*Tiler*) -- The CTA Tiler to use
        * **num\_multicast** (*int*) -- The multicast factor
        * **internal\_type** (*Type**[**Numeric**]*) -- Optional internal data type to use when the tensor data type is not supported by the TMA unit

    Returns:
    :   A TMA Copy Atom associated with the TMA tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

cutlass.cute.nvgpu.cpasync.tma\_partition( : *atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *cta\_coord: cutlass.cute.typing.Coord*, : *cta\_layout: cutlass.cute.typing.Layout*, : *smem\_tensor: cutlass.cute.typing.Tensor*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *\**, : *loc=None*, : *ip=None*, ) → Tuple[cutlass.cute.typing.Tensor, cutlass.cute.typing.Tensor]
:   Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.

cutlass.cute.nvgpu.cpasync.create\_tma\_multicast\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *cta\_coord\_vmnk: cutlass.cute.typing.Coord*, : *mcast\_mode: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Int16
:   Computes a multicast mask for a TMA load Copy.

    Parameters:
    :   * **cta\_layout\_vmnk** (*Layout*) -- The VMNK layout of the cluster
        * **cta\_coord\_vmnk** (*Coord*) -- The VMNK coordinate of the current CTA
        * **mcast\_mode** (*int*) -- The tensor mode in which to multicast

    Returns:
    :   The resulting mask

    Return type:
    :   Int16

cutlass.cute.nvgpu.cpasync.prefetch\_descriptor( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Prefetches the TMA descriptor associated with the TMA Atom.

cutlass.cute.nvgpu.cpasync.copy\_tensormap( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *tensormap\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
    pointer.

    Parameters:
    :   * **tma\_atom** ([*CopyAtom*](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) -- The TMA Copy Atom
        * **tensormap\_ptr** (*Pointer*) -- The pointer to the memory location to copy the tensormap to

cutlass.cute.nvgpu.cpasync.update\_tma\_descriptor( : *tma\_atom: [CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   Updates the TMA descriptor in the memory location pointed to by the provided pointer using
    information from a TMA Copy Atom and the provided GMEM tensor.

    Specifically, the following fields of the TMA descriptor will be updated:

    1. the GMEM tensor base address
    2. the GMEM tensor shape
    3. the GMEM tensor stride

    Other fields of the TMA descriptor are left unchanged.

    Parameters:
    :   * **tma\_atom** ([*CopyAtom*](cute.html#cutlass.cute.CopyAtom "cutlass.cute.CopyAtom")) -- The TMA Copy Atom
        * **gmem\_tensor** (*Tensor*) -- The GMEM tensor
        * **tensormap\_ptr** (*Pointer*) -- The pointer to the memory location of the descriptor to udpate

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_acquire( : *tma\_desc\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

cutlass.cute.nvgpu.cpasync.cp\_fence\_tma\_desc\_release( : *tma\_desc\_global\_ptr: cutlass.cute.typing.Pointer*, : *tma\_desc\_shared\_ptr: cutlass.cute.typing.Pointer*, : *\**, : *loc=None*, : *ip=None*, ) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy).

cutlass.cute.nvgpu.cpasync.fence\_tma\_desc\_release(*\**, *loc=None*, *ip=None*) → None
:   See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).
