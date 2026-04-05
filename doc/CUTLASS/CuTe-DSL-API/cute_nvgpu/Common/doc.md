# Common

*class* cutlass.cute.nvgpu.OpError(*\*args: Any*, *\*\*kwargs: Any*)
:   Bases: `DSLBaseError`

    An exception class for Op construction errors.

cutlass.cute.nvgpu.normalize\_field\_to\_ir\_name(*field*, *admissible\_fields*) → str
:   Normalize a field specifier to its IR logical field name.

    Accepted inputs:

    * Enum value present in admissible\_fields (must expose \_to\_ir\_field\_name()).
    * Exact string IR name (e.g., "accum\_c", "neg\_a", "sf\_a").

    Any other form is rejected.

*class* cutlass.cute.nvgpu.MmaUniversalOp(*abacc\_dtype: Type[cutlass.cute.typing.Numeric]*)
:   Bases: `MmaOp`

    The universal MMA Operation.

    This Operation currently expects the A/B operands as well as the accumulator to share the same
    data types.

    Parameters:
    :   **abacc\_dtype** (*Type**[**Numeric**]*) -- The data type for the A/B operands and the accumulator

    abacc\_dtype*: Type[cutlass.cute.typing.Numeric]*

*class* cutlass.cute.nvgpu.MmaUniversalTrait(*value: cutlass.\_mlir.ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.CopyUniversalOp
:   Bases: `CopyOp`

    The universal Copy Operation.

    When creating a Copy Atom out of this operation, the expected usage pattern is

    ```
    op = cute.nvgpu.CopyUniversalOp()
    atom = cute.make_copy_atom(
        op,
        tensor_dtype,
        num_bits_per_copy=64,
        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL
    )
    ```

    * `tensor_dtype` is the data type used to build the reference TV Layout (either the source or the destination TV Layout) in unit of tensor elements and is used for partitioning by `TiledCopy` for example
    * `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom execution. This can be larger than the width of the above data type. When not provided, the compiler will do a best effort at auto-vectorizing.
    * `l1c_evict_priority` is a kw argument specifying the L1 cache eviction priority hint for the copy operation. Defaults to `EVICT_NORMAL` if not provided.
    * `invariant` is a kw argument specifying whether the load is invariant (read-only data that never changes). This enables compiler optimizations like instruction reordering. Defaults to `False` if not provided.

*class* cutlass.cute.nvgpu.CopyUniversalTrait(*value: cutlass.\_mlir.ir.Value*)
:   Bases: `Trait`

*class* cutlass.cute.nvgpu.MemoryOrder(*value*)
:   Bases: `Enum`

    An enumeration.

*class* cutlass.cute.nvgpu.MemoryScope(*value*)
:   Bases: `Enum`

    An enumeration.

*class* cutlass.cute.nvgpu.CacheEvictionPriority(*value*)
:   Bases: `Enum`

    An enumeration.

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_A( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor]
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the MK projections of the TiledMMA for A tensor loads.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a MMA Tiler
    * a TiledMma
    * a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the M-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   * **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) -- The Copy Operation to construct an Atom for
        * **gmem\_tensor** (*Tensor*) -- The GMEM tensor to be loaded by this copy atom
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) -- Shared memory layout to load the tensor into (PDSL)
        * **mma\_tiler\_mnk** (*Shape*) -- The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        * **tiled\_mma** ([*atom.TiledMma*](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")) -- The TiledMMA that will consume the load as operands
        * **cluster\_shape\_vmnk** (*Shape*) -- The Cluster-level shape in VMNK dimensions
        * **internal\_type** (*Type**[**Numeric**]*) -- An optional parameter for the internal data type to when element
          type does not match the copy type

    Returns:
    :   A copy atom for this operation and the associated TMA coord tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]

cutlass.cute.nvgpu.make\_tiled\_tma\_atom\_B( : *op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SOp") | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.copy.CopyBulkTensorTileG2SMulticastOp")*, : *gmem\_tensor: cutlass.cute.typing.Tensor*, : *smem\_layout: cutlass.cute.typing.Layout | cutlass.cute.typing.ComposedLayout*, : *mma\_tiler\_mnk: cutlass.cute.typing.Shape*, : *tiled\_mma: [TiledMma](cute.html#cutlass.cute.TiledMma "cutlass.cute.atom.TiledMma")*, : *cluster\_shape\_vmnk: cutlass.cute.typing.Shape | None = None*, : *\**, : *internal\_type: Type[cutlass.cute.typing.Numeric] | None = None*, : *loc=None*, : *ip=None*, ) → Tuple[[CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), cutlass.cute.typing.Tensor]
:   Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
    accounting for the NK projections of the TiledMMA for B tensor loads.

    Given

    * a GMEM tensor
    * a SMEM layout
    * a MMA Tiler
    * a TiledMma
    * a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled\_mma (considering the N-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    Parameters:
    :   * **op** (*Union**[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp")*,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp "cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp")*]*) -- The Copy Operation to construct an Atom for
        * **gmem\_tensor** (*Tensor*) -- The GMEM tensor to be loaded by this copy atom
        * **smem\_layout** (*Union**[**Layout**,* *ComposedLayout**]*) -- Shared memory layout to load the tensor into (PDSL)
        * **mma\_tiler\_mnk** (*Shape*) -- The MMA Tiler shape (TILE\_M, TILE\_N, TILE\_K) in MNK dimensions
        * **tiled\_mma** (*core.TiledMma*) -- The TiledMMA that will consume the load as operands
        * **cluster\_shape\_vmnk** (*Shape*) -- The Cluster-level shape in VMNK dimensions
        * **internal\_type** (*Type**[**Numeric**]*) -- An optional parameter for the internal data type to when element
          type does not match the copy type

    Returns:
    :   A Copy Atom for this Operation and the associated TMA tensor

    Return type:
    :   Tuple[[atom.CopyAtom](cute.html#cutlass.cute.CopyAtom "cutlass.cute.atom.CopyAtom"), Tensor]
