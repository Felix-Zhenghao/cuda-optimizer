*class* cutlass.pipeline.PipelineCpAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineCpAsync is used for CpAsync producers and AsyncThread consumers

    *static* create( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: cutlass.cutlass\_dsl.Int32*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *defer\_sync: bool = False*, )
    :   Helper function that computes necessary attributes and returns a `PipelineCpAsync` instance.

        Parameters:
        :   * **barrier\_storage** (*cute.Pointer*) -- Pointer to the shared memory address for this pipeline's mbarriers
            * **num\_stages** (*Int32*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the consumer agent
            * **producer\_mask** (*Int32**,* *optional*) -- Mask for signaling arrives for the producer agent, defaults to None
            * **consumer\_mask** (*Int32**,* *optional*) -- Mask for signaling arrives for the consumer agent, defaults to None

        Returns:
        :   A new `PipelineCpAsync` instance configured with the provided parameters

        Return type:
        :   [PipelineCpAsync](#cutlass.pipeline.PipelineCpAsync "cutlass.pipeline.PipelineCpAsync")

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None

*class* cutlass.pipeline.PipelineTmaAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).

    is\_signalling\_thread*: cutlass.cutlass\_dsl.Boolean*

    *static* init\_empty\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: cutlass.cutlass\_dsl.Int32*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, )
    :   Initialize the empty barrier arrive signal.

        This function determines which threads should signal empty barrier arrives based on the cluster layout
        and multicast modes. It returns the destination CTA rank and whether the current thread should signal.

        Parameters:
        :   * **cta\_layout\_vmnk** (*cute.Layout*) -- Layout describing the cluster shape and CTA arrangement
            * **tidx** (*Int32*) -- Thread index within the warp
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]*) -- Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

        Raises:
        :   **AssertionError** -- If both multicast modes are disabled (0,0)

        Returns:
        :   Tuple containing destination CTA rank and boolean indicating if current thread signals

        Return type:
        :   tuple[Int32, Boolean]

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *tidx: cutlass.cutlass\_dsl.Int32 | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *defer\_sync: bool = False*, )
    :   Create a new `PipelineTmaAsync` instance.

        Parameters:
        :   * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the consumer agent
            * **tx\_count** (*int*) -- Number of bytes expected to be written to the transaction barrier for one stage
            * **barrier\_storage** (*cute.Pointer**,* *optional*) -- Pointer to the shared memory address for this pipeline's mbarriers, defaults to None
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) -- Layout of the cluster shape, defaults to None
            * **tidx** (*Int32**,* *optional*) -- Thread index to consumer async threads, defaults to None
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) -- Tuple specifying multicast modes for m and n dimensions (each 0 or 1), defaults to (1,1)

        Raises:
        :   **ValueError** -- If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New `PipelineTmaAsync` instance

        Return type:
        :   [PipelineTmaAsync](#cutlass.pipeline.PipelineTmaAsync "cutlass.pipeline.PipelineTmaAsync")

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA consumer release conditionally signals the empty buffer to the producer.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, ) → None

*class* cutlass.pipeline.PipelineTmaUmma( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).

    is\_leader\_cta*: bool*

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *\**, : *loc=None*, : *ip=None*, ) → [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")
    :   Returns a SyncObject corresponding to an agent's PipelineOp.

    \_compute\_mcast\_arrival\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *mcast\_mode\_mn: tuple[int, int]*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *mcast\_mode\_mn: tuple[int, int] = (1, 1)*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )
    :   Creates and initializes a new PipelineTmaUmma instance.

        Parameters:
        :   * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- CooperativeGroup for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- CooperativeGroup for the consumer agent
            * **tx\_count** (*int*) -- Number of bytes expected to be written to the transaction barrier for one stage
            * **barrier\_storage** (*cute.Pointer**,* *optional*) -- Pointer to the shared memory address for this pipeline's mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) -- Layout of the cluster shape
            * **mcast\_mode\_mn** (*tuple**[**int**,* *int**]**,* *optional*) -- Tuple specifying multicast modes for m and n dimensions (each 0 or 1)

        Raises:
        :   **ValueError** -- If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineTmaUmma instance configured with the provided parameters

        Return type:
        :   [PipelineTmaUmma](#cutlass.pipeline.PipelineTmaUmma "cutlass.pipeline.PipelineTmaUmma")

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineAsyncUmma( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineAsyncUmma is used for AsyncThread producers and UMMA consumers (e.g. Blackwell input fusion pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    \_compute\_leading\_cta\_rank( : *cta\_v\_size*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes the leading CTA rank.

    \_compute\_is\_leader\_cta( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes leader threadblocks for 2CTA kernels. For 1CTA, all threadblocks are leaders.

    \_compute\_peer\_cta\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes a mask for signaling arrivals to multicasting threadblocks.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )
    :   Creates and initializes a new PipelineAsyncUmma instance.

        Parameters:
        :   * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- CooperativeGroup for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- CooperativeGroup for the consumer agent
            * **barrier\_storage** (*cute.Pointer**,* *optional*) -- Pointer to the shared memory address for this pipeline's mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) -- Layout of the cluster shape

        Raises:
        :   **ValueError** -- If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new PipelineAsyncUmma instance configured with the provided parameters

        Return type:
        :   [PipelineAsyncUmma](#cutlass.pipeline.PipelineAsyncUmma "cutlass.pipeline.PipelineAsyncUmma")

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   UMMA consumer release buffer empty, cta\_group needs to be provided.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineUmmaAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    \_compute\_tmem\_sync\_mask( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *\**, : *loc=None*, : *ip=None*, )
    :   Computes a mask to signal completion of tmem buffers for 2CTA kernels.

    \_compute\_peer\_cta\_rank(*\**, *loc=None*, *ip=None*)
    :   Computes a mask to signal release of tmem buffers for 2CTA kernels.

    create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, : *loc=None*, : *ip=None*, )
    :   Creates an instance of PipelineUmmaAsync with computed attributes.

        Parameters:
        :   * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the consumer agent
            * **barrier\_storage** (*cute.Pointer**,* *optional*) -- Pointer to the shared memory address for this pipeline's mbarriers
            * **cta\_layout\_vmnk** (*cute.Layout**,* *optional*) -- Layout of the cluster shape

        Raises:
        :   **ValueError** -- If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   New instance of `PipelineUmmaAsync`

        Return type:
        :   [PipelineUmmaAsync](#cutlass.pipeline.PipelineUmmaAsync "cutlass.pipeline.PipelineUmmaAsync")

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   UMMA producer commit buffer full, cta\_group needs to be provided.

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineClcFetchAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, )
:   Bases: `object`

    PipelineClcFetchAsync implements a producer-consumer pipeline for Cluster Launch
    Control based dynamic scheduling. Both producer and consumer operate asynchronously
    using barrier synchronization to coordinate across pipeline stages and cluster CTAs.

    * Producer: waits for empty buffer, signals full barrier with transection bytes
      across all CTAs in cluster, hardware autosignals each CTA's mbarrier when
      transaction bytes are written, then the satte advance to next buffer slot.
    * Consumer: waits for full barrier, then load respinse from local SMEM, then
      sigals CTA 0's empty barrier to allow buffer reuse.

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    num\_stages*: int*

    producer\_mask*: cutlass.cutlass\_dsl.Int32 | None*

    consumer\_mask*: cutlass.cutlass\_dsl.Int32 | None*

    is\_signalling\_thread*: cutlass.cutlass\_dsl.Boolean*

    *static* \_init\_full\_barrier\_arrive\_signal( : *cta\_layout\_vmnk: cutlass.cute.typing.Layout*, : *tidx: cutlass.cutlass\_dsl.Int32*, )
    :   Computes producer barrier signaling parameters, returns destination CTA rank
        (0 to cluster\_size-1) based on thread ID, and a boolean flag indicating if
        this thread participates in signaling.

        Parameters:
        :   * **cta\_layout\_vmnk** -- Cluster layout defining CTA count
            * **tidx** -- Thread ID within the CTA

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, )
    :   This helper function computes any necessary attributes and returns an instance of PipelineClcFetchAsync.
        :param barrier\_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: int
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group: CooperativeGroup for the consumer agent
        :type consumer\_group: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param producer\_mask: Mask for signaling arrives for the producer agent, defaults to `None`
        :type producer\_mask: Int32, optional
        :param consumer\_mask: Mask for signaling arrives for the consumer agent, defaults to `None`
        :type consumer\_mask: Int32, optional

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   Producer acquire waits for empty buffer and sets transaction expectation on full barrier.

        Parameters:
        :   * **state** -- Pipeline state pointing to the current buffer stage
            * **try\_acquire\_token** -- Optional token to skip the empty barrier wait

    consumer\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   Consumer waits for full barrier to be signaled by hardware multicast.

        Parameters:
        :   * **state** -- Pipeline state pointing to the current buffer stage
            * **try\_wait\_token** -- Optional token to skip the full barrier wait

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )

    producer\_get\_barrier( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer

    producer\_tail( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   Ensures all in-flight buffers are released before producer exits.

        Parameters:
        :   * **state** -- Pipeline state with current position in the buffer
            * **try\_acquire\_token** -- Optional token to skip the empty barrier waits

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_signalling\_thread: cutlass.cutlass\_dsl.Boolean*, ) → None

*class* cutlass.pipeline.PipelineTmaMultiConsumersAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.

    is\_leader\_cta*: bool*

    sync\_object\_empty\_umma*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty\_async*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    cta\_group*: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_umma: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group\_async: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *tx\_count: int*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *cta\_layout\_vmnk: cutlass.cute.typing.Layout | None = None*, : *defer\_sync: bool = False*, )
    :   This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
        :param barrier\_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier\_storage: cute.Pointer
        :param num\_stages: Number of buffer stages for this pipeline
        :type num\_stages: Int32
        :param producer\_group: CooperativeGroup for the producer agent
        :type producer\_group: CooperativeGroup
        :param consumer\_group\_umma: CooperativeGroup for the UMMA consumer agent
        :type consumer\_group\_umma: CooperativeGroup
        :param consumer\_group\_async: CooperativeGroup for the AsyncThread consumer agent
        :type consumer\_group\_async: CooperativeGroup
        :param tx\_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx\_count: int
        :param cta\_layout\_vmnk: Layout of the cluster shape
        :type cta\_layout\_vmnk: cute.Layout | None

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   TMA producer commit is a noop since TMA instruction itself updates the transaction count.

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *op\_type: [PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")*, : *\**, : *loc=None*, : *ip=None*, )

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *is\_leader\_cta: bool*, : *sync\_object\_empty\_umma: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty\_async: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, ) → None

*class* cutlass.pipeline.PipelineTmaStore( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )
:   Bases: [`PipelineAsync`](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")

    PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
    :   This helper function computes any necessary attributes and returns an instance of `PipelineTmaStore`.

        Parameters:
        :   * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the producer agent

        Returns:
        :   A new `PipelineTmaStore` instance

        Return type:
        :   [PipelineTmaStore](#cutlass.pipeline.PipelineTmaStore "cutlass.pipeline.PipelineTmaStore")

    producer\_acquire(*\**, *loc=None*, *ip=None*)

    producer\_commit(*\**, *loc=None*, *ip=None*)

    consumer\_wait(*\**, *loc=None*, *ip=None*)

    consumer\_release(*\**, *loc=None*, *ip=None*)

    producer\_tail(*\**, *loc=None*, *ip=None*)

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None

*class* cutlass.pipeline.PipelineProducer( : *pipeline*, : *state*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
:   Bases: `object`

    A class representing a producer in an asynchronous pipeline.

    This class manages the producer side of an asynchronous pipeline, handling
    synchronization and state management for producing data. It provides methods for
    acquiring, committing, and advancing through pipeline stages.

    Variables:
    :   * **\_\_pipeline** -- The asynchronous pipeline this producer belongs to
        * **\_\_state** -- The current state of the producer in the pipeline
        * **\_\_group** -- The cooperative group this producer operates in

    **Examples:**

    ```
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to acquire the current buffer without blocking
        try_acquire_token = producer.try_acquire()

        # Do something else independently
        ...

        # Wait for current buffer to be empty & Move index to next stage
        # If try_acquire_token is True, return immediately
        # If try_acquire_token is False, block until buffer is empty
        handle = producer.acquire_and_advance(try_acquire_token)

        # Produce data
        handle.commit()
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
    :   Bases: `ImmutableResourceHandle`

        *property* barrier
        :   Get the barrier pointer for the current pipeline stage.

            Returns:
            :   Pointer to the barrier for the current stage

            Return type:
            :   cute.Pointer

        commit(*\**, *loc=None*, *ip=None*)
        :   Signal that data production is complete for the current stage.

            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    \_\_init\_\_( : *pipeline*, : *state*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
    :   Initialize a new Producer instance.

        Parameters:
        :   * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) -- The pipeline this producer belongs to
            * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) -- Initial pipeline state
            * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*

    \_\_state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    \_\_group*: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    clone()
    :   Create a new Producer instance with the same state.

    reset(*\**, *loc=None*, *ip=None*)
    :   Reset the count of how many handles this producer has committed.

    acquire( : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")
    :   Wait for the current buffer to be empty before producing data.
        This is a blocking operation.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[**Boolean**]*) -- Optional token to try to acquire the buffer

        Returns:
        :   A handle to the producer for committing the data

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    advance(*\**, *loc=None*, *ip=None*)
    :   Move to the next pipeline stage.

    acquire\_and\_advance( : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle")
    :   Acquire the current buffer and advance to the next pipeline stage.

        This method combines the acquire() and advance() operations into a single call.
        It first waits for the current buffer to be empty before producing data,
        then advances the pipeline to the next stage.

        Parameters:
        :   **try\_acquire\_token** (*Optional**[**Boolean**]*) -- Token indicating whether to try non-blocking acquire.
            If True, returns immediately without waiting. If False or None, blocks
            until buffer is empty.

        Returns:
        :   A handle to the producer that can be used to commit data to the
            acquired buffer stage

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")

    try\_acquire( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean
    :   Attempt to acquire the current buffer without blocking.

        This method tries to acquire the current buffer stage for producing data
        without waiting. It can be used to check buffer availability before
        committing to a blocking acquire operation.

        Returns:
        :   A boolean token indicating whether the buffer was successfully acquired

        Return type:
        :   Boolean

    commit( : *handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineProducer.ImmutableResourceHandle") | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   Signal that data production is complete for the current stage.

        This allows consumers to start processing the data.

        Parameters:
        :   **handle** (*Optional**[*[*ImmutableResourceHandle*](#cutlass.pipeline.PipelineProducer.ImmutableResourceHandle "cutlass.pipeline.PipelineProducer.ImmutableResourceHandle")*]*) -- Optional handle to commit, defaults to None

        Raises:
        :   **AssertionError** -- If provided handle does not belong to this producer

    tail(*\**, *loc=None*, *ip=None*)
    :   Ensure all used buffers are properly synchronized before producer exit.

        This should be called before the producer finishes to avoid dangling signals.

*class* cutlass.pipeline.PipelineConsumer( : *pipeline*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
:   Bases: `object`

    A class representing a consumer in an asynchronous pipeline.

    The Consumer class manages the consumer side of an asynchronous pipeline, handling
    synchronization and state management for consuming data. It provides methods for
    waiting, releasing, and advancing through pipeline stages.

    Variables:
    :   * **\_\_pipeline** -- The asynchronous pipeline this consumer belongs to
        * **\_\_state** -- The current state of the consumer in the pipeline
        * **\_\_group** -- The cooperative group this consumer operates in

    **Examples:**

    ```
    pipeline = PipelineAsync.create(...)
    producer, consumer = pipeline.make_participants()
    for i in range(iterations):
        # Try to wait for buffer to be full
        try_wait_token = consumer.try_wait()

        # Do something else independently
        ...

        # Wait for buffer to be full & Move index to next stage
        # If try_wait_token is True, return immediately
        # If try_wait_token is False, block until buffer is full
        handle = consumer.wait_and_advance(try_wait_token)

        # Consume data
        handle.release(  )  # Signal buffer is empty

        # Alternative way to do this is:
        # handle.release()  # Signal buffer is empty
    ```

    *class* ImmutableResourceHandle( : *\_ImmutableResourceHandle\_\_origin: [cutlass.pipeline.sm90.PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [cutlass.pipeline.helpers.PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
    :   Bases: `ImmutableResourceHandle`

        release(*\**, *loc=None*, *ip=None*)
        :   Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.

        \_\_init\_\_( : *\_ImmutableResourceHandle\_\_origin: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*, : *\_ImmutableResourceHandle\_\_immutable\_state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

    \_\_init\_\_( : *pipeline*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, )
    :   Initialize a new Consumer instance.

        Parameters:
        :   * **pipeline** ([*PipelineAsync*](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")) -- The pipeline this consumer belongs to
            * **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) -- Initial pipeline state
            * **group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- The cooperative group for synchronization

    \_\_pipeline*: [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.sm90.PipelineAsync")*

    \_\_group*: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*

    \_\_state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    clone()
    :   Create a new Consumer instance with the same state.

    reset(*\**, *loc=None*, *ip=None*)
    :   Reset the count of how many handles this consumer has consumed.

    wait( : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")
    :   Wait for data to be ready in the current buffer. This is a blocking operation
        that will not return until data is available.

        Parameters:
        :   **try\_wait\_token** (*Optional**[**Boolean**]*) -- Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    advance(*\**, *loc=None*, *ip=None*)
    :   Advance the consumer to the next pipeline stage.

        This updates the internal state to point to the next buffer in the pipeline.
        Should be called after consuming data from the current buffer.

    wait\_and\_advance( : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle")
    :   Atomically wait for data and advance to next pipeline stage.

        This is a convenience method that combines wait() and advance() into a single
        atomic operation. It will block until data is available in the current buffer,
        then automatically advance to the next stage.

        Parameters:
        :   **try\_wait\_token** (*Optional**[**Boolean**]*) -- Token used to attempt a non-blocking wait for the buffer.
            If provided and True, returns immediately if buffer is not ready.

        Returns:
        :   An immutable handle to the consumer that can be used to release the buffer
            once data consumption is complete

        Return type:
        :   [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle")

    try\_wait( : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean
    :   Non-blocking check if data is ready in the current buffer.

        This method provides a way to test if data is available without blocking.
        Unlike wait(), this will return immediately regardless of buffer state.

        Returns:
        :   True if data is ready to be consumed, False if the buffer is not yet ready

        Return type:
        :   Boolean

    release( : *handle: [ImmutableResourceHandle](#cutlass.pipeline.PipelineConsumer.ImmutableResourceHandle "cutlass.pipeline.sm90.PipelineConsumer.ImmutableResourceHandle") | None = None*, : *\**, : *loc=None*, : *ip=None*, )
    :   Signal that data consumption is complete for the current stage.
        This allows producers to start producing new data.

cutlass.pipeline.make\_pipeline\_state( : *type: [PipelineUserType](#cutlass.pipeline.PipelineUserType "cutlass.pipeline.helpers.PipelineUserType")*, : *stages: int*, : *\**, : *loc=None*, : *ip=None*, )
:   Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.

cutlass.pipeline.pipeline\_init\_arrive( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, : *is\_relaxed: bool = False*, : *\**, : *loc=None*, : *ip=None*, )
:   Fences the mbarrier\_init and sends an arrive if using clusters.

cutlass.pipeline.pipeline\_init\_wait( : *cluster\_shape\_mn: cutlass.cute.typing.Layout | None = None*, : *\**, : *loc=None*, : *ip=None*, )
:   Syncs the threadblock or cluster

cutlass.pipeline.agent\_sync( : *group: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *is\_relaxed: bool = False*, : *\**, : *loc=None*, : *ip=None*, )
:   Syncs all threads within an agent.

cutlass.pipeline.arrive(*barrier\_id: int*, *num\_threads: int*, *\**, *loc=None*, *ip=None*)
:   The aligned flavor of arrive is used when all threads in the CTA will execute the
    same instruction. See PTX documentation.

cutlass.pipeline.arrive\_unaligned( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )
:   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

cutlass.pipeline.wait(*\**, *loc=None*, *ip=None*)
:   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
    If synchronizing two warps in a producer/consumer pairing, the arrive count would be
    32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
    or consumer are counted for mbarriers, while all threads participating in the sync
    are counted for NamedBarriers.

cutlass.pipeline.wait\_unaligned( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )

cutlass.pipeline.arrive\_and\_wait( : *barrier\_id: int*, : *num\_threads: int*, : *\**, : *loc=None*, : *ip=None*, )

cutlass.pipeline.sync(*barrier\_id: int = 0*, *\**, *loc=None*, *ip=None*)
