# cutlass.pipeline

*class* cutlass.pipeline.Agent(*value*)
:   Bases: `Enum`

    Agent indicates what is participating in the pipeline synchronization.

    Thread *= 1*

    ThreadBlock *= 2*

    ThreadBlockCluster *= 3*

*class* cutlass.pipeline.CooperativeGroup( : *agent: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int = 1*, : *alignment=None*, )
:   Bases: `object`

    CooperativeGroup contains size and alignment restrictions for an Agent.

    \_\_init\_\_( : *agent: [Agent](#cutlass.pipeline.Agent "cutlass.pipeline.helpers.Agent")*, : *size: int = 1*, : *alignment=None*, )

*class* cutlass.pipeline.PipelineOp(*value*)
:   Bases: `Enum`

    PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.

    AsyncThread *= 1*

    TCGen05Mma *= 2*

    TmaLoad *= 3*

    ClcLoad *= 4*

    TmaStore *= 5*

    Composite *= 6*

    AsyncLoad *= 7*

*class* cutlass.pipeline.SyncObject
:   Bases: `ABC`

    Abstract base class for hardware synchronization primitives.

    This class defines the interface for different types of hardware synchronization
    mechanisms including shared memory barriers, named barriers, and fences.

    *abstract* arrive() → None

    *abstract* wait() → None

    *abstract* arrive\_and\_wait() → None

    *abstract* arrive\_and\_drop() → None

    *abstract* get\_barrier() → cutlass.cute.typing.Pointer | int | None

    *abstract* max() → int | None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.MbarrierArray
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    MbarrierArray implements an abstraction for an array of smem barriers.

    \_\_init\_\_( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, : *\**, : *loc=None*, : *ip=None*, ) → None

    recast\_to\_new\_op\_type( : *new\_op\_type: [PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp")*, ) → [MbarrierArray](#cutlass.pipeline.MbarrierArray "cutlass.pipeline.helpers.MbarrierArray")
    :   Creates a copy of MbarrierArray with a different op\_type without re-initializing barriers

    mbarrier\_init(*\**, *loc=None*, *ip=None*) → None
    :   Initializes an array of mbarriers using warp 0.

    arrive( : *index: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None
    :   Select the arrive corresponding to this MbarrierArray's PipelineOp.

        Parameters:
        :   * **index** (*int*) -- Index of the mbarrier in the array to arrive on
            * **dst** (*int* *|* *None*) -- Destination parameter for selective arrival, which can be either a mask or destination cta rank.
              When None, both `TCGen05Mma` and `AsyncThread` will arrive on their local mbarrier.
              - For `TCGen05Mma`, `dst` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
              in the cluster with rank = 0, 1, and 3).
              - For `AsyncThread`, `dst` serves as a destination cta rank (e.g., 3 means threads will arrive on
              the mbarrier with rank = 3 in the cluster).
            * **cta\_group** (`cute.nvgpu.tcgen05.CtaGroup`, optional) -- CTA group for `TCGen05Mma`, defaults to None for other op types

    arrive\_mbarrier( : *index: int*, : *dst\_rank: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None

    arrive\_cp\_async\_mbarrier( : *index: int*, : *\**, : *loc=None*, : *ip=None*, )

    arrive\_tcgen05mma( : *index: int*, : *mask: int | None*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup")*, : *\**, : *loc=None*, : *ip=None*, ) → None

    arrive\_and\_expect\_tx( : *index: int*, : *tx\_count: int*, : *\**, : *loc=None*, : *ip=None*, ) → None

    arrive\_and\_expect\_tx\_with\_dst( : *index: int*, : *tx\_count: int*, : *dst: int | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None

    try\_wait( : *index: int*, : *phase: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cutlass\_dsl.Boolean

    wait( : *index: int*, : *phase: int*, : *\**, : *loc=None*, : *ip=None*, ) → None

    arrive\_and\_wait( : *index: int*, : *phase: int*, : *dst: int*, : *cta\_group: [CtaGroup](cute_nvgpu_tcgen05.html#cutlass.cute.nvgpu.tcgen05.CtaGroup "cutlass.cute.nvgpu.tcgen05.mma.CtaGroup") | None = None*, : *\**, : *loc=None*, : *ip=None*, ) → None

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None

    get\_barrier( : *index: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer

    max() → int

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.NamedBarrier(*barrier\_id: int*, *num\_threads: int*)
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    NamedBarrier is an abstraction for named barriers managed by hardware.
    There are 16 named barriers available, with barrier\_ids 0-15.

    See the [PTX documentation](https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar).

    barrier\_id*: int*

    num\_threads*: int*

    arrive(*\**, *loc=None*, *ip=None*) → None
    :   The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.

    arrive\_unaligned(*\**, *loc=None*, *ip=None*) → None
    :   The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.

    wait(*\**, *loc=None*, *ip=None*) → None
    :   NamedBarriers do not have a standalone wait like mbarriers, only an arrive\_and\_wait.
        If synchronizing two warps in a producer/consumer pairing, the arrive count would be
        32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
        or consumer are counted for mbarriers, while all threads participating in the sync
        are counted for NamedBarriers.

    wait\_unaligned(*\**, *loc=None*, *ip=None*) → None

    arrive\_and\_wait(*\**, *loc=None*, *ip=None*) → None

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None

    sync(*\**, *loc=None*, *ip=None*) → None

    get\_barrier(*\**, *loc=None*, *ip=None*) → int

    max() → int

    \_\_init\_\_(*barrier\_id: int*, *num\_threads: int*) → None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.PipelineOrder( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, )
:   Bases: `object`

    PipelineOrder is used for managing ordered pipeline execution with multiple groups.

    This class implements a pipeline ordering mechanism where work is divided into groups
    and stages, allowing for controlled progression through pipeline stages with proper
    synchronization between different groups.

    The pipeline ordering works as follows:
    - The pipeline is divided into âlength' number of groups
    - Each group has âdepth' number of stages
    - Groups execute in a specific order with synchronization barriers
    - Each group waits for the previous group to complete before proceeding

    **Example:**

    ```
    # Create pipeline order with 3 groups, each with 2 stages
    pipeline_order = PipelineOrder.create(
        barrier_storage=smem_ptr,      # shared memory pointer for barriers
        depth=2,                       # 2 stages per group
        length=3,                      # 3 groups total
        group_id=0,                    # current group ID (0, 1, or 2)
        producer_group=producer_warp   # cooperative group for producers
    )

    # In the pipeline loop
    for stage in range(num_stages):
        pipeline_order.wait()          # Wait for previous group to complete
        # Process current stage
        pipeline_order.arrive()        # Signal completion to next group
    ```

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    depth*: int*

    length*: int*

    group\_id*: int*

    state*: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*

    *static* create( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *depth: int*, : *length: int*, : *group\_id: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *defer\_sync: bool = False*, )

    get\_barrier\_for\_current\_stage\_idx(*group\_id*)

    arrive(*\**, *loc=None*, *ip=None*)

    wait(*\**, *loc=None*, *ip=None*)

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *depth: int*, : *length: int*, : *group\_id: int*, : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, ) → None

*class* cutlass.pipeline.TmaStoreFence(*num\_stages: int = 0*)
:   Bases: [`SyncObject`](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")

    TmaStoreFence is used for a multi-stage epilogue buffer.

    \_\_init\_\_(*num\_stages: int = 0*) → None

    arrive(*\**, *loc=None*, *ip=None*) → None

    wait(*\**, *loc=None*, *ip=None*) → None

    arrive\_and\_wait(*\**, *loc=None*, *ip=None*) → None

    arrive\_and\_drop(*\**, *loc=None*, *ip=None*) → None

    get\_barrier(*\**, *loc=None*, *ip=None*) → None

    max() → None

    tail(*\**, *loc=None*, *ip=None*) → None

    \_abc\_impl *= <\_abc.\_abc\_data object>*

*class* cutlass.pipeline.PipelineUserType(*value*)
:   Bases: `Enum`

    An enumeration.

    Producer *= 1*

    Consumer *= 2*

    ProducerConsumer *= 3*

*class* cutlass.pipeline.PipelineState(*stages: int*, *count*, *index*, *phase*)
:   Bases: `object`

    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.

    \_\_init\_\_(*stages: int*, *count*, *index*, *phase*)

    clone() → [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")

    *property* index*: cutlass.cutlass\_dsl.Int32*

    *property* count*: cutlass.cutlass\_dsl.Int32*

    *property* stages*: int*

    *property* phase*: cutlass.cutlass\_dsl.Int32*

    reset\_count(*\**, *loc=None*, *ip=None*)

    advance(*\**, *loc=None*, *ip=None*) → None

    reverse(*\**, *loc=None*, *ip=None*)

*class* cutlass.pipeline.PipelineAsync( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, )
:   Bases: `object`

    PipelineAsync is a generic pipeline class where both the producer and consumer are
    AsyncThreads. It also serves as a base class for specialized pipeline classes.

    This class implements a producer-consumer pipeline pattern where both sides operate
    asynchronously. The pipeline maintains synchronization state using barrier objects
    to coordinate between producer and consumer threads.

    The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

    Table 2 Pipeline State Transitions

    | Barrier | State | p.acquire | p.commit | c.wait | c.release |
    | --- | --- | --- | --- | --- | --- |
    | empty\_bar | empty | <Return> | n/a | n/a |  |
    | empty\_bar | wait | <Block> | n/a | n/a | -> empty |
    | full\_bar | wait | n/a | -> full | <Block > | n/a |
    | full\_bar | full | n/a |  | <Return> | n/a |

    Where:

    * p: producer
    * c: consumer
    * <Block>: This action is blocked until transition to a state allow it to proceed by other side
      - e.g. `p.acquire()` is blocked until `empty_bar` transition to `empty` state by `c.release()`

    ```
    Array of mbarriers as circular buffer:

         Advance Direction
       <-------------------

        Producer   Consumer
            |         ^
            V         |
       +-----------------+
     --|X|X|W|D|D|D|D|R|X|<-.
    /  +-----------------+   \
    |                        |
    `------------------------'
    ```

    Where:

    * X: Empty buffer (initial state)
    * W: Producer writing (producer is waiting for buffer to be empty)
    * D: Data ready (producer has written data to buffer)
    * R: Consumer reading (consumer is consuming data from buffer)

    **Example:**

```python
    # Create pipeline with 5 stages
    pipeline = PipelineAsync.create(
        num_stages=5,                   # number of pipeline stages
        producer_group=producer_warp,
        consumer_group=consumer_warp
        barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
    )

    producer, consumer = pipeline.make_participants()
    # Producer side
    for i in range(num_iterations):
        handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
        # Write data to pipeline buffer
        handle.commit()   # Signal buffer is full

    # Consumer side
    for i in range(num_iterations):
        handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
        # Read data from pipeline buffer
        handle.release()  # Signal buffer is empty
    ```

    sync\_object\_full*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    sync\_object\_empty*: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*

    num\_stages*: int*

    producer\_mask*: cutlass.cutlass\_dsl.Int32 | None*

    consumer\_mask*: cutlass.cutlass\_dsl.Int32 | None*

    *static* \_make\_sync\_object( : *barrier\_storage: cutlass.cute.typing.Pointer*, : *num\_stages: int*, : *agent: tuple[[PipelineOp](#cutlass.pipeline.PipelineOp "cutlass.pipeline.helpers.PipelineOp"), [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")]*, : *tx\_count: int = 0*, ) → [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")
    :   Returns a SyncObject corresponding to an agent's PipelineOp.

    *static* create( : *\**, : *num\_stages: int*, : *producer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *consumer\_group: [CooperativeGroup](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.helpers.CooperativeGroup")*, : *barrier\_storage: cutlass.cute.typing.Pointer | None = None*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None = None*, : *defer\_sync: bool = False*, )
    :   Creates and initializes a new PipelineAsync instance.

        This helper function computes necessary attributes and returns an instance of PipelineAsync
        with the specified configuration for producer and consumer synchronization.

        Parameters:
        :   * **barrier\_storage** (*cute.Pointer*) -- Pointer to the shared memory address for this pipeline's mbarriers
            * **num\_stages** (*int*) -- Number of buffer stages for this pipeline
            * **producer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the producer agent
            * **consumer\_group** ([*CooperativeGroup*](#cutlass.pipeline.CooperativeGroup "cutlass.pipeline.CooperativeGroup")) -- `CooperativeGroup` for the consumer agent
            * **producer\_mask** (*Int32**,* *optional*) -- Mask for signaling arrives for the producer agent
            * **consumer\_mask** (*Int32**,* *optional*) -- Mask for signaling arrives for the consumer agent

        Raises:
        :   **ValueError** -- If barrier\_storage is not a cute.Pointer instance

        Returns:
        :   A new `PipelineAsync` instance

        Return type:
        :   [PipelineAsync](#cutlass.pipeline.PipelineAsync "cutlass.pipeline.PipelineAsync")

    producer\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_acquire\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )

    producer\_try\_acquire( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )

    producer\_commit( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )

    consumer\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *try\_wait\_token: cutlass.cutlass\_dsl.Boolean | None = None*, : *\**, : *loc=None*, : *ip=None*, )

    consumer\_try\_wait( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )

    consumer\_release( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )

    producer\_get\_barrier( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer

    producer\_tail( : *state: [PipelineState](#cutlass.pipeline.PipelineState "cutlass.pipeline.helpers.PipelineState")*, : *\**, : *loc=None*, : *ip=None*, )
    :   Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        Parameters:
        :   **state** ([*PipelineState*](#cutlass.pipeline.PipelineState "cutlass.pipeline.PipelineState")) -- The pipeline state that points to next useful buffer

    make\_producer(*\**, *loc=None*, *ip=None*)

    make\_consumer(*\**, *loc=None*, *ip=None*)

    make\_participants(*\**, *loc=None*, *ip=None*)

    \_\_init\_\_( : *sync\_object\_full: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *sync\_object\_empty: [SyncObject](#cutlass.pipeline.SyncObject "cutlass.pipeline.helpers.SyncObject")*, : *num\_stages: int*, : *producer\_mask: cutlass.cutlass\_dsl.Int32 | None*, : *consumer\_mask: cutlass.cutlass\_dsl.Int32 | None*, ) → None

