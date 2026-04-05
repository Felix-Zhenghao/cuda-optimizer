# 5.6. Device-Callable APIs and Intrinsics

This chapter contains reference material and API documentation for APIs and intrinsics which can be called from CUDA kernels and device code.

## 5.6.1. Memory Barrier Primitives Interface

The primitives API is a C-like interface to `cuda::barrier` functionality. These primitives are available by including the `<cuda_awbarrier_primitives.h>` header.

### 5.6.1.1. Data Types

```
typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;
```

### 5.6.1.2. Memory Barrier Primitives API

```
uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
```

* `bar` must be a pointer to `__shared__` memory.
* `expected_count <= __mbarrier_maximum_count()`
* Initialize `*bar` expected arrival count for the current and next phase to `expected_count`.

```
void __mbarrier_inval(__mbarrier_t* bar);
```

* `bar` must be a pointer to the barrier object residing in shared memory.
* Invalidation of `*bar` is required before the corresponding shared memory can be repurposed.

```
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
```

* Initialization of `*bar` must happen before this call.
* Pending count must not be zero.
* Atomically decrement the pending count for the current phase of the barrier.
* Return an arrival token associated with the barrier state immediately prior to the decrement.

```
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

* Initialization of `*bar` must happen before this call.
* Pending count must not be zero.
* Atomically decrement the pending count for the current phase and expected count for the next phase of the barrier.
* Return an arrival token associated with the barrier state immediately prior to the decrement.

```
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```

* `token` must be associated with the immediately preceding phase or current phase of `*bar`.
* Returns `true` if `token` is associated with the immediately preceding phase of `*bar`, otherwise returns `false`.

```
bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);
```

* `phase_parity` must indicate the parity of either the current phase or the immediately preceding phase of `*bar`. A value of `true` corresponds to odd-numbered phases and a value of `false` corresponds to even-numbered phases.
* Returns `true` if `phase_parity` indicates the integer parity of the immediately preceding phase of `*bar`, otherwise returns `false`.

```
bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_nanosec);
```

* `token` must be associated with the immediately preceding phase or current phase of `*bar`.
* Returns `true` if `token` is associated with the immediately preceding phase of `*bar`. Otherwise, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes (returns `true`) OR before the phase completes following a system-dependent time limit (returns `false`).
* `max_sleep_nanosec` specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit.

```
bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_nanosec);
```

* `phase_parity` must indicate the parity of either the current phase or the immediately preceding phase of `*bar`. A value of `true` corresponds to odd-numbered phases and a value of `false` corresponds to even-numbered phases.
* Returns `true` if `phase_parity` indicates the integer parity of the immediately preceding phase of `*bar`. Otherwise, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes (returns `true`) OR before the phase completes following a system-dependent time limit (returns `false`).
* `max_sleep_nanosec` specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit.

## 5.6.2. Pipeline Primitives Interface

Pipeline primitives provide a C-like interface for the functionality available in `<cuda/pipeline>`. The pipeline primitives interface is available by including the `<cuda_pipeline.h>` header. When compiling without ISO C++ 2011 compatibility, include the `<cuda_pipeline_primitives.h>` header.

Note

The pipeline primitives API only supports tracking asynchronous copies from global memory to shared memory with specific size and alignment requirements. It provides equivalent functionality to a `cuda::pipeline` object with `cuda::thread_scope_thread`.

### 5.6.2.1. `memcpy_async` Primitive

```
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```

* Request that the following operation be submitted for asynchronous evaluation:

  ```
  size_t i = 0;
  for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
  for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
  ```
* Requirements:

  + `dst_shared` must be a pointer to the shared memory destination for the `memcpy_async`.
  + `src_global` must be a pointer to the global memory source for the `memcpy_async`.
  + `size_and_align` must be 4, 8, or 16.
  + `zfill <= size_and_align`.
  + `size_and_align` must be the alignment of `dst_shared` and `src_global`.
* It is a race condition for any thread to modify the source memory or observe the destination memory prior to waiting for the `memcpy_async` operation to complete. Between submitting a `memcpy_async` operation and waiting for its completion, any of the following actions introduces a race condition:

  + Loading from `dst_shared`.
  + Storing to `dst_shared` or `src_global`.
  + Applying an atomic update to `dst_shared` or `src_global`.

### 5.6.2.2. Commit Primitive

```
void __pipeline_commit();
```

* Commit submitted `memcpy_async` to the pipeline as the current batch.

### 5.6.2.3. Wait Primitive

```
void __pipeline_wait_prior(size_t N);
```

* Let `{0, 1, 2, ..., L}` be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread.
* Wait for completion of batches *at least* up to and including `L-N`.

### 5.6.2.4. Arrive On Barrier Primitive

```
void __pipeline_arrive_on(__mbarrier_t* bar);
```

* `bar` points to a barrier in shared memory.
* Increments the barrier arrival count by one, when all memcpy\_async operations sequenced before this call have completed, the arrival count is decremented by one and hence the net effect on the arrival count is zero. It is user's responsibility to make sure that the increment on the arrival count does not exceed `__mbarrier_maximum_count()`.

