#### 9.7.15.3. [Matrix Data-types](#asynchronous-warpgroup-level-matrix-data-types)

The matrix multiply and accumulate operation is supported separately on integer, floating-point,
sub-byte integer and single bit data-types. All operands must contain the same basic type kind,
i.e., integer or floating-point.

For floating-point matrix multiply and accumulate operation, different matrix operands may have
different precision, as described later.

For integer matrix multiply and accumulate operation, both multiplicand matrices (A and B) must have
elements of the same data-type, e.g. both signed integer or both unsigned integer.

| Data-type | Multiplicands (A or B) | Accumulator (D) |
| --- | --- | --- |
| Integer | both `.u8` or both `.s8` | `.s32` |
| Floating Point | `.f16` | `.f16`, `.f32` |
| Alternate floating Point | `.bf16` | `.f32` |
| Alternate floating Point | `.tf32` | `.f32` |
| Alternate floating Point | `.e4m3`, `.e5m2` | `.f16`, `.f32` |
| Single-bit integer | `.b1` | `.s32` |

---

#### 9.7.15.4. [Async Proxy](#asynchronous-warpgroup-level-matrix-async-proxy)

The `wgmma.mma_async` operations are performed in the asynchronous proxy (or async proxy).

Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the async
proxy, `fence.proxy.async` should be used to synchronize memory between generic proxy and the
async proxy.

The completion of a `wgmma.mma_async` operation is followed by an implicit generic-async proxy
fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as
its completion is observed. `wgmma.commit_group` and `wgmma.wait_group` operations must be used
to wait for the completion of the `wgmma.mma_async` instructions.
