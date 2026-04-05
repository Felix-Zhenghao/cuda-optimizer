# Attention (continued)

## API

### SDPA FP16/BF16 Backward

This operation computes gradient tensors for scaled dot product attention (SDPA) using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). You are required to pass the stats tensor from the forward operation to the backward operation as input.

#### C++ API

```
// returns [dQ, dK, dV]
std::array<std::shared_ptr<Tensor_attributes>, 3>
sdpa_backward(std::shared_ptr<Tensor_attributes> q,
              std::shared_ptr<Tensor_attributes> k,
              std::shared_ptr<Tensor_attributes> v,
              std::shared_ptr<Tensor_attributes> o,
              std::shared_ptr<Tensor_attributes> dO,
              std::shared_ptr<Tensor_attributes> stats,
              SDPA_backward_attributes);
```

The `options` parameter of type `SDPA_backward_attributes` is used to control the attributes of backward operation, as detailed below:

```
SDPA_backward_attributes& set_attn_scale(std::shared_ptr<Tensor_attributes> value);
SDPA_backward_attributes& set_attn_scale(float const value);

// ========================== BEGIN    var len options =====================
SDPA_backward_attributes& set_padding_mask(bool const value);

// integer tensor that specifies the sequence length of each batch
SDPA_backward_attributes& set_seq_len_q(std::shared_ptr<Tensor_attributes> value);
SDPA_backward_attributes& set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

// the maximum number of sequence tokens for all batches, used for workspace allocation
SDPA_backward_attributes& set_max_total_seq_len_q(int64_t const value);
SDPA_backward_attributes& set_max_total_seq_len_kv(int64_t const value);
// ==========================  END     var len options =====================

// ========================== BEGIN score mod options =====================
SDPA_backward_attributes& set_score_mod(std::function<Tensor_t(Graph_t, Tensor_t)>);

// Use in combination to set_diagonal_alignment to set (bottom right) causal masking
SDPA_backward_attributes& set_diagonal_alignment(DiagonalAlignment_t const alignment);
SDPA_backward_attributes& set_diagonal_band_left_bound(int const value);
SDPA_backward_attributes& set_diagonal_band_right_bound(int const value);

// DEPRECATED
// Sets the diagonal position to TOP_LEFT
// calls set_diagonal_band_right_bound(0) if no right_bound was specified
SDPA_backward_attributes& set_causal_mask(bool const value);

// DEPRECATED
// Sets the diagonal position to BOTTOM_RIGHT
// and calls set_diagonal_band_right_bound(0) if no right_bound was specified
SDPA_backward_attributes& set_causal_mask_bottom_right(bool const value);

// DEPRECATED
// calls set_diagonal_band_left_bound(value)
SDPA_backward_attributes& set_sliding_window_length(int const value);

SDPA_backward_attributes& set_bias(std::shared_ptr<Tensor_attributes> value);
SDPA_backward_attributes& set_dbias(std::shared_ptr<Tensor_attributes> value);

SDPA_backward_attributes& set_alibi_mask(bool const value);
// ==========================  END  score modoptions =====================

// ========================== BEGIN   dropout options =====================
SDPA_backward_attributes& set_dropout(float const probability,
                                      std::shared_ptr<Tensor_attributes> seed,
                                      std::shared_ptr<Tensor_attributes> offset);
SDPA_backward_attributes& set_dropout(std::shared_ptr<Tensor_attributes> mask,
                                      std::shared_ptr<Tensor_attributes> scale,
                                      std::shared_ptr<Tensor_attributes> scale_inv);

// for debugging dropout mask with seed and offset
SDPA_backward_attributes& set_rng_dump(std::shared_ptr<Tensor_attributes> value);
// ==========================  END    dropout options =====================

SDPA_backward_attributes& set_deterministic_algorithm(bool const value);

SDPA_backward_attributes& set_compute_data_type(DataType_t const value);
```

#### Python API

```
graph.sdpa_backward(
    q,                                    # Query tensor from forward pass
    k,                                    # Key tensor from forward pass
    v,                                    # Value tensor from forward pass
    o,                                    # Output tensor from forward pass
    dO,                                   # Gradient of output
    stats,                                # Softmax statistics from forward pass
    attn_scale=None,                      # Attention scale factor (must match forward)
    bias=None,                            # Bias tensor from forward pass
    dBias=None,                           # Output tensor for bias gradient
    use_alibi_mask=False,                 # Enable ALiBi (must match forward)
    use_padding_mask=False,               # Enable variable sequence length masking
    seq_len_q=None,                       # Per-batch query sequence lengths
    seq_len_kv=None,                      # Per-batch key/value sequence lengths
    max_total_seq_len_q=None,             # Max total tokens for Q (ragged tensors)
    max_total_seq_len_kv=None,            # Max total tokens for KV (ragged tensors)
    diagonal_alignment=TOP_LEFT,          # Diagonal alignment (must match forward)
    diagonal_band_left_bound=None,        # Left bound (must match forward)
    diagonal_band_right_bound=None,       # Right bound (must match forward)
    dropout=None,                         # Dropout config (must match forward)
    use_deterministic_algorithm=False,    # Force deterministic gradient computation
    compute_data_type=NOT_SET,            # Computation data type
    name=None,                            # Operation name
)
```

**Args:**

* `q` (cudnn\_tensor): The query data from the forward pass.
* `k` (cudnn\_tensor): The key data from the forward pass.
* `v` (cudnn\_tensor): The value data from the forward pass.
* `o` (cudnn\_tensor): The output data from the forward pass.
* `dO` (cudnn\_tensor): The gradient of the loss with respect to the output.
* `stats` (cudnn\_tensor): The softmax statistics tensor from the forward pass (`generate_stats=True`).
* `attn_scale` (Optional[Union[float, cudnn\_tensor]]): The attention scale factor. Must match the forward pass.
* `bias` (Optional[cudnn\_tensor]): The bias tensor from the forward pass.
* `dBias` (Optional[cudnn\_tensor]): Output tensor to store the bias gradient.
* `use_alibi_mask` (Optional[bool]): Enable ALiBi. Must match the forward pass configuration.
* `use_padding_mask` (Optional[bool]): Enable variable sequence length masking. Must match forward pass.
* `seq_len_q` (Optional[cudnn\_tensor]): Per-batch query sequence lengths.
* `seq_len_kv` (Optional[cudnn\_tensor]): Per-batch key/value sequence lengths.
* `max_total_seq_len_q` (Optional[int]): Maximum total sequence tokens for Q when using ragged tensors. Used for workspace allocation. Defaults to \(B \times S\_q\) if not provided.
* `max_total_seq_len_kv` (Optional[int]): Maximum total sequence tokens for KV when using ragged tensors. Used for workspace allocation. Defaults to \(B \times S\_{kv}\) if not provided.
* `diagonal_alignment` (Optional[cudnn.diagonal\_alignment]): Must match the forward pass.
* `diagonal_band_left_bound` (Optional[int]): Must match the forward pass.
* `diagonal_band_right_bound` (Optional[int]): Must match the forward pass.
* `dropout` (Optional[tuple]): Dropout configuration. Must match the forward pass to ensure the same dropout mask is applied.
* `use_deterministic_algorithm` (Optional[bool]): If True, forces deterministic gradient computation. This ensures bitwise-identical results across multiple runs but may be slower. Default is False.
* `compute_data_type` (Optional[cudnn.data\_type]): Data type for internal computation.
* `name` (Optional[str]): Name for the operation.

**Returns:**

* `dQ` (cudnn\_tensor): The gradient with respect to the query tensor.
* `dK` (cudnn\_tensor): The gradient with respect to the key tensor.
* `dV` (cudnn\_tensor): The gradient with respect to the value tensor.

**Important Notes:**

* The backward operation does NOT support paged attention. K and V must be contiguous tensors.
* All masking and dropout configurations must exactly match the forward pass to ensure correct gradients.
* When using ragged tensors, set `max_total_seq_len_q` and `max_total_seq_len_kv` to the maximum total tokens (sum of sequence lengths) for proper workspace allocation.
* Python sample: [samples/python/51\_sdpa\_backward.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/51_sdpa_backward.ipynb)
* C++ sample: [samples/cpp/sdpa](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/sdpa)
* Python tests (v2 with randomized configurations): [test/python/test\_mhas\_v2.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_mhas_v2.py)

#### Tensors

##### Input Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| dO | GPU | FP16 or BF16 | \((B, H\_{q}, S\_{q}, D\_{v})\) |

##### Output Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| dQ | GPU | FP16 or BF16 | \((B, H\_{q}, S\_{q}, D\_{qk})\) |
| dK | GPU | FP16 or BF16 | \((B, H\_{k}, S\_{kv}, D\_{qk})\) |
| dV | GPU | FP16 or BF16 | \((B, H\_{v}, S\_{kv}, D\_{v})\) |

**Example Usage:**

```
# Backward pass graph
graph_backward = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

q = graph_backward.tensor_like(q_gpu)
k = graph_backward.tensor_like(k_gpu)
v = graph_backward.tensor_like(v_gpu)
o = graph_backward.tensor_like(o_gpu)
dO = graph_backward.tensor_like(dO_gpu)
stats = graph_backward.tensor_like(stats_gpu)

dQ, dK, dV = graph_backward.sdpa_backward(
    name="sdpa_backward",
    q=q,
    k=k,
    v=v,
    o=o,
    dO=dO,
    stats=stats,
    attn_scale=attn_scale,
    diagonal_band_right_bound=0,  # Must match forward
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    use_deterministic_algorithm=True,  # For reproducible training
)

dQ.set_output(True).set_dim(q_gpu.shape).set_stride(q_gpu.stride())
dK.set_output(True).set_dim(k_gpu.shape).set_stride(k_gpu.stride())
dV.set_output(True).set_dim(v_gpu.shape).set_stride(v_gpu.stride())
```

### SDPA Backward FE OSS API (SM100, D=256)

This experimental FE OSS API provides a CUTE DSL implementation of the SDPA backward pass for head dimension `256` on NVIDIA Blackwell GPUs (`SM100+`). It computes `dQ`, `dK`, and `dV` from the forward tensors plus `dO` and `LSE`. Available through a standalone API (see [sdpa\_bwd\_d256.md](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#sdpa-backward-fe-oss-sm100-d256) for details) or as part of the experimental [SDPA Pytorch custom operator](#scaled-dot-product-attention-pytorch-op).

### SDPA PyTorch Custom Op (Experimental)

A high-level PyTorch custom operator that wraps the cuDNN SDPA forward and backward graphs into a single, autograd-compatible function. This provides a drop-in replacement for `torch.nn.functional.scaled_dot_product_attention` that routes computation through cuDNN.

**Key features:**

* Full autograd support (forward + backward)
* `torch.compile` compatible via FakeTensor/meta registration
* Graph caching for efficient repeated execution
* Supports FP16, BF16 datatypes
* Supports causal masking, sliding window, padding mask, GQA/MQA, and ragged tensors

**Limitations:**

* `attn_mask` and `dropout` are not yet supported
* FP8 is not supported (use the Graph API directly)
* For head dimension `256`, the specialized backward path currently supports only plain BHSD inputs. `seq_len_q`, `seq_len_kv`, `cumulative_seq_len_q`, and `cumulative_seq_len_kv` are not supported on that backward path.

#### Python API

```
from cudnn.experimental.ops import scaled_dot_product_attention

output = scaled_dot_product_attention(
    query,                        # (B, H_q, S_q, D) -- FP16 or BF16
    key,                          # (B, H_k, S_kv, D)
    value,                        # (B, H_v, S_kv, D_v)
    attn_mask=None,               # Not yet supported, must be None
    dropout_p=0.0,                # Not yet supported, must be 0.0
    is_causal=False,              # Apply causal (upper-triangular) mask
    scale=None,                   # Attention scale, defaults to 1/sqrt(D)
    enable_gqa=False,             # Enable grouped-query attention (H_q > H_k)
    *,
    diagonal_alignment=0,         # 0 = TOP_LEFT, 1 = BOTTOM_RIGHT
    left_bound=-1,                # Sliding window left bound (-1 = disabled)
    right_bound=-1,               # Sliding window right bound (-1 = disabled)
    seq_len_q=None,               # Actual query seq lengths (B, 1, 1, 1) INT32
    seq_len_kv=None,              # Actual key/value seq lengths (B, 1, 1, 1) INT32
    cumulative_seq_len_q=None,    # Ragged offset for Q (B+1, 1, 1, 1) INT32
    cumulative_seq_len_kv=None,   # Ragged offset for KV (B+1, 1, 1, 1) INT32
)
```

**Args:**

* `query` (torch.Tensor): Query tensor in BHSD layout `(B, H_q, S_q, D)`.
* `key` (torch.Tensor): Key tensor in BHSD layout `(B, H_k, S_kv, D)`.
* `value` (torch.Tensor): Value tensor in BHSD layout `(B, H_v, S_kv, D_v)`.
* `attn_mask` (Optional[torch.Tensor]): Not yet supported. Must be `None`.
* `dropout_p` (float): Not yet supported. Must be `0.0`.
* `is_causal` (bool): If `True`, applies a causal mask (sets `right_bound=0`).
* `scale` (Optional[float]): Attention scale factor. Defaults to `1/sqrt(D)`.
* `enable_gqa` (bool): When `False`, raises `ValueError` if `H_q != H_k`. Set to `True` for grouped-query or multi-query attention.
* `diagonal_alignment` (int): `0` for TOP\_LEFT, `1` for BOTTOM\_RIGHT alignment.
* `left_bound` (int): Left sliding-window bound. `-1` disables.
* `right_bound` (int): Right sliding-window bound. `-1` disables. `0` for causal.
* `seq_len_q` (Optional[torch.Tensor]): Per-batch query sequence lengths `(B, 1, 1, 1)` INT32.
* `seq_len_kv` (Optional[torch.Tensor]): Per-batch key/value sequence lengths `(B, 1, 1, 1)` INT32.
* `cumulative_seq_len_q` (Optional[torch.Tensor]): Ragged offset for Q `(B+1, 1, 1, 1)` INT32.
* `cumulative_seq_len_kv` (Optional[torch.Tensor]): Ragged offset for KV `(B+1, 1, 1, 1)` INT32.

For head dimension `256`, backward support is narrower than the general SDPA op contract: the specialized `d=256` backward path requires plain BHSD tensors and does not support `seq_len_q`, `seq_len_kv`, `cumulative_seq_len_q`, or `cumulative_seq_len_kv`.

**Returns:**

* `output` (torch.Tensor): Attention output `(B, H_q, S_q, D_v)`.

#### Example Usage

```
import torch
from cudnn.experimental.ops import scaled_dot_product_attention

B, H, S, D = 2, 8, 1024, 128

q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)

# Forward
output = scaled_dot_product_attention(q, k, v, is_causal=True)

# Backward (autograd handles this automatically)
loss = output.sum()
loss.backward()
# q.grad, k.grad, v.grad are now populated
```

#### Tests

* Python tests: [test/python/test\_cudnn\_sdpa\_op.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_cudnn_sdpa_op.py)

### SDPA FP8 Forward

This operation computes the scaled dot product attention (SDPA) in the 8-bit floating point (FP8) datatype, using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). It is applicable for both training and inference phases, with an option to generate a stats tensor to be used for backwards training computation.

The FP8 datatype consists of two encodings:

* `FP8_E4M3` (1 sign bit, 4 exponent bits, and 3 mantissa bits)
* `FP8_E5M2` (1 sign bit, 5 exponent bits, 2 mantissa bits).

Due to the limited numerical precision of FP8 data type, for practical use cases, you must scale values computed in FP32 format before storing them in FP8 format, and descale the values stored in FP8 format before performing computations on them. For more information, refer to [the Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

The suggested value for the scaling factor is computed as: (Max representable value in the fp8 format) / (Max absolute value seen in the tensor for the previous layer).

* For E4M3, the suggested scaling factor is `448.f/ prev_layer_tensor_amax` (rounded to the nearest lower power of two)
* For E5M2, the suggested scaling factor is `57344.f/ prev_layer_tensor_amax` (rounded to the nearest lower power of two)

The suggested value for the descale factor is the reciprocal of the scale factor.

Since scaling and descaling are critical for convergence with FP8 datatype, you are required to pass scaling and descaling input tensors, as well as amax output tensors.

#### C++ API

```
// returns [o, stats, amax_s, amax_o]
std::array<std::shared_ptr<Tensor_attributes>, 4>
Graph::sdpa_fp8(std::shared_ptr<Tensor_attributes> q,
                std::shared_ptr<Tensor_attributes> k,
                std::shared_ptr<Tensor_attributes> v,
                std::shared_ptr<Tensor_attributes> descale_q,
                std::shared_ptr<Tensor_attributes> descale_k,
                std::shared_ptr<Tensor_attributes> descale_v,
                std::shared_ptr<Tensor_attributes> descale_s,
                std::shared_ptr<Tensor_attributes> scale_s,
                std::shared_ptr<Tensor_attributes> scale_o,
                SDPA_fp8_attributes attributes);
```

The `options` parameter of type `SDPA_fp8_attributes` is used to control the attributes of the forward operation, as detailed below:

```
// Indicates that softmax_stats should be generated (useful during training).
// If false, the softmax_stats output will be nullptr.
SDPA_fp8_attributes&
set_generate_stats(bool const value);

SDPA_fp8_attributes&
set_logit_max(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_score_sum_exp(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_attn_scale(float const value);

SDPA_fp8_attributes&
set_causal_mask(bool const value);

SDPA_fp8_attributes&
set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_padding_mask(bool const value);

SDPA_fp8_attributes&
set_seq_len_q(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_attributes&
set_dropout(float const probability,
            std::shared_ptr<Tensor_attributes> seed,
            std::shared_ptr<Tensor_attributes> offset);

SDPA_fp8_attributes&
set_dropout(std::shared_ptr<Tensor_attributes> mask,
            std::shared_ptr<Tensor_attributes> scale);

// DEPRECATED
// Calls set_generate_stats(!value) (note the negation of `value`).
SDPA_fp8_attributes&
set_is_inference(bool const value);
```

#### Python API

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    descale_q (cudnn_tensor): Descale factor for query.
    descale_k (cudnn_tensor): Descale factor for key.
    descale_v (cudnn_tensor): Descale factor for value.
    descale_s (cudnn_tensor): Descale factor for S tensor.
    scale_s (cudnn_tensor): Scale factor for S tensor.
    scale_o (cudnn_tensor): Scale factor for output.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.
    generate_stats (Optional[bool]): If true, compute and output softmax stats (useful at training time). Default is None, but one of {generate_stats, is_inference} must be set.
Deprecated Args:
    is_inference (Optional[bool]): If false, compute and output softmax stats. Prefer generate_stats instead (NOTE: generate_stats takes the negation of the argument to is_inference).

Returns:
    o (cudnn_tensor): The output data.
    stats (Optional[cudnn_tensor]): The softmax statistics, if generate_stats is true.
    amax_s (cudnn_tensor): The absolute maximum of S tensor.
    amax_o (cudnn_tensor): The absolute maximum of output tensor.
```

#### Configurable Options

The current FP8 support is a subset of the options supported in FP16 and BF16 support.

* Attention scale (`attn_scale`): Applies a scaling factor to attention scores before the softmax, such as \(\frac{1}{\sqrt{\text{d}}}\). Set to 1.0 by default.
* Causal mask: Fills the upper triangular matrix of attention scores with negative infinity.

#### Limitations

* Requires Hopper (SM90) or newer architecture.
* Head dimension must be a multiple of 16.
* Limited masking options compared to FP16/BF16 (causal mask only).
* Requires explicit scale/descale tensors for all FP8 inputs and outputs.

#### Tensors

The tensors in forward operation are defined as the following:

\(P = QK^T\)

\(S = \text{softmax}(P)\)

\(O = SV\)

##### Input Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| Q | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{qk})\) |
| K | GPU | E4M3 or E5M2 | \((B, H\_{k}, S\_{kv}, D\_{qk})\) |
| V | GPU | E4M3 or E5M2 | \((B, H\_{v}, S\_{kv}, D\_{v})\) |
| Descale Q | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale K | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale V | GPU | FP32 | \((1, 1, 1, 1)\) |
| (Bias mask) Bias Mask | GPU | E4M3 or E5M2 | \((1, 1, S\_{q}, S\_{kv})\), \((1, H\_{q}, S\_{q}, S\_{kv})\), \((B, 1, S\_{q}, S\_{kv})\), or \((B, H\_{q}, S\_{q}, S\_{kv})\) |
| (Padding mask) Sequence Length Q | GPU | INT32 | \((B, 1, 1, 1)\) |
| (Padding mask) Sequence Length KV | GPU | INT32 | \((B, 1, 1, 1)\) |
| (Philox RNG Dropout) Seed | CPU or GPU | INT32 or INT64 | \((1, 1, 1, 1)\) |
| (Philox RNG Dropout) Offset | CPU or GPU | INT32 or INT64 | \((1, 1, 1, 1)\) |
| (Custom Dropout Mask) Mask | GPU | E4M3 or E5M2 | \((1, 1, S\_{q}, S\_{kv})\), \((1, H\_{q}, S\_{q}, S\_{kv})\), \((B, 1, S\_{q}, S\_{kv})\), or \((B, H\_{q}, S\_{q}, S\_{kv})\) |
| (Custom Dropout Mask) Scale | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale S | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale S | GPU | FP32 | \((1, 1, 1, 1)\) |

##### Output Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| O | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{v})\) |
| Stats (training only) | GPU | FP32 | \((B, H\_{q}, S\_{q}, 1)\) |
| AMax S | GPU | FP32 | \((1, 1, 1, 1)\) |
| AMax O | GPU | FP32 | \((1, 1, 1, 1)\) |

Where:

* \(B\) is the batch size
* \(H\_{q}\) is the number of query heads
* \(H\_{k}\) is the number of key heads
* \(H\_{v}\) is the number of value heads
* \(S\_{q}\) is the sequence length of the query
* \(S\_{kv}\) is the sequence length of the key and value
* \(D\_{qk}\) is the embedding dimension per head of query and key
* \(D\_{v}\) is the embedding dimension per head of value

#### Samples and tests

* C++ sample: [samples/cpp/sdpa](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/sdpa)

### SDPA FP8 Backward

This operation computes the gradients for scaled dot product attention (SDPA) 8-bit floating point (FP8) datatype, using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). You are required to pass the stats tensor from the forward operation to the backward operation as input.

* C++ sample: [samples/cpp/sdpa](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/sdpa)

#### C++ API

```
// returns [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP]
std::array<std::shared_ptr<Tensor_attributes>, 7>
Graph::sdpa_fp8_backward(std::shared_ptr<Tensor_attributes> q,
                         std::shared_ptr<Tensor_attributes> k,
                         std::shared_ptr<Tensor_attributes> v,
                         std::shared_ptr<Tensor_attributes> o,
                         std::shared_ptr<Tensor_attributes> dO,
                         std::shared_ptr<Tensor_attributes> Stats,
                         std::shared_ptr<Tensor_attributes> descale_q,
                         std::shared_ptr<Tensor_attributes> descale_k,
                         std::shared_ptr<Tensor_attributes> descale_v,
                         std::shared_ptr<Tensor_attributes> descale_o,
                         std::shared_ptr<Tensor_attributes> descale_do,
                         std::shared_ptr<Tensor_attributes> descale_s,
                         std::shared_ptr<Tensor_attributes> descale_dp,
                         std::shared_ptr<Tensor_attributes> scale_s,
                         std::shared_ptr<Tensor_attributes> scale_dq,
                         std::shared_ptr<Tensor_attributes> scale_dk,
                         std::shared_ptr<Tensor_attributes> scale_dv,
                         std::shared_ptr<Tensor_attributes> scale_dp,
                         SDPA_fp8_backward_attributes attributes);
```

The `options` parameter of type `SDPA_fp8_backward_attributes` is used to control the attributes of the backward operation, as detailed below:

```
SDPA_fp8_backward_attributes&
set_attn_scale(std::shared_ptr<Tensor_attributes> value);

SDPA_fp8_backward_attributes&
set_attn_scale(float const value);

SDPA_fp8_backward_attributes&
set_causal_mask(bool const value);
```

#### Python API

```
Args:
    q (cudnn_tensor): The query data.
    k (cudnn_tensor): The key data.
    v (cudnn_tensor): The value data.
    o (cudnn_tensor): The output data.
    dO (cudnn_tensor): The output gradient data.
    stats (cudnn_tensor): The softmax statistics in case the operation is in a training step.
    descale_q (cudnn_tensor): Descale factor for query.
    descale_k (cudnn_tensor): Descale factor for key.
    descale_v (cudnn_tensor): Descale factor for value.
    descale_o (cudnn_tensor): Descale factor for output.
    descale_dO (cudnn_tensor): Descale factor for output gradient.
    descale_s (cudnn_tensor): Descale factor for S tensor.
    descale_dP (cudnn_tensor): Descale factor for P gradient tensor.
    scale_s (cudnn_tensor): Scale factor for S tensor.
    scale_dQ (cudnn_tensor): Scale factor for query gradient.
    scale_dK (cudnn_tensor): Scale factor for key gradient.
    scale_dV (cudnn_tensor): Scale factor for value gradient.
    scale_dP (cudnn_tensor): Scale factor for dP gradient.
    attn_scale (Optional[Union[float, cudnn_tensor]]): The scale factor for attention. Default is None.
    use_causal_mask (Optional[bool]): Whether to use causal mask. Default is False.
    compute_data_type (Optional[cudnn.data_type]): The data type for computation. Default is NOT_SET.
    name (Optional[str]): The name of the operation.

Returns:
    dQ (cudnn_tensor): The query gradient data.
    dK (cudnn_tensor): The key gradient data.
    dV (cudnn_tensor): The value gradient data.
    amax_dQ (cudnn_tensor): The absolute maximum of query gradient tensor.
    amax_dK (cudnn_tensor): The absolute maximum of key gradient tensor.
    amax_dV (cudnn_tensor): The absolute maximum of value gradient tensor.
    amax_dP (cudnn_tensor): The absolute maximum of dP tensor.
```

#### Limitations

* Requires Hopper (SM90) or newer architecture.
* Dropout is not supported in FP8 backward pass.
* Only causal masking is supported.
* Requires explicit scale/descale tensors for all FP8 inputs and outputs.

#### Tensors

The tensors in backward operation are defined as the following:

\(dV = S^TdO\)

\(dS = dOV^T\)

\(dP = \text{dSoftmax}(dS)\)

\(dQ = dPK\)

\(dK = QdP\)

##### Input Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| Q | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{qk})\) |
| K | GPU | E4M3 or E5M2 | \((B, H\_{k}, S\_{kv}, D\_{qk})\) |
| V | GPU | E4M3 or E5M2 | \((B, H\_{v}, S\_{kv}, D\_{v})\) |
| O | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{v})\) |
| dO | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{v})\) |
| Stats | GPU | FP32 | \((B, H\_{q}, S\_{q}, 1)\) |
| Descale Q | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale K | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale V | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale O | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale dO | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale S | GPU | FP32 | \((1, 1, 1, 1)\) |
| Descale dP | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale S | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale dQ | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale dK | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale dV | GPU | FP32 | \((1, 1, 1, 1)\) |
| Scale dP | GPU | FP32 | \((1, 1, 1, 1)\) |

##### Output Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| dQ | GPU | E4M3 or E5M2 | \((B, H\_{q}, S\_{q}, D\_{qk})\) |
| dK | GPU | E4M3 or E5M2 | \((B, H\_{k}, S\_{kv}, D\_{qk})\) |
| dV | GPU | E4M3 or E5M2 | \((B, H\_{v}, S\_{kv}, D\_{v})\) |
| Amax dQ | GPU | FP32 | \((1, 1, 1, 1)\) |
| Amax dK | GPU | FP32 | \((1, 1, 1, 1)\) |
| Amax dV | GPU | FP32 | \((1, 1, 1, 1)\) |
| Amax dP | GPU | FP32 | \((1, 1, 1, 1)\) |

Where:

* \(B\) is the batch size
* \(H\_{q}\) is the number of query heads
* \(H\_{k}\) is the number of key heads
* \(H\_{v}\) is the number of value heads
* \(S\_{q}\) is the sequence length of the query
* \(S\_{kv}\) is the sequence length of the key and value
* \(D\_{qk}\) is the embedding dimension per head of query and key
* \(D\_{v}\) is the embedding dimension per head of value

## FAQs

### Logical vs Physical Layout

#### BHSD Layout (Batch-Head-Sequence-Dim)

The default logical layout where dimensions are ordered as \((B, H, S, D)\).

* **Dimensions:** \([B, H\_q, S\_q, D\_{qk}]\)
* **Strides:** \([H\_q \times S\_q \times D\_{qk}, S\_q \times D\_{qk}, D\_{qk}, 1]\)

This is the most common layout and matches PyTorch's default attention tensor ordering.

#### BSHD Layout (Batch-Sequence-Head-Dim)

A physical layout where sequence comes before heads in memory, while maintaining the logical \((B, H, S, D)\) dimension order.

* **Dimensions:** \([B, H\_q, S\_q, D\_{qk}]\) (logical order, unchanged)
* **Strides:** \([S\_q \times H\_q \times D\_{qk}, D\_{qk}, H\_q \times D\_{qk}, 1]\)

Note: The dimension order remains \((B, H, S, D)\) but strides are reordered so that in memory, sequence varies faster than head.

### Determinism Support

#### Fprop

Always deterministic on all architectures

#### Bprop

SM100 - Deterministic requires cuDNN backend version 9.19.0 or later.
SM80 and SM120 do not support deterministic algorithm with ragged input tensor.
SM90 - Determinism is supported.

### cuDNN Flex Attention API

SDPA and SDPA backward operations now accept the functions `set_score_mod` and `set_score_mod_bprop`, which allows modification of the attention score matrix. These functions can be used to program a sub-graph of pointwise operations that can subsequently be used to program the score modifier. Note that this function usage is mutually exclusive to the usage of ready made options. Also, note that the graph argument in the score\_mod function is not the same as the sdpa graph. So, any tensor to be passed as input to the score-mod sub-graph must first be registered with main graph and subsequently passed as argument to the score\_mod function. The SDPA operation also now accepts the function `set_block_mask`, which applies a block mask to the score matrix. The implementation assumes a 128 x 128 block size.

## cuDNN Version History for SDPA

This section documents features and fixes introduced in each cuDNN version for SDPA operations.

### Version 9.19.0

* FP8 deterministic algorithm support on Blackwell
* d\_qk=192 with d\_v=128 support for FP8

### Version 9.18.0

* THD/Ragged support on Ampere and Ada (SM80/SM89)
* Deterministic algorithm on Blackwell for FP16/BF16

### Version 9.15.0

* Padding mask support for Unified SDPA
* Paged attention inputs for Unified SDPA

### Version 9.14.0

* Block mask support for Unified SDPA
* **Known Issue**: Non-causal + s\_kv > 1024 + sliding window may have issues

### Version 9.13.0

* Unified SDPA implementation (requires 9.13.1)
* FP8 output in FP16/BF16 format on Blackwell
* Sink token support

### Version 9.11.0

* DeepSeek configuration (d\_qk=192, d\_v=128) backward on Hopper
* Blackwell backward support with d\_qk=192

### Version 9.10.2

* Paged attention with packed page tables

### Version 9.10.0/9.10.1

* **Known Issues**: General stability issues - recommend using 9.10.2+

### Version 9.9.0

* Various head dimension expansions for decode mode

### Version 9.7.0

* Bottom-right causal masking for FP8 (SM100+)
* Paged + ragged combination support

### Version 9.6.0

* GQA with ragged offset support
* Bottom-right causal mask seqlen flexibility

### Version 9.5.0

* Paged attention support
* dBias with variable sequence lengths

### Version 9.3.0

* Bottom-right causal masking for FP16/BF16
* **Minimum recommended version for new deployments**

### Version 9.2.0

* Sliding window attention

### Version 9.1.0

* FP8 SDPA support (Hopper+)

### Version 9.0.0

* Sequence length flexibility (s\_q, s\_kv not required to be multiples of 64)

### Version 8.9.6

* Padding mask, ALiBi mask support
* Bias mask support

### Version 8.9.3

* Initial SDPA support (SM80+)
