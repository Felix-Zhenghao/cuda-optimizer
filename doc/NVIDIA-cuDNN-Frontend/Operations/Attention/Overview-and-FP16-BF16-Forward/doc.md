# Attention

## Scaled Dot Product Attention

This operation computes the scaled dot product attention (SDPA), as

\(\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V\)

using the FlashAttention-2 algorithm as described in the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). It is applicable for both training and inference phases, with an option to generate a stats tensor to be used for backwards training computation.

## Support Matrix

cudnn SDPA operation requires SM80 (Ampere) or newer architectures and cuda toolkit 12.x or newer.

The support matrix is based on the latest cudnn backend version 9.18.1

| Arch | Datatype | Layout | Paged   Attn | Masking | Deterministic | Head dim |
| --- | --- | --- | --- | --- | --- | --- |
| Ampere/Ada   (Prefill) | fp16, bf16 | BHSD, BSHD, Interleaved[1],   Padded[2], Ragged[3] | Yes | Yes[4] | Yes | d <= 256 |
| Ampere/Ada   (Decode) | fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 128 |
| Ampere/Ada   (Bprop) | fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | NA | Yes | Yes | d <= 128 |
|  |  |  |  |  |  |  |
| Hopper   (Prefill) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 256[5]   (d\_qk = 192, d\_vo = 128) |
| Hopper   (Decode) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 256   (d\_qk = 192, d\_vo = 128) |
| Hopper   (Bprop) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | NA | Yes | Yes | d <= 256   (d\_qk = 192, d\_vo = 128) |
|  |  |  |  |  |  |  |
| Blackwell (B200/B300)   (Prefill) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 256   (d\_qk = 192, d\_vo = 128) |
| Blackwell (B200/B300)   (Decode) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 128   (d\_qk = 192, d\_vo = 128) |
| Blackwell (B200/B300)   (Bprop) | fp8, fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | NA | Yes | Yes | d <= 128   (d\_qk = 192, d\_vo = 128) |
|  |  |  |  |  |  |  |
| Blackwell (Consumer)   (Prefill) | fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 256 |
| Blackwell (Consumer)   (Decode) | fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | Yes | Yes | Yes | d <= 128   (d\_qk = 192, d\_vo = 128) |
| Blackwell (Consumer)   (Bprop) | fp16, bf16 | BHSD, BSHD, Interleaved,   Padded, Ragged | NA | Yes | Yes | d <= 128 |

### Glossary

[1] Interleaved q,k,v tensors. Generally they have layouts as BS3HD, B3SHD.

[2] Padded, variable length sequences (requires padding mask). When sequences in a batch have different lengths, use `use_padding_mask=True` with sequence length tensors.

   *Setup:*
   - Set `use_padding_mask=True`
   - Provide `seq_len_q` tensor of shape `(B, 1, 1, 1)` with actual query sequence lengths
   - Provide `seq_len_kv` tensor of shape `(B, 1, 1, 1)` with actual key/value sequence lengths

   *Example:*
   Batch with sequences "aa" (length 2) and "bbb" (length 3), max length `S=8`:

* `seq_len_q = [2, 3]`
* `seq_len_kv = [2, 3]`

  ```
  Q[b=0] = aa000000  (6 padding tokens)
  Q[b=1] = bbb00000  (5 padding tokens)
  ```
* Dimensions: \([B=2, H=1, S=8, D=64]\)
* Strides: \([512, 64, 64, 1]\) (standard BHSD)

   cuDNN automatically masks out padding tokens during attention computation.

[3] Ragged Layout.

   For memory efficiency, variable-length sequences can be **packed** together without padding. This is called THD layout where \(T = \sum(\text{seq\_len})\) is the total number of valid tokens.

   **Requirements:**

* Must set ragged offset tensor via `tensor.set_ragged_offset(ragged_offset_tensor)`

   **Ragged Offset Tensor:**

* Shape: \((B + 1, 1, 1, 1)\)
* Contains cumulative token offsets in **elements** (not bytes)
* Last element is the total number of tokens

   **Memory Layout visualization:**

     *Example:*

     Same sequences "aa" and "bbb" packed together:

* `seq_len_q = [2, 3]`
* `seq_len_kv = [2, 3]`

  ```
  Q = aabbb  (no padding, T=5 total tokens)
  ```
* Dimensions: \([B=2, H=1, S=8, D=64]\) (S is still max sequence length)
* Strides: \([512, 64, 64, 1]\) (strides unchanged, but ignored for ragged)
* **Ragged offset:** \([0, 2 \times H \times D, 5 \times H \times D] = [0, 128, 320]\)

     *Partially Packed Layout:*

     Tokens within each batch can be contiguous without being globally packed.

* Ragged offset: \([0, 4 \times H \times D, 7 \times H \times D] = [0, 256, 448]\)

  ```
  Q = aa00bbb0  (batch 0 at offset 0, batch 1 at offset 4)
  ```

     *Not Supported:*

     Tokens that are not contiguous within a batch cannot be represented.

* `seq_len_q = [2, 3]`

  ```
  Q = a0abbb00bb000000  (tokens interleaved - NOT SUPPORTED)
  ```

   **Note that Q,K,V and their gradients can be individually ragged or not.**

   **Backward Pass with THD:**

   When using THD layout with cudnn, maximum total tokens are needed for efficient workspace allocation. If not set, defaults to \(B \times S\) which may overallocate memory.

[4] None, Causal, Sliding window, Additive Bias, Softcap, Arbitrary masking.

[5] d\_qo should be equal to d\_kv. (Except when d\_qk == 192 and d\_vo = 128, which is also supported.)

### Important Notes on Support Surface

1. All attention flavors MHA, MQA, GQA are supported.
2. The head dim (d) should be a multiple of 8 for fp16/bf16 and multiple of 16 for fp8 data-types.
3. The seqlens s\_q, and s\_kv can have arbitrary value.
4. The layout of q,k,v,o and dq, dk, dv, do can be independent of each other.
5. **Dropout**: Randomly zeros some of the attention weights after the softmax as a form of regularization.
   You can configure dropout in two ways:

   * **Philox RNG dropout** (more performant): Provide:

     + An RNG seed tensor (INT32 or INT64)
     + An RNG offset tensor (INT32 or INT64)
     + A float representing the dropout probability (probability that any weight is set to zero)
     + (Debug only) Output RNG dump tensor to capture the generated dropout mask
   * **Custom dropout mask**: Provide:

     + A `dropout mask` tensor matching the attention weights' dimensions. Dimensions set to 1 will broadcast.
     + A `dropout scale` tensor to adjust remaining weights, typically \(1 / (1 - \text{dropout probability})\).
6. Stats from fprop is supported (Max, Sum). In addition QKClip required for KimiK2, Qwen are also supported optionally.

## Benchmarks

To run the sdpa benchmarks, refer to [benchmarks/sdpa](https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/sdpa_benchmark_training/README.md) folder. Current results:

### GB200 - Llama 3.1 Causal (top\_left)

![Llama 3.1 Causal on GB200](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb200_918_only_cudnn/llama3.1_top_left_causal.png)

* SDPA parameters: `batch=1; num_q_heads=64; num_kv_heads=8; head_dim=128; is_causal=True`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB200 GPU

### GB200 - Llama 3.1 Non-Causal (no\_mask)

![Llama 3.1 Non-Causal on GB200](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb200_918_only_cudnn/llama3.1_no_mask.png)

* SDPA parameters: `batch=1; num_q_heads=64; num_kv_heads=8; head_dim=128; is_causal=False`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB200 GPU

### GB200 - DeepSeek V3 Causal (top\_left)

![DeepSeek V3 Causal on GB200](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb200_918_only_cudnn/dsv3_top_left_causal.png)

* SDPA parameters: `batch=1; num_q_heads=128; num_kv_heads=128; head_dim_qk=192; head_dim_vo=128; is_causal=True`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB200 GPU

### GB300 - Llama 3.1 Causal (top\_left)

![Llama 3.1 Causal on GB300](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb300_918_only_cudnn/llama3.1_top_left_causal.png)

* SDPA parameters: `batch=1; num_q_heads=64; num_kv_heads=8; head_dim=128; is_causal=True`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB300 GPU

### GB300 - Llama 3.1 Non-Causal (no\_mask)

![Llama 3.1 Non-Causal on GB300](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb300_918_only_cudnn/llama3.1_no_mask.png)

* SDPA parameters: `batch=1; num_q_heads=64; num_kv_heads=8; head_dim=128; is_causal=False`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB300 GPU

### GB300 - DeepSeek V3 Causal (top\_left)

![DeepSeek V3 Causal on GB300](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/benchmark/sdpa_benchmark_training/results/gb300_918_only_cudnn/dsv3_top_left_causal.png)

* SDPA parameters: `batch=1; num_q_heads=128; num_kv_heads=128; head_dim_qk=192; head_dim_vo=128; is_causal=True`
* Sequence lengths shown on x-axis
* Results obtained on NVIDIA GB300 GPU

## API

### SDPA FP16/BF16 Forward

#### C++ API

```
// returns [output, softmax_stats]
std::array<std::shared_ptr<Tensor_attributes>, 2>
sdpa(std::shared_ptr<Tensor_attributes> q,
     std::shared_ptr<Tensor_attributes> k,
     std::shared_ptr<Tensor_attributes> v,
     SDPA_attributes options);
```

The `options` parameter of type `SDPA_attributes` is used to control the attributes of the forward operation, as detailed below:

```
// Indicates that softmax_stats should be generated (useful during training).
// If false, the softmax_stats output will be nullptr.
SDPA_attributes& set_generate_stats(bool const value);

// Indicates whether the kernel should output max of attention score
// and numerically stable sum of exponents using normalized values wrt max score
SDPA_attributes& set_logit_max(std::shared_ptr<Tensor_attributes> value);
SDPA_attributes& set_score_sum_exp(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes& set_attn_scale(std::shared_ptr<Tensor_attributes> value);
SDPA_attributes& set_attn_scale(float const value);

// DEPRECATED
// Calls set_generate_stats(!value) (note the negation of `value`).
SDPA_attributes& set_is_inference(bool const value);

// ========================== BEGIN paged attn options =====================
SDPA_attributes& set_paged_attention_k_table(std::shared_ptr<Tensor_attributes> value);
SDPA_attributes& set_paged_attention_v_table(std::shared_ptr<Tensor_attributes> value);
SDPA_attributes& set_paged_attention_max_seq_len_kv(int const value);
// ==========================  END  paged attn options =====================

// ========================== BEGIN    var len options =====================
SDPA_attributes& set_padding_mask(bool const value);

// integer tensor that specifies the sequence length of each batch
SDPA_attributes& set_seq_len_q(std::shared_ptr<Tensor_attributes> value);
SDPA_attributes& set_seq_len_kv(std::shared_ptr<Tensor_attributes> value);
// ==========================  END     var len options =====================

// ========================== BEGIN score mod options =====================
SDPA_attributes& set_score_mod(std::function<Tensor_t(Graph_t, Tensor_t)>);

// Use in combination to set diagonal masking
SDPA_attributes& set_diagonal_alignment(DiagonalAlignment_t const alignment);
SDPA_attributes& set_diagonal_band_left_bound(int const value);
SDPA_attributes& set_diagonal_band_right_bound(int const value);

// DEPRECATED
// Sets the diagonal position to TOP_LEFT
// calls set_diagonal_band_right_bound(0) if no right_bound was specified
SDPA_attributes& set_causal_mask(bool const value);

// DEPRECATED
// Sets the diagonal position to BOTTOM_RIGHT
// and calls set_diagonal_band_right_bound(0) if no right_bound was specified
SDPA_attributes& set_causal_mask_bottom_right(bool const value);

// DEPRECATED
// calls set_diagonal_band_left_bound(value)
SDPA_attributes& set_sliding_window_length(int const value);

SDPA_attributes& set_bias(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes& set_block_mask(std::shared_ptr<Tensor_attributes> value);

SDPA_attributes& set_alibi_mask(bool const value);
// ==========================  END  score mod options =====================

// ========================== BEGIN   dropout options =====================
SDPA_attributes& set_dropout(float const probability,
                             std::shared_ptr<Tensor_attributes> seed,
                             std::shared_ptr<Tensor_attributes> offset);

SDPA_attributes& set_dropout(std::shared_ptr<Tensor_attributes> mask,
                             std::shared_ptr<Tensor_attributes> scale);

// for debugging dropout mask with seed and offset
SDPA_attributes& set_rng_dump(std::shared_ptr<Tensor_attributes> value);
// ==========================  END    dropout options =====================

// ========================== BEGIN   experimental options ================
// Sets the underlying SDPA implementation to use (default is AUTO).
SDPA_attributes& set_implementation(AttentionImplementation_t value);
// ==========================  END    experimental options ================

SDPA_attributes& set_compute_data_type(DataType_t value);
```

#### Python API

```
graph.sdpa(
    q,                                    # Query tensor
    k,                                    # Key tensor (or container for paged attention)
    v,                                    # Value tensor (or container for paged attention)
    attn_scale=None,                      # Attention scale factor (float or tensor)
    bias=None,                            # Additive bias mask tensor
    block_mask=None,                      # Block mask tensor (128x128 tiles, UNIFIED only)
    use_alibi_mask=False,                 # Enable ALiBi positional encoding
    use_padding_mask=False,               # Enable variable sequence length masking
    seq_len_q=None,                       # Per-batch query sequence lengths
    seq_len_kv=None,                      # Per-batch key/value sequence lengths
    diagonal_alignment=TOP_LEFT,          # Diagonal alignment: TOP_LEFT or BOTTOM_RIGHT
    diagonal_band_left_bound=None,        # Left bound for sliding window (None = no bound)
    diagonal_band_right_bound=None,       # Right bound for causal mask (0 = causal, None = no bound)
    dropout=None,                         # Dropout config: (prob, seed, offset) or (mask, scale)
    rng_dump=None,                        # Debug: output tensor for RNG dropout mask
    paged_attention_k_table=None,         # Page table for K container
    paged_attention_v_table=None,         # Page table for V container
    paged_attention_max_seq_len_kv=None,  # Max KV sequence length for paged attention
    generate_stats=None,                  # Output softmax stats for training (True/False)
    implementation=AUTO,                  # SDPA implementation: AUTO, COMPOSITE, UNIFIED
    compute_data_type=NOT_SET,            # Computation data type
    name=None,                            # Operation name
)
```

**Args:**

* `q` (cudnn\_tensor): The query data with shape \((B, H\_q, S\_q, D\_{qk})\).
* `k` (cudnn\_tensor): The key data. When `paged_attention_k_table` is provided, this is a container of non-contiguous key blocks.
* `v` (cudnn\_tensor): The value data. When `paged_attention_v_table` is provided, this is a container of non-contiguous value blocks.
* `attn_scale` (Optional[Union[float, cudnn\_tensor]]): Scale factor for attention scores. Typically \(\frac{1}{\sqrt{d}}\). Default is None (no scaling).
* `bias` (Optional[cudnn\_tensor]): Additive bias mask for attention scores. Supports broadcasting.
* `block_mask` (Optional[cudnn\_tensor]): Block-level mask for 128x128 tiles. Only supported with UNIFIED implementation.
* `use_alibi_mask` (Optional[bool]): Enable ALiBi (Attention with Linear Biases) positional encoding. Requires `diagonal_band_right_bound=0`.
* `use_padding_mask` (Optional[bool]): Enable variable sequence length masking. Must also provide `seq_len_q` and `seq_len_kv`.
* `seq_len_q` (Optional[cudnn\_tensor]): Per-batch query sequence lengths with shape \((B, 1, 1, 1)\).
* `seq_len_kv` (Optional[cudnn\_tensor]): Per-batch key/value sequence lengths with shape \((B, 1, 1, 1)\).
* `diagonal_alignment` (Optional[cudnn.diagonal\_alignment]): Alignment for diagonal masking. `TOP_LEFT` for standard causal, `BOTTOM_RIGHT` for prefix-LM style.
* `diagonal_band_left_bound` (Optional[int]): Left bound for sliding window attention. Masks columns at or before `row_idx - left_bound`.
* `diagonal_band_right_bound` (Optional[int]): Right bound for causal masking. Set to 0 for causal mask. Masks columns beyond `row_idx + right_bound`.
* `dropout` (Optional[tuple]): Dropout configuration. Either `(probability, seed, offset)` for Philox RNG or `(mask, scale)` for custom mask.
* `rng_dump` (Optional[cudnn\_tensor]): Debug tensor to capture the Philox RNG dropout mask.
* `paged_attention_k_table` (Optional[cudnn\_tensor]): Page table with block offsets into the K container.
* `paged_attention_v_table` (Optional[cudnn\_tensor]): Page table with block offsets into the V container.
* `paged_attention_max_seq_len_kv` (Optional[int]): Maximum sequence length for K/V caches. Recommended when using paged attention.
* `generate_stats` (Optional[bool]): If True, output softmax statistics for backward pass. Required for training.
* `implementation` (Optional[cudnn.attention\_implementation]): SDPA implementation to use. `AUTO` (default), `COMPOSITE`, or `UNIFIED`.
* `compute_data_type` (Optional[cudnn.data\_type]): Data type for internal computation.
* `name` (Optional[str]): Name for the operation.

**Returns:**

* `o` (cudnn\_tensor): The output attention data with shape \((B, H\_q, S\_q, D\_v)\).
* `stats` (Optional[cudnn\_tensor]): Softmax statistics with shape \((B, H\_q, S\_q, 1)\) when `generate_stats=True`.

#### Configurable Options

* **Attention scale** (`attn_scale`): Applies a scaling factor to attention scores before the softmax, such as \(\frac{1}{\sqrt{\text{d}}}\). Set to 1.0 by default. Can be passed as a float or as a tensor.
* **Bias mask**: Applies an additive bias mask to attention scores. You must pass a bias tensor as specified in the tensors section below. The dimensions that are passed as 1 will apply a broadcasted mask over attention scores.
* **Block mask**: Masks out tiles of attention scores at a 128x128 block granularity. The block mask is a uint8 tensor where each bit represents whether a 128x128 tile should be computed (1) or masked out (0). This is supported with the UNIFIED implementation.
* **ALiBi mask**: Attention with Linear Biases (ALiBi) is an additive mask applied to the attention scores as described in the paper [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409). When using ALiBi, `diagonal_band_right_bound` must be set to exactly 0 (causal masking).
* **Padding mask** (Variable Sequence Length): Masks out padded time steps to ignore them in computation. You must pass per-batch sequence length tensors as specified in the tensors section below. In padded or ragged layout (discussed below) where the actual seqlen can be less than the max seqlens of a graph, certain batches can be skipped by setting the actual seqlen of the corresponding batch to 0.
* **Diagonal masking options**: These options control causal and sliding window masking:
* **Diagonal Alignment** (`diagonal_alignment`): Specifies where the diagonal starts. Options are:

  + `TOP_LEFT`: The diagonal starts at the top-left of the attention matrix. Used for standard causal masking.
  + `BOTTOM_RIGHT`: The diagonal starts at the bottom-right of the attention matrix, aligned with the actual sequence length. Useful for prefix-LM or when \(S\_q \neq S\_{kv}\).
* **Diagonal Band Right Bound** (`diagonal_band_right_bound`): Specifies that attention scores beyond column `row_idx + right_bound` are masked with negative infinity. Setting this to 0 enables causal masking.
* **Diagonal Band Left Bound** (`diagonal_band_left_bound`): Specifies that attention scores at or before column `row_idx - left_bound` are masked with negative infinity. This enables sliding window attention.
* **Common masking patterns**:

  + Causal mask (top-left): `diagonal_alignment=TOP_LEFT`, `right_bound=0`
  + Causal mask (bottom-right): `diagonal_alignment=BOTTOM_RIGHT`, `right_bound=0`
  + Sliding window: Set `left_bound` to window size
  + Band attention: Set both `left_bound` and `right_bound`
* **Paged attention**: Enables non-contiguous K/V caches to reduce memory fragmentation. See the [PagedAttention paper](https://arxiv.org/abs/2309.06180).

  + **Requirements**:

    - Pass `page_table_k` tensor with block offsets into the K container (optional if K is not paged)
    - Pass `page_table_v` tensor with block offsets into the V container (optional if V is not paged)
    - Pass sequence length tensors (`seq_len_q`, `seq_len_kv`) for padding mask
    - Optionally pass `paged_attention_max_seq_len_kv` for the maximum KV sequence length (recommended)
  + **Offset calculation**:

    - \(K\_{cache}[b,h,s,d] = K\_{container}[page\\_table\\_k[b,1,s / bs\_k, 1], h, s \mod bs\_k, d]\)
    - \(V\_{cache}[b,h,s,d] = V\_{container}[page\\_table\\_v[b,1,s / bs\_v, 1], h, s \mod bs\_v, d]\)
  + **Packed page tables**: Page tables can also use ragged offsets to pack only the necessary block indices, useful for frameworks that prefer packed representations.
* **Implementation**: Select the underlying SDPA implementation:

  + `AUTO` (default): Auto-selects the best implementation. Recommended for most users.
  + `COMPOSITE`: Standard cuDNN graph representing SDPA as distinct operations.
  + `UNIFIED`: Optimized fused SDPA operation (cuDNN 9.13.1+). Supports a subset of features including block masking.
* **Generate stats** (`generate_stats`): When `True`, outputs softmax statistics needed for backward pass during training. Set to `True` for training, `False` for inference.

#### Limitations

* Head dimension must be a multiple of 8.
* ALiBi requires causal masking (`diagonal_band_right_bound=0`).
* Block masking is only supported with the UNIFIED implementation.
* Ampere/Ada architectures are limited to head dimensions up to 256 for prefill, 128 for decode and backward.

#### Tensors

##### Input Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| Q | GPU | FP16 or BF16 | \((B, H\_{q}, S\_{q}, D\_{qk})\) |
| K | GPU | FP16 or BF16 | \((B, H\_{k}, S\_{kv}, D\_{qk})\), or \((num\\_blocks\_{k}, H\_{k}, bs\_{k}, D\_{qk})\) in case of paged K cache |
| V | GPU | FP16 or BF16 | \((B, H\_{v}, S\_{kv}, D\_{v})\), or \((num\\_blocks\_{v}, H\_{v}, bs\_{v}, D\_{v})\) in case of paged V cache |
| (Bias mask) Bias Mask | GPU | FP16 or BF16 | \((1, 1, S\_{q}, S\_{kv})\), \((1, H\_{q}, S\_{q}, S\_{kv})\), \((B, 1, S\_{q}, S\_{kv})\), or \((B, H\_{q}, S\_{q}, S\_{kv})\) |
| (Padding mask/Paged Caches) Sequence Length Q | GPU | INT32 | \((B, 1, 1, 1)\) |
| (Padding mask/Paged Caches) Sequence Length KV | GPU | INT32 | \((B, 1, 1, 1)\) |
| (Philox RNG Dropout) Seed | CPU or GPU | INT32 or INT64 | \((1, 1, 1, 1)\) |
| (Philox RNG Dropout) Offset | CPU or GPU | INT32 or INT64 | \((1, 1, 1, 1)\) |
| (Custom Dropout Mask) Mask | GPU | FP16 or BF16 | \((1, 1, S\_{q}, S\_{kv})\), \((1, H\_{q}, S\_{q}, S\_{kv})\), \((B, 1, S\_{q}, S\_{kv})\), or \((B, H\_{q}, S\_{q}, S\_{kv})\) |
| (Custom Dropout Mask) Scale | GPU | FP32 | \((1, 1, 1, 1)\) |
| (Packed Layout) Ragged Offset | GPU | INT32 | \((B + 1, 1, 1, 1)\) |
| (Paged Attention) Page Table K | GPU | INT32 | \((B, 1, ceil(S\_{kv}/bs\_{k}), 1)\) |
| (Paged Attention) Page Table V | GPU | INT32 | \((B, 1, ceil(S\_{kv}/bs\_{v}), 1)\) |
| (Paged Attention) Max Sequence Length KV | CPU | INT32 or INT64 | \((1, 1, 1, 1)\) |

##### Output Tensors

| Tensor Name | Device | Data Type | Dimensions |
| --- | --- | --- | --- |
| O | GPU | FP16 or BF16 | \((B, H\_{q}, S\_{q}, D\_{v})\) |
| Stats (training only) | GPU | FP32 | \((B, H\_{q}, S\_{q}, 1)\) |
| (Philox RNG Dropout) RNG Dump | GPU | FP32 | \((B, H\_{q}, S\_{q}, S\_{kv})\) |

Where:

* \(B\) is the batch size
* \(H\_{q}\) is the number of query heads
* \(H\_{k}\) is the number of key heads
* \(H\_{v}\) is the number of value heads
* \(S\_{q}\) is the sequence length of the query
* \(S\_{kv}\) is the sequence length of the key and value
* \(D\_{qk}\) is the embedding dimension per head of query and key
* \(D\_{v}\) is the embedding dimension per head of value
* \(bs\_{k}\) is the (power of 2) block size of the K container
* \(bs\_{v}\) is the (power of 2) block size of the V container
* \(num\\_blocks\_{k}\) is the number of blocks in the K container
* \(num\\_blocks\_{v}\) is the number of blocks in the V container

#### Samples and Tests

* Python forward sample: [samples/python/50\_sdpa\_forward.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/50_sdpa_forward.ipynb)
* Python backward sample: [samples/python/51\_sdpa\_backward.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/51_sdpa_backward.ipynb)
* Python prefill sample with paged caches: [samples/python/52\_sdpa\_with\_paged\_caches.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/52_sdpa_with_paged_caches.ipynb)
* Python decode sample with packed paged caches: [samples/python/53\_sdpa\_decode\_with\_paged\_caches.ipynb](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/53_sdpa_decode_with_paged_caches.ipynb)
* C++ sample: [samples/cpp/sdpa](https://github.com/NVIDIA/cudnn-frontend/tree/main/samples/cpp/sdpa)
* Python tests (v2 with randomized configurations): [test/python/test\_mhas\_v2.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_mhas_v2.py)

**Example Usage:**

```
import cudnn
import torch
import math

# Create graph
graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# Create tensor descriptors
q = graph.tensor_like(q_gpu)
k = graph.tensor_like(k_gpu)
v = graph.tensor_like(v_gpu)

# Forward pass with causal masking
o, stats = graph.sdpa(
    name="sdpa",
    q=q,
    k=k,
    v=v,
    attn_scale=1.0 / math.sqrt(d),
    generate_stats=True,                          # For training
    diagonal_band_right_bound=0,                  # Causal mask
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
)

o.set_output(True).set_dim(shape_o).set_stride(stride_o)
stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

# Build and execute
graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
```

