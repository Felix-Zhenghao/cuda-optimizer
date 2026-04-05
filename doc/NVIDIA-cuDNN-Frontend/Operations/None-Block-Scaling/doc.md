# Block Scaling

## Block Scale Quantize

The block scale quantize operation computes the quantized output
and scaling factor tensors from a higher precision tensor.

The MXFP8 recipe quantizes across 32 FP32 elements along the rows
(and optionally columns) to produce 32 FP8 output values (E4M3 or E5M2)
and 1 FP8 scaling factor (E8M0). The NVFP4 recipe quantizes across
16 FP32 elements along the rows to produce 16 FP4 output values (E2M1)
and 1 FP8 scaling factor (E4M3).

The computation can be mathematically represented by the following equation:

\( scale = quantize\\_round\\_up(amax(vals) / vmax\\_otype) \)
\( output = quantize\\_round\\_to\\_even(vals / scale) \)

Where:

* vals is a block of elements.
* vmax\_otype is the maximum value representable by the output data type.

### C++ API

```
std::array<std::shared_ptr<Tensor_attributes>, 2> block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                                                                       Block_scale_quantize_attributes);
```

where the output array is in the order of `[y, scale]`

Block\_scale\_quantize\_attributes is a lightweight structure with setters:

```
Block_scale_quantize_attributes&
set_block_size(int32_t const value)

Block_scale_quantize_attributes&
set_axis(int64_t const value)

Block_scale_quantize_attributes&
set_transpose(bool const value)
```

## Block Scale Dequantize

The block scale dequantize operation computes the dequantized output
tensor from quantized input and scale tensors.

The computation can be mathematically represented by the following equation:

\( output = dequantize(vals \* scale) \)

Where:

* vals is a block of elements.
* scale is broadcast to the block size.

### C++ API

```
std::shared_ptr<Tensor_attributes> block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                                                          std::shared_ptr<Tensor_attributes> scale,
                                                          Block_scale_dequantize_attributes);
```

Block\_scale\_dequantize\_attributes is a lightweight structure with setters:

```
Block_scale_dequantize_attributes&
set_block_size(int32_t const value, int32_t idx = 0)

Block_scale_dequantize_attributes&
set_block_size(const int32_t* values, int32_t len = 1)

Block_scale_dequantize_attributes&
set_block_size(const std::vector<int32_t>& values)
```

---

# Concatenate

The Concatenate operation merges two or more tensors into one, along the specified axis. The user may also specify an in-place merge. The operation provides the capabilities of the [cudnn backend's concatenate operation](../../../backend/v9.20.0/api/cudnn-graph-library.html#cudnn-backend-operation-concat-descriptor "(in NVIDIA cuDNN Backend)").

## C++ API

```
std::shared_ptr<Tensor_attributes>
concatenate(std::vector<std::shared_ptr<Tensor_attributes>>, Concatenate_attributes);
```

Concatenate attributes is a lightweight structure with inputs, outputs, and setters:

```
std::vector<std::shared_ptr<Tensor_attributes>> inputs;

std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

Concatenate_attributes&
set_axis(int64_t const value)

Concatenate_attributes&
set_in_place_index(int64_t const value)
```

---

# Convolutions

## Convolution Fprop

Convolution fprop computes:

\( response = image \* filter \)

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv\_fprop\_attributes is a lightweight structure with setters:

```
Conv_fprop_attributes&
set_padding(std::vector<int64_t>)

Conv_fprop_attributes&
set_stride(std::vector<int64_t>)

Conv_fprop_attributes&
set_dilation(std::vector<int64_t>)

Conv_fprop_attributes&
set_name(std::string const&)

Conv_fprop_attributes&
set_compute_data_type(DataType_t value)

Conv_fprop_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

* conv\_fprop

  + image
  + weight
  + padding
  + stride
  + dilation
  + compute\_data\_type
  + name

## Convolution Dgrad

Convolution dgrad computes data gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_dgrad_attributes);
```

Conv\_dgrad\_attributes is a lightweight structure with setters:

```
Conv_dgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_dgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_dgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_dgrad_attributes&
set_name(std::string const&)

Conv_dgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_dgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

* conv\_dgrad

  + filter
  + loss
  + padding
  + stride
  + dilation
  + compute\_data\_type
  + name

## Convolution Wgrad

Convolution wgrad computes weight gradient during backpropagation.

### C++ API

```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_wgrad_attributes);
```

Conv\_wgrad\_attributes is a lightweight structure with setters:

```
Conv_wgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_wgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_wgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_wgrad_attributes&
set_name(std::string const&)

Conv_wgrad_attributes&
set_compute_data_type(DataType_t value)

Conv_wgrad_attributes&
set_convolution_mode(ConvolutionMode_t mode_)
```

### Python API

* conv\_wgrad

  + image
  + loss
  + padding
  + stride
  + dilation
  + compute\_data\_type
  + name

---

# Matmul

The Matmul operation computes:

\( C[M, N] = A[M, K] \* B[K, N] \)

Last two dimensions of input dimensions are interpreted as M, N, K. All other preceding dimensions are interpreted as batch dimensions. The operation also has broadcasting capabilities which are described in [cudnn backend's matmul operation](../../../backend/v9.20.0/api/cudnn-graph-library.html#cudnn-backend-operation-matmul-descriptor "(in NVIDIA cuDNN Backend)").

## C++ API

```
std::shared_ptr<Tensor_attributes>
Matmul(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Matmul_attributes);
```

Matmul attributes is a lightweight structure with setters:

```
Matmul_attributes&
set_name(std::string const&)

Matmul_attributes&
set_compute_data_type(DataType_t value)
```

## Python API

* matmul

  + A
  + B
  + name
  + compute\_data\_type

---

# MoE Grouped Matmul

The MoE Grouped Matmul operation computes a grouped matmul operation based on given first token offset, token index, and token ks in three modes (None, Gather, and Scatter):

In None and Scatter modes:
\( Output[1, S \* topK, N] = Token[1, S \* topK, K] \* Weight[E, K, N] \)

In Gather mode:
\( Output[1, S \* topK, N] = Token[1, S, K] \* Weight[E, K, N] \)

FirstTokenOffset has shape [B \* E, 1, 1] and is used in all three modes.

TokenIndex has shape [1, S \* topK, 1] and is used in the Gather and Scatter modes.

TokenKs has shape [1, S \* topK, 1] and is used in the Scatter mode.

TopK as an int32\_t element needs to be explicitly provided in the Scatter mode.

## C++ API

```
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token, std::shared_ptr<Tensor_attributes> weight, std::shared_ptr<Tensor_attributes> first_token_offset, std::shared_ptr<Tensor_attributes> token_index, std::shared_ptr<Tensor_attributes> token_ks, moe_grouped_matmul_attribute);
```

Moe\_grouped\_matmul attributes is a lightweight structure with setters:

```
Moe_grouped_matmul&
set_name(std::string const&)

Moe_grouped_matmul&
set_mode(MoeGroupedMatmulMode_t mode)

Moe_grouped_matmul&
set_compute_data_type(DataType_t value)

Moe_grouped_matmul&
set_top_k(int32_t top_k_value)
```
