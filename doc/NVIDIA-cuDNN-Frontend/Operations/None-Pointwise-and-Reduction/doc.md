# Pointwise and Reduction

## Pointwise

The pointwise operation performs an elementwise operation between two tensors. The operation used is controlled by pointwise mode `cudnn_frontend::PointwiseMode_t`.

### C++ API

```
std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);
```

The pointwise mode dictates the API among the choices above. Refer to the documentation of `cudnn_frontend::PointwiseMode_t` for details.

Pointwise attributes is a lightweight structure with setters:

```
Pointwise_attributes&
set_mode(PointwiseMode_t)

Pointwise_attributes&
set_axis(int64_t)

Pointwise_attributes&
set_relu_lower_clip(float)

Pointwise_attributes&
set_relu_upper_clip(float)

Pointwise_attributes&
set_relu_lower_clip_slope(float)

Pointwise_attributes&
set_name(std::string const&)

Pointwise_attributes&
set_compute_data_type(DataType_t value)
```

### Python API

* add

  + a
  + b
  + compute\_data\_type
  + name
* bias

  + input
  + bias
  + compute\_data\_type
  + name
* rsqrt

  + input
  + compute\_data\_type
  + name
* sub

  + a
  + b
  + compute\_data\_type
  + name
* mul

  + a
  + b
  + compute\_data\_type
  + name
* scale

  + input
  + scale
  + compute\_data\_type
  + name
* relu

  + input
  + compute\_data\_type
  + name
* gelu

  + input
  + compute\_data\_type
  + name
* elu

  + input
  + compute\_data\_type
  + name
* cmp\_gt

  + input
  + comparison
  + compute\_data\_type
  + name

## Reduction

The reduction operation reduces an input tensor using an operation controlled by `cudnn_frontend::ReductionMode_t`. The dimensions in input tensors to reduce are deduced using output tensor dimensions.

### C++ API

```
std::shared_ptr<Tensor_attributes>
reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes);
```

Reduction attributes is a lightweight structure with setters:

```
Reduction_attributes&
set_mode(ReductionMode_t)

Reduction_attributes&
set_name(std::string const&)

Reduction_attributes&
set_compute_data_type(DataType_t value)
```

---

# Resampling

## Resampling Forward

The resample operation represents the resampling of the spatial dimensions of an image to a desired value.

The output array contains two tensors:

* The resampled output tensor.
* The computed index tensor.

Note

The index tensor is only output in training mode of max pooling. It can be fed to backward pass for faster performance.

### Resample Attributes

The Resample\_attributes class is used to configure the resampling operation. It provides the following setters:

```
# The resampling mode, such as average pooling, max pooling, bi-linear, or cubic.
auto set_resampling_mode(ResampleMode_t const& value) -> Resample_attributes&;

# The padding mode, such as zero or neg infinity.
auto set_padding_mode(PaddingMode_t const& value) -> Resample_attributes&;

# The window size to be used for the resampling operation.
auto set_window(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_window(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# The stride values to be used for the resampling operation.
auto set_stride(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_stride(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# The padding values to be applied before and after the resampling input.
auto set_pre_padding(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_pre_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;
auto set_post_padding(std::vector<int64_t> const& value) -> Resample_attributes&;
auto set_post_padding(std::vector<cudnnFraction_t> const& value) -> Resample_attributes&;

# Indicates that index should be generated (such as during training).
# If false, the index output will be nullptr.
auto set_generate_index(bool const value) -> Resample_attributes&;

# DEPRECATED.
# Calls set_generate_index(!value) (note the negation of `value`).
auto set_is_inference(bool const value) -> Resample_attributes&;
```

For more information on exact support surfaces across different versions, refer to [ResampleFwd](../developer/graph-api.html#resamplefwd-runtime-fusion-engine) in the *Frontend Developer Guide*.

Python API for resampling forward will be supported soon.

## Resampling Backward

To be supported soon.

---

# Slice

The slice operation extracts a portion of a tensor:

\( Y = X[start\_0:end\_0, start\_1:end\_1, ..., start\_n:end\_n] \)

Where \(X\) is the input tensor, \(Y\) is the output tensor, and \(start\_i\) and \(end\_i\) are the start and end indices for the \(i\)-th dimension.

The operation allows for flexible slicing across any number of dimensions, supporting Python-style slice syntax including start, stop, and step parameters.

## C++ API

```
std::shared_ptr<Tensor_attributes>
Slice(std::shared_ptr<Tensor_attributes> input, Slice_attributes);
```

Slice attributes is a lightweight structure with setters:

```
Slice_attributes&
set_slices(std::vector<std::pair<int64_t, int64_t>> const value)

Slice_attributes&
set_name(std::string const&)

Slice_attributes&
set_compute_data_type(DataType_t value)
```

## Python API:

* slice

  + input

    - The input tensor to be sliced
  + slices

    - A list of Python slice objects, one for each dimension
  + name

    - Optional name for the operation
  + compute\_data\_type

    - Optional compute data type for the operation

Example usage:

```
# Create an input tensor

input_tensor = graph.tensor(dims = [4, 8, 16])

# Perform slicing
sliced_tensor = graph.slice(input_tensor,
                            slices=[slice(1, 3), slice(2, 6), slice(0, 16)],
                            name="my_slice",
                            compute_data_type=cudnn.float32)
```
