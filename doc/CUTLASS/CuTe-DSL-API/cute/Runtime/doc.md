# Runtime

Description

## API documentation

*class* cutlass.cute.runtime.\_Pointer(*\*args: Any*, *\*\*kwargs: Any*)
:   Bases: `Pointer`

    Runtime representation of a pointer that can inter-operate with various data structures,
    including numpy arrays and device memory.

    Parameters:
    :   * **pointer** (*int* *or* *pointer-like object*) -- The pointer to the data
        * **dtype** (*Type*) -- Data type of the elements pointed to
        * **mem\_space** (*\_cute\_ir.AddressSpace**,* *optional*) -- Memory space where the pointer resides, defaults to generic
        * **assumed\_align** (*int**,* *optional*) -- Assumed alignment of input pointer in bytes, defaults to None

    Variables:
    :   * **\_pointer** -- The underlying pointer
        * **\_dtype** -- Data type of the elements
        * **\_addr\_space** -- Memory space of the pointer
        * **\_assumed\_align** -- Alignment of the pointer in bytes
        * **\_desc** -- C-type descriptor for the pointer
        * **\_c\_pointer** -- C-compatible pointer representation

    \_\_init\_\_( : *pointer*, : *dtype*, : *mem\_space: cutlass.\_mlir.dialects.cute.AddressSpace = cutlass.\_mlir.dialects.cute.AddressSpace.generic*, : *assumed\_align=None*, )

    size\_in\_bytes() → int

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*

    *property* dtype*: Type[cutlass.cute.typing.Numeric]*

    *property* memspace

    align( : *min\_align: int*, : *\**, : *loc=None*, : *ip=None*, ) → cutlass.cute.typing.Pointer

*class* cutlass.cute.runtime.\_Tensor(*\*args: Any*, *\*\*kwargs: Any*)
:   Bases: `Tensor`

    \_\_init\_\_( : *tensor*, : *assumed\_align=None*, : *use\_32bit\_stride=False*, : *\**, : *enable\_tvm\_ffi=False*, )

    load\_dltensor()
    :   Lazily load the DLTensorWrapper.

        This function loads the DLTensorWrapper when needed,
        avoiding overhead in the critical path of calling JIT functions.

    mark\_layout\_dynamic(*leading\_dim: int | None = None*)
    :   Marks the tensor layout as dynamic based on the leading dimension.

        Parameters:
        :   **leading\_dim** (*int**,* *optional*) -- The leading dimension of the layout, defaults to None

        When `leading_dim` is None, the leading dimension is deduced as follows.

        1. If exactly one dimension has stride 1, that dimension is used.
        2. If multiple dimensions have stride 1 but exactly one of them has size > 1,
           that dimension is used.
        3. If multiple dimensions have stride 1 but none or more than one has size > 1,
           an error is raised.
        4. If no dimension has stride 1, all strides remain dynamic.

        When `leading_dim` is explicitly specified, marks the layout as dynamic while setting the
        stride at `leading_dim` to 1. Also validates that the specified `leading_dim` is consistent
        with the existing layout by checking that the corresponding stride of that dimension is 1.

        Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

        Returns:
        :   The tensor with dynamic layout

        Return type:
        :   [\_Tensor](#cutlass.cute.runtime._Tensor "cutlass.cute.runtime._Tensor")

    mark\_compact\_shape\_dynamic( : *mode: int*, : *stride\_order: tuple[int, ...] | None = None*, : *divisibility: int = 1*, )
    :   Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

        Parameters:
        :   * **mode** (*int*) -- The mode of the compact shape, defaults to 0
            * **stride\_order** -- Consistent with torch.Tensor.dim\_order. Defaults to None.

        Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
        It starts from the outermost to the innermost dimension.
        :type stride\_order: tuple[int, ...], optional
        :param divisibility: The divisibility constraint for the compact shape, defaults to 1
        :type divisibility: int, optional
        :return: The tensor with dynamic compact shape
        :rtype: \_Tensor

        If `stride_order` is not provided, the stride ordering will be automatically deduced from the layout.
        Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
        An error is raised if automatic deduction fails.

        If `stride_order` is explicitly specified, it does the consistency check with the layout.

        For example:
        - Layout: (4,2):(1,4) has stride\_order: (1,0) indicates the innermost dimension is 0(4:1), the outermost dimension is 1(2:4)
        - Layout: (5,3,2,4):(3,1,15,30) has stride\_order: (3,2,0,1) indicates the innermost dimension is 1(3:1), the outermost dimension is 3(4:30).

        Using torch.Tensor.dim\_order() to get the stride order of the torch tensor.
        .. code-block:: python
        a = torch.empty(3, 4)
        t = cute.runtime.from\_dlpack(a)
        t = t.mark\_compact\_shape\_dynamic(mode=0, stride\_order=a.dim\_order())

    *property* element\_type*: Type[cutlass.cute.typing.Numeric]*

    *property* memspace

    *property* size\_in\_bytes*: int*

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*

    *property* iterator

    *property* layout

    *property* shape

    *property* stride

    *property* leading\_dim
    :   Get the leading dimension of this Tensor.

        Returns:
        :   The leading dimension index or indices

        Return type:
        :   int or tuple or None

        The return value depends on the tensor's stride pattern:

        * If a single leading dimension is found, returns an integer index
        * If nested leading dimensions are found, returns a tuple of indices
        * If no leading dimension is found, returns None

    fill(*value: cutlass.cute.typing.Numeric*)

    *property* data\_ptr

    *property* dynamic\_shapes\_mask
    :   Get the mask of dynamic shapes in the tensor.

    *property* dynamic\_strides\_mask
    :   Get the mask of dynamic strides in the tensor.

cutlass.cute.runtime.\_get\_cute\_type\_str(*inp*)

*class* cutlass.cute.runtime.\_FakeTensor(*\*args: Any*, *\*\*kwargs: Any*)
:   Bases: `Tensor`

    Fake Tensor implementation as a placeholder.
    It mimics the interface of Tensor, but does not hold real data or allow indexing.
    Used for compilation or testing situations where only shape/type/layout information is needed.
    All attempts to access or mutate data will raise errors.

    \_\_init\_\_( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *stride: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, : *use\_32bit\_stride: bool = False*, : *compact: bool = False*, )

    *property* mlir\_type*: cutlass.\_mlir.ir.Type*

    *property* element\_type*: Type[cutlass.cute.typing.Numeric]*

    *property* memspace

    *property* iterator

    *property* shape

    *property* stride

    *property* leading\_dim

    *property* dynamic\_shapes\_mask

    *property* dynamic\_strides\_mask

    fill(*value: cutlass.cute.typing.Numeric*)

cutlass.cute.runtime.make\_fake\_compact\_tensor( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *stride\_order: tuple[int, ...] | None = None*, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, : *use\_32bit\_stride: bool = False*, )
:   Create a fake tensor with the specified shape, element type, and a compact memory layout.

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) -- Data type of the tensor elements.
        * **shape** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) -- Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
        * **stride\_order** (*tuple**[**int**,* *...**]**,* *optional*) -- Order in which strides (memory layout) are assigned to the tensor dimensions.
          If None, the default layout is left-to-right order (known as column-major order for flatten layout).
          Otherwise, it should be a permutation order of the dimension indices.
          The mode with stride\_order 0 is the fastest changing (leading) dimension, and N-1 is the slowest changing.
        * **memspace** (*AddressSpace**,* *optional*) -- Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
        * **assumed\_align** (*int**,* *optional*) -- Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.
        * **use\_32bit\_stride** (*bool**,* *optional*) -- Whether to use 32-bit stride for dynamic dimensions. If True and the total size of the
          layout (cosize(layout)) fits within int32, then dynamic strides will use 32-bit integers for improved performance.
          Only applies when dimensions are dynamic. Defaults to False.

    Returns:
    :   An instance of a fake tensor with the given properties and compact layout.

    Return type:
    :   [\_FakeTensor](#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

    **Examples:**

```python
    @cute.jit
    def foo(x: cute.Tensor):
        ...

    x = make_fake_compact_tensor(
        cutlass.Float32, (100, cute.sym_int32(divisibility=8)), stride_order=(1, 0)
    )

    # Compiled function will take a tensor with the type:
    #   tensor<ptr<f32, generic> o (100,?{div=8}):(?{i32 div=8},1)>
    compiled_foo = cute.compile(foo, x)

    # Default stride order is left-to-right order (0, 1, ..., n-1)
    y = make_fake_compact_tensor(cutlass.Float32, (8, 3, 2)) # y.stride == (1, 8, 24)
    ```

cutlass.cute.runtime.make\_fake\_tensor( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *shape: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *stride: tuple[int | cutlass.cute.typing.SymInt, ...]*, : *\**, : *memspace: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.gmem*, : *assumed\_align: int | None = None*, )
:   Create a fake tensor with the specified element type, shape, and stride.

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) -- Data type of the tensor elements.
        * **shape** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) -- Shape of the tensor, consisting of static (int) or dynamic (SymInt) dimensions.
        * **stride** (*tuple**[**Union**[**int**,* *SymInt**]**,* *...**]*) -- Stride of the tensor, consisting of static (int) or dynamic (SymInt) values.
        * **memspace** (*AddressSpace**,* *optional*) -- Memory space where the fake tensor resides. Defaults to AddressSpace.gmem.
        * **assumed\_align** (*int**,* *optional*) -- Assumed byte alignment for the tensor data. If None, the default alignment is the dtype width, & at least 1 byte.

    Returns:
    :   An instance of a fake tensor with the given properties.

    Return type:
    :   [\_FakeTensor](#cutlass.cute.runtime._FakeTensor "cutlass.cute.runtime._FakeTensor")

*class* cutlass.cute.runtime.\_FakeStream(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)
:   Bases: `object`

    A fake stream that can be used as a placeholder for a stream in compilation.

    When use\_tvm\_ffi\_env\_stream is True and the function is compiled with TVM-FFI,
    the argument will be skipped from the function signature and we pass in
    this value through the environment stream obtained from caller context
    (e.g. torch.cuda.current\_stream()).

    \_\_init\_\_(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)

    use\_tvm\_ffi\_env\_stream*: bool*

cutlass.cute.runtime.make\_fake\_stream(*\**, *use\_tvm\_ffi\_env\_stream: bool = False*)
:   Create a fake stream that can be used as a placeholder for a stream in compilation.

    When use\_tvm\_ffi\_env\_stream is True and the function is compiled with TVM-FFI,
    the argument will be skipped from the function signature and we pass in
    this value through the environment stream obtained from caller context
    (e.g. torch.cuda.current\_stream()). This can speedup the calling process
    since we no longer need to do stream query in python.

    Parameters:
    :   **use\_tvm\_ffi\_env\_stream** (*bool*) -- Whether to skip this parameter use environment stream instead.

cutlass.cute.runtime.from\_dlpack( : *tensor\_dlpack*, : *assumed\_align=None*, : *use\_32bit\_stride=False*, : *\**, : *enable\_tvm\_ffi=False*, : *force\_tf32=False*, ) → cutlass.cute.typing.Tensor
:   Convert from tensor object supporting \_\_dlpack\_\_() to a CuTe Tensor.

    Parameters:
    :   * **tensor\_dlpack** (*object*) -- Tensor object that supports the DLPack protocol
        * **assumed\_align** (*int**,* *optional*) -- Assumed alignment of the tensor (bytes), defaults to None,
          if None, will use the element size bytes as the assumed alignment.
        * **use\_32bit\_stride** (*bool**,* *optional*) -- Whether to use 32-bit stride, defaults to False. When True, the dynamic
          stride bitwidth will be set to 32 for small problem size (cosize(layout) <= Int32\_max) for better performance.
          This is only applied when the dimension is dynamic.
        * **enable\_tvm\_ffi** (*bool**,* *optional*) -- Whether to enable TVM-FFI, defaults to False. When True, the tensor will be converted to
          a TVM-FFI function compatible tensor.
        * **force\_tf32** (*bool**,* *optional*) -- Whether to force the element type to TFloat32 if the element type is Float32.

    Returns:
    :   A CuTe Tensor object

    Return type:
    :   Tensor

    **Examples:**

```python
    import torch
    from cutlass.cute.runtime import from_dlpack
    x = torch.randn(100, 100)
    y = from_dlpack(x)
    y.shape
    # (100, 100)
    type(y)
    # <class 'cutlass.cute.Tensor'>
    ```

cutlass.cute.runtime.make\_ptr( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *value: int | \_Pointer*, : *mem\_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic*, : *assumed\_align=None*, ) → cutlass.cute.typing.Pointer
:   Create a pointer from a memory address

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) -- Data type of the pointer elements
        * **value** (*Union**[**int**,* *ctypes.\_Pointer**]*) -- Memory address as integer or ctypes pointer
        * **mem\_space** (*AddressSpace**,* *optional*) -- Memory address space, defaults to AddressSpace.generic
        * **align\_bytes** (*int**,* *optional*) -- Alignment in bytes, defaults to None

    Returns:
    :   A pointer object

    Return type:
    :   Pointer

```python
    import numpy as np
    import ctypes

    from cutlass import Float32
    from cutlass.cute.runtime import make_ptr

    # Create a numpy array
    a = np.random.randn(16, 32).astype(np.float32)

    # Get pointer address as integer
    ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Create pointer from address
    y = make_ptr(cutlass.Float32, ptr_address)

    # Check properties
    print(y.element_type)
    print(type(y))  # <class 'cutlass.cute.Pointer'>
    ```

cutlass.cute.runtime.nullptr( : *dtype: Type[cutlass.cute.typing.Numeric]*, : *mem\_space: cutlass.cute.typing.AddressSpace = cutlass.cute.typing.AddressSpace.generic*, : *assumed\_align=None*, ) → cutlass.cute.typing.Pointer
:   Create a null pointer which is useful for compilation

    Parameters:
    :   * **dtype** (*Type**[**Numeric**]*) -- Data type of the pointer elements
        * **mem\_space** (*AddressSpace**,* *optional*) -- Memory address space, defaults to AddressSpace.generic

    Returns:
    :   A null pointer object

    Return type:
    :   Pointer

*class* cutlass.cute.runtime.TensorAdapter(*arg*)
:   Bases: `object`

    Convert a DLPack protocol supported tensor/array to a cute tensor.

    \_\_init\_\_(*arg*)

cutlass.cute.runtime.find\_runtime\_libraries( : *\**, : *enable\_tvm\_ffi: bool = True*, ) → List[str]
:   Find the runtime libraries that needs to be available for loading modules.

    Parameters:
    :   **enable\_tvm\_ffi** (*bool**,* *optional*) -- Whether to enable TVM-FFI.

    Returns:
    :   A list of runtime libraries that needs to be available for loading modules.

    Return type:
    :   list

cutlass.cute.runtime.load\_module(*file\_path: str*, *\**, *enable\_tvm\_ffi: bool = False*)
:   Load a module from a file path.

    Parameters:
    :   * **file\_path** (*str*) -- The path to the module file
        * **enable\_tvm\_ffi** (*bool**,* *optional*) -- Whether to enable TVM-FFI, defaults to True. When True, the module will be loaded as a TVM-FFI module.

    Returns:
    :   A module object

    Return type:
    :   module
