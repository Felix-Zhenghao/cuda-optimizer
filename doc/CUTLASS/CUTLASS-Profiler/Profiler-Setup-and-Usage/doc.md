![ALT](img/ALT.png)

# CUTLASS Profiler

The CUTLASS Profiler is a command-line driven test and profiling environment for CUTLASS computations
defined in the CUTLASS Instance Library. The CUTLASS Profiler is capable of executing each GEMM, Sparse Gemm,
Conv2d, and Conv3d kernel.

The CUTLASS Profiler may be compiled with:

```bash
$ make cutlass_profiler -j
```

To limit compilation time, only one tile size (typically 128x128) and threadblock cluster size (typically 2x1x1) is instantiated for each data type,
math instruction, and layout. To instantiate all sizes, set the following environment variable when running CMake from an
empty `build/` directory.

```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all  -DCUTLASS_UNITY_BUILD_ENABLED=ON
...
$ make cutlass_profiler -j
```

Enabling the unity build places multiple kernel instances in one compilation unit, thereby reducing size of the compiled
binary and avoiding linker limitations on some platforms.

The CUTLASS Profiler sources are stored in:

```
tools/
  profiler/
```

# Emitting kernels via `emit_kernel_listing.py`

We provide a Python script `emit_kernel_listing.py` that allows a user to selectively test a subset of profiler-based kernels stamped out in `generator.py`. A unique benefit to generate kernels and test via this script is that it can feed a series of runtime arguments, such as different `M`/`N`/`K` and `alpha`/`beta`, to each kernel, instead of relying on a single default value. It also properly generates runtime datatype and cluster shapes for certain kernels to help reduce the generated kernel count and accordingly the total compilation time. An interested user may refer to [emit\_kernel\_listing.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/emit_kernel_listing.py) for details. To enable this new feature, a user should add `-DCUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=ON` when building CUTLASS profiler.

## Instantiating more kernels with Hopper

With Hopper (SM90), you will need to use an additional flag,
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations,
which unlike previous architectures, will be in the order of millions of kernels.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler employs a four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations. This global instantiation level decides the behavior of multiple "generators" by defining how many and which combinations of configurations are produced. If a global instantiation level contains fewer than four digits, it can be padded with leading zeros to ensure it is four digits long. Each of the four digits in the global level corresponds to a specific category that influences kernel generation, from right to left:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Schedule Pruning**

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

* **Level 0**: Only `(1, 2, 1)` cluster shape.
* **Level 1**: Clusters with 2 CTAs.
* **Level 2**: Clusters with 1 or 2 CTAs.
* **Level 3**: Clusters with 1, 2, or 4 CTAs.
* **Level 4**: Clusters with 1, 2, 4, or 8 CTAs.
* **Level 5**: Clusters with 1, 2, 4, 8, or 16 CTAs.

The MMA multipliers are combined with MMA instruction shapes (WGMMA shapes) to form CTA shapes. The levels for MMA multipliers determine the configurations generated for different data types.

* **Levels [0, 3]**: Control the specific configurations generated for various data types.
* **Level 9**: Activates exhaustive mode, generating all possible configurations.

Higher levels encompass a broader range of CTA configurations, resulting in more comprehensive kernel generation.

Instruction shape levels control the selection of WGMMA shapes used in kernel generation:

* **Level 0**: Generates the "default" shape only.
* **Level 1**: Includes additional shapes for unpruned cases, specifically for TF32 data type.
* **Level 2**: Includes shapes that are powers of 2.
* **Level 3**: Includes all other shapes.

The detailed definition of the three instantiation levels controlling cluster shape, MMA shape multiplier, and instruction shape can be found in [sm90\_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_shapes.py).

Schedule pruning levels decide the epilogue schedule and mainloop schedule to stamp out a kernel instance. As defined in `get_valid_schedules` in [sm90\_utils.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_utils.py),

* **Level >= 1**: Indicates that no pruning is being applied.
* **Level 0**: Indicates pruning according to existing [generator.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py) behavior.

An instantiation level `500`, which is padded to `0500`, thus indicates:

* **Instruction Shapes**: At level 0, generating only the "default" shape.
* **MMA Multipliers**: At level 0, generating only one multiplier, `(2, 1, 4)`.
* **Cluster Sizes**: At level 5, allowing for clusters with 1, 2, 4, 8, or 16 CTAs.
* **Schedule Pruning**: At level 0, where pruning is applied according to the existing `generator.py` behavior.

## Instantiating more MMA shapes with Hopper

When instantiating more tile shapes, specially non-power-of-2 Tile-N shapes, make sure to enable `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES`.
This may lead to some increase in per-kernel compilation times.
When `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` is set, then `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES` is enabled by default.

## Mixed input data type kernels for Hopper

With Hopper (SM90), the kernel generator will generate the following combinations of mixed input data types ("mixed dtype"):

| dtype(A) | dtype(B) |
| --- | --- |
| e4m3 | f16, bf16 |
| e5m2 | f16, bf16 |
| int8 | f16, bf16 |
| uint8 | f16, bf16 |
| int4 | f16, bf16 |
| int4 | e4m3, e5m2 |
| uint4 | f16, bf16 |
| int2 | f16, bf16 |
| uint2 | f16, bf16 |

For each mixed dtype kernel, the kernel generator will generate combinations of three different running modes:

* Convert-only
* Scale-only
* Scale-with-zero-point-shifting

For {4-bits-dtype, 8-bits-dtype} x 16-bits-dtype, the kernel generator will further generate kernels using shuffled layouts for the narrow data type matrix, which may have a better performance compared to its non-shuffle counter parts.

## Instantiating more kernels with Blackwell

Blackwell (SM100) and Blackwell Ultra similarly support
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="100f" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm100_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler uses the same four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations for Blackwell as well:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Data Type and Schedule Pruning**

Note for Blackwell kernels an MMA shape multiplier is no longer necessary since Blackwell kernels do not have a different
ping pong or cooperative schedule. The profiler ignores this digit when instantiating.

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

* **Level 0**: Only dynamic cluster shapes.
* **Level 1**: For 1SM kernels `(1, 1, 1)` and `(2, 1, 1)` for 2SM kernels.
* **Level 2**: For 1SM kernels we also have `(1, 2, 1)` and for 2SM we have `(2, 2, 1)` and `(4, 1, 1)`.
* **Level 3**: For 1SM kernels we have `(1, 4, 1)` and for 2SM we have `(2, 4, 1)` and `(4, 2, 1)`.
* **Level 4**: For 1SM kernels we have `(4, 4, 1)` and for 2SM we have `(4, 4, 1)`.
* **Level 5**: For 1SM kernels we have `(2, 1, 1)`.
* **Level 6**: For 1SM kernels we have `(2, 2, 1)` and `(4, 1, 1)` and for 2SM kernels we have `(8, 1, 1)`.
* **Level 7**: For 1SM kernels we have `(2, 4, 1)` and `(4, 2, 1)`
* **Level 8**: For 1SM kernels we have `(1, 8, 1)` and `(8, 1, 1)`

Instruction shape levels control the selection of MMA shapes used in kernel generation:

* **Level 0**: Generates the "default" shape only.
* **Level 1**: Includes additional shapes for FP8, FP6, and FP4 as well as MX and NVFP4.
* **Level 2**: Includes small tile shapes.
* **Level 3**: Includes some non-power of 2 shapes.
* **Level 4**: Includes further small tile shapes and non-power of 2 shapes.
* **Level 5**: Includes all shapes.

The detailed definition of the three instantiation levels controlling cluster shape and instruction shape can be found in [sm100\_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm100_shapes.py).

## CUTLASS Profiler usage

The CUTLASS Profiler usage statement may be obtained by executing `cutlass_profiler --help` and appears as follows.

```bash
CUTLASS Performance Tool
usage:

    cutlass_profiler [options]

  --help

  --mode=<string>                                  Cutlass profiler execution mode.
                                                    --mode=profile    regular verification and profiling (default)
                                                    --mode=dry_run    no kernels are launched or workspaces allocated
                                                    --mode=enumerate  lists all operation kind and operations
                                                    --mode=trace      executes a single device-side computation with
                                                                       no other kernel launches

  --device-info                                    Prints information on all GPUs present in the system

  --operation=<operation_kind>                     CUTLASS operation to profile.

  --kernels=<string_list>                          Filter operations by kernel names. For example, call all kernels with
                                                   ("s1688" and "nt") or ("s844" and "tn" and "align8") in their
                                                   operation name using --kernels="s1688*nt, s884*tn*align8"

  --kernels-file=<path>                            Same behavior as `kernels`, but kernel names are specified in a file with
                                                   one kernel name on each line. Set of profiled kernels is the union of kernels
                                                   specified here and those specified in `kernels`.

  --ignore-kernels=<string_list>                   Excludes kernels whose names match anything in this list.

Device:
  --device=<int>                                   CUDA Device ID

  --compute-capability=<int>                       Override the compute capability.

  --llc-capacity=<capacity in KiB>                 Capacity of last-level cache in kilobytes. If this is non-zero,
                                                   profiling phases cycle through different input tensors to induce
                                                   capacity misses in the L2.

  --allocations=<name>:<device>,<name>:<device>    Pairs of allocation names to devices. If <device> is negative,
                                                   the execution device is used

Initialization:
  --initialization=<bool>                          Enables initialization (default: true). If false, device memory is
                                                   not initialized after allocation.

  --initialization-provider=<provider>             Selects initialization provider {host, device*}. (default: '*')

  --dist=<distribution>                            Data distribution of input tensors {uniform*, gaussian, identity, sequential}
                                                    --dist=uniform,min:<double>,max:<double>,scale:<integer>
                                                    --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>
                                                    --dist=sequential,start:<double>,delta:<double>,scale:<integer>
                                                    --dist=identity

  --seed=<int>                                     Random number generator seed. Used to enforce deterministic
                                                   initialization.

Library:
  --library-algo-mode=<mode>                       Indicates algorithm mode used to call libraries such as cuBLAS and cuDNN.
                                                   mode={default*,matching,best}

  --library-algos=<range-list>                     If --algorithm-mode=best, permits specifying a selection of algorithms.

Profiling:
  --workspace-count=<workspace count>              Number of discrete workspaces maintained to avoid cache-resident
                                                 If zero (default), the amount is chosen for each workload based on
                                                 capacity of the last-level cache.

  --profiling-iterations=<iterations>              Number of iterations to profile each kernel. If zero, kernels
                                                   are launched up to the profiling duration. If non-zero, this
                                                   overrides `profiling-duration` and `min-iterations`.

  --profiling-duration=<duration>                  Time to spend profiling each kernel (ms). Overriden by
                                                   `profiling-iterations` when `profiling-iterations` != 0.
                                                   Note that `min-iterations` must also be satisfied.

  --min-iterations=<iterations>                    Minimum number of iterations to spend profiling each kernel, even if
                                                   `profiling-duration` has been met.

  --warmup-iterations=<iterations>                 Number of iterations to execute each kernel prior to profiling (default: 10).

  --use-cuda-graphs=<bool>                         If true, kernels are launched in a CUDA graph. Useful when the kernel launch time is a bottleneck.

  --sleep-duration=<duration>                      Number of ms to sleep between profiling periods (ms).

  --profiling-enabled=<bool>                       If true, profiling is actually conducted.

Verification:
  --verification-enabled=<bool>                    Whether to perform verification checks.

  --epsilon=<error>                                Error threshold. Setting to zero (default) requires
                                                   bit-level equivalence.

  --nonzero-floor=<floor>                          Results whose absolute value is less than this quantity
                                                   are treated as zero for comparisons.

  --save-workspace=<string>                        Specifies when to save the GEMM inputs and results to the filesystem.
                                                    --save-workspace=never      never save workspace (default)
                                                    --save-workspace=incorrect  save workspace for incorrect results
                                                    --save-workspace=always     always save workspace

  --verification-providers=<providers>             List of providers used to verify result. (default: '*')
                                                   Gemm verification-providers {cublas*}
                                                   Conv2d verification-providers {cudnn*, device*, host}

Report:
  --append=<bool>                                  If true, result is appended to possibly existing file. Otherwise,
                                                   any existing file is overwritten.

  --output=<path>                                  Path to output file for machine readable results. Operation kind and '.csv' is appended.

  --junit-output=<path>                            Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.

  --report-not-run=<bool>                          If true, reports the status of all kernels including those that
                                                   do not satisfy the given arguments.

  --tags=<column:tag,...>                          Inserts leading columns in output table and uniform values for each
                                                   column. Useful for generating pivot tables.

  --verbose=<bool>                                 Prints human-readable text to stdout. If false, nothing is written to stdout.

About:
  --version                                        CUTLASS 2.4.0 built on Nov 19 2020 at 11:59:00

Operations:

     gemm                                          General matrix-matrix product. D = alpha * A*B + beta * C
     spgemm                                        Structured sparse GEMM. D = alpha * A*B + beta * C
     conv2d                                        Conv2d operation. Output(Tensor4D) = alpha * Input(Tensor4D) * Filter(Tensor4D) + beta * Input(Tensor4D)
     conv3d                                        Conv3d operation. Output(Tensor5D) = alpha * Input(Tensor5D) * Filter(Tensor5D) + beta * Input(Tensor5D)

For details about a particular function, specify the function name with --help.

Example:

  $ cutlass_profiler --operation=Gemm --help

  $ cutlass_profiler --operation=Conv3d --help

  $ cutlass_profiler --operation=Conv2d --help
```

