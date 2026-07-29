# TensorCore Integration Architecture

Maintenance 15 selects exact published TensorCore `0.22.0` commit
`19bfae35fbc773b55cac7bcd659dda57c4dee6d6`, tree
`53aa10520a50c0714e79c685d814cbae1b6f7740`.

TensorDSLab consumes the public TensorCore roots for:

- `Coordinates`, `TensorAxis`, and `OffsetAxis`;
- `TensorFieldSpec`, `TensorField`, `TensorKernelSpec`, `TensorKernel`, and
  `TensorCollection`;
- constrained integer/float values;
- `RngKey`, `RngElements`, `RngAddress`, and `CounterRng`; and
- Gaussian, Poisson, and Multinomial Distributions.

TensorDSLab directly composes this substrate. It does not fork generic
validation, movement, coordinate windows, flattening, tensor snapshots, RNG
encoding, or distribution sampling. Public-root imports are required; private
TensorCore imports are forbidden.

TensorDSLab adds exact semantic quantity Axis/Spec leaves and detector Product
laws. Most-derived validation must run after every Spec, Field, Kernel, and
Collection movement. `BitDepth` remains an exact integer kernel while
floating calculation policy is prepared separately.

Maintenance 16 centralizes reusable package-owned admission in the private,
export-empty `tensor_dslab.common.requirements` package. Specs own dtype, Unit,
axis, and operation geometry. Fields and Kernels own exact-Spec and represented
value laws. Product preparation retains cross-object alignment, conversion,
capacity planning, and scientific relationships; Product validation retains
completed-result and storage relationships. Requirement functions validate
without normalizing, converting, aligning, mutating, or returning replacement
objects.

The five public typed Kernel collections are defined beside their vocabulary
in each Product's singular `kernel.py`; their supported facade names and order
are unchanged. `common/alignment.py` owns only alignment, permutation,
conversion, and one-Kernel materialization mechanics. Tensor and RNG-address
span preflight lives in the private capacity requirements owner.

Source Fields may use any application semantic class whose exact Spec is a
`QuantityFieldSpec` and satisfies the Product source equation. This is why
source-taking public methods use `tuple[TensorField[Any], ...]`: TensorCore's
Spec parameter is invariant, and runtime semantic admission is deliberate.

The dependency statement is exact-commit evidence, not a broad compatibility
claim. Integrated CUDA remains deferred.
