# TensorDSLab Rebuild Architecture

The current rebuild target is Maintenance 15's spec-composed Product parts
bin. The normative details are in:

- [Maintenance 15 architecture](../implementation/maintenance_15_spec_composed_products_and_application_boundary.md)
- [Maintenance 15 executable work order](../implementation/maintenance_15_execution_work_order.md)
- [Tensor integration](tensors.md)
- [Scientific parity](../parity.md)

## Representation Stack

```text
TensorCore Coordinates
  -> TensorDSLab semantic Axis
  -> TensorCore FieldSpec / KernelSpec
  -> TensorDSLab quantity-aware exact semantic Spec
  -> TensorCore TensorField / TensorKernel
  -> TensorDSLab Product / coefficient leaf
```

`ExampleAxis` and `ChannelAxis` are representation-polymorphic semantic roles.
`TimeAxis` and `FrequencyAxis` are `QuantityAxis` leaves whose exact integer
Coordinates remain distinct from physical scale and Unit.

Each Product and coefficient is a fieldless exact semantic leaf. Quantity
state lives in Specs, never on tensor values. Configs compose exact output
Specs and typed Kernel collections. A fresh prepared Config owns immutable
alignment/source-provenance facts; Runtime, Plan, generic Product, generic
Config, and generic readout abstractions are absent.

## Ownership

TensorCore owns generic Coordinates, Axis/Spec/Field/Kernel/Collection
mechanics, movement, addressed RNG, and Distributions. TensorDSLab owns
physical units, semantic roles, Product and coefficient Specs, typed
collections, source/unit/geometry laws, scientific equations, stochastic role
keys, preparation policy, and completed Product validation. Applications own
workflow composition and persistence.

## Execution Invariants

- Sources are never moved implicitly.
- Preparation completes unit, device, dtype, role, coordinate, geometry,
  storage-capacity, and RNG-capacity policy before effects.
- Kernel conditioning coordinates may be reordered into output order while
  source coordinates must already be equivalent.
- Prepared source provenance is retained by identity and later enforced by
  positional structural equality.
- Deterministic transforms preserve ordinary autograd connectivity.
- Every generated Product is fresh, contiguous, and source-disjoint.
- Stochastic Products use public TensorCore addressed Distributions only.

Earlier stage and maintenance work orders remain immutable historical evidence
for the contracts they closed; they do not override this current target.
