# Validation

Validation proves public contracts, scientific behavior, and package ownership
at the boundary that owns each claim. Tests should exercise supported use and
must not mirror private implementation structure or harden the package against
callers who deliberately leave the public API.

## Current State

Stage 2 is Merged / Closed at
`e8c62caf001ee7f58f766d7234747ed1d9a21e35`. Maintenance 1 is Merged / Closed
at `3af8ab4acf834b07e3d027fb530e5f12934999a5`. Those commits remain the
historical TensorCore `0.6` production baseline.

[Stage 3: TensorCore 0.7 Product Foundation](implementation/stage_3_tensorcore_0_7_product_foundation.md)
is Merged / Closed. Exact implementation candidate
`9250192587d1e05e71f09c9cda4ba9d0bce09bde` passed fixed-commit Validation
and independent Review, and Review's post-merge closeout produced `main`
`97e17c3177ac217aeb42a077db78f4bd223d51fa`. Design accepted the final Stage 3
closeout on clean `main` at
`5ff13eb3c0735abfda454a334be59faac35259c2`. The accepted production package
now implements the product-centered TensorCore `0.7` foundation described in
[Rebuild Architecture](architecture/rebuild.md). The recorded two CUDA skips
remain qualifications rather than GPU evidence.

[Stage 4 work order](implementation/stage_4_deterministic_waveform_products.md)
is Merged / Closed. Exact implementation candidate
`3eb8ad19a36308ca2b73d41d219a7a3b4b46c1da` passed fixed-commit Validation
and independent Review; Review's clean fast-forward closeout produced `main`
`b3ebfcd9473537dd385195afea374bd2f426c6c0`. Design independently repeated the
post-merge full-suite, exact-archive, static-typing, import-isolation, and
artifact gates without finding an issue. The accepted production package now
contains the private pure, analog, and digitized producers with eager
functionality first. The recorded three CUDA skips are qualifications rather
than GPU evidence.

The focused
[Stage 5 work order](implementation/stage_5_readout_rng_and_stochastic_noise.md)
is Merged / Closed through exact implementation candidate
`538089910be0fcaceff363c43e41e92e87af2efd` and Review's evidence-only
closeout `c6a506d3658b24197806b9e230480211a254a35a`. Fixed-commit Validation,
independent Review, and Design's post-merge audit found no unresolved issue in
the private positional RNG or complete exact-zero, IID-white, and
caller-supplied PSD noise producer. The five conditional CUDA skips are
qualifications rather than GPU evidence. Measured fusion remains a later
stage.

The focused
[Stage 6 work order](implementation/stage_6_charge_simulation.md) is
Merged / Closed through exact implementation candidate
`fb8d15e8658d6f72dfc1bbfbc2bf6a14a6b39b58` and Review's evidence-only
closeout `ea979862b05f4ef543f6971c86641df317232479`. Fixed-commit Validation,
independent Review, and Design's post-merge audit found no unresolved issue in
the aggregate samplers, dark counts, timing jitter, fixed-`K` cascade, S1/S2
ledgers, overflow diagnostics, smearing, or complete private Charge producer.
The full suite ran 174 tests: 164 passed and 10 conditional CUDA tests skipped;
the focused Stage 6 run executed 65 tests: 60 passed and 5 CUDA tests skipped.
Pyright `1.1.408` reported no findings. This is eager CPU-only evidence because
CUDA was unavailable. It adds no public `simulate_readout(...)` surface.

## Closed Maintenance 2 RNG And Module-Migration Evidence

Maintenance 2 is Merged / Closed through exact implementation candidate
`89a188abe330c06aa0b54c27cd61ac32a4fe9f63` and Design closeout
`9cbf8af3692740cd8e0bfbd1734d7ea91d95806a`. Its TensorCore dependency gate
is closed by published version `0.9.0` at exact commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`, which provides generic `RngKey`,
`CounterRng`, `Threefry4x32`, `logical_positions`, public `uniform`,
`gaussian`, `poisson`, and `binomial`, plus the focused
`require_same_dtype` relationship. TensorDSLab independently selected and
probed that exact commit, and the closed implementation pins it through public
package-root imports. The historical consumer proposal is fulfilled. Its full
suite ran 157 tests: 148 passed and 9 conditional CUDA tests skipped. Pyright
reported no diagnostics. CUDA was unavailable, so the recorded evidence is
eager CPU-only and makes no GPU or cross-backend claim.

The accepted Maintenance 2 evidence proves:

- exact dependency pin and public package-root imports only;
- TensorCore owns Random123 known-answer vectors, raw-word generation,
  logical positions, fixed-point uniforms, internal Box-Muller, parameterized
  Gaussian draws, Poisson inversion/PTRS, binomial inversion/BTRS, their
  generic numerical domains and exhaustion behavior, and generic
  seed/key/address validation;
- TensorDSLab imports no protected TensorCore RNG mechanics and duplicates no
  generic engine or promoted distribution implementation;
- TensorCore evidence covers public `uniform(...)`/`gaussian(...)`
  ordinal-plus-count bounds, Gaussian exact zero-scale/identity/affine
  branches, no arbitrary broadcasting, gradient-bearing-law rejection,
  finite-output envelope, fresh result storage, and public
  `poisson(...)`/`binomial(...)` value domains, deterministic branches,
  mappings, word schedules, and exhaustion;
- Charge retains the unchanged upward-rounded Stage 6 analytic smearing check
  over the real ledger bound; separately, that bound is floored to the
  greatest target-dtype ledger it covers and the resulting same-device scale
  is checked against TensorCore's public Gaussian prepared-scale finite-output
  envelope before any Charge effect, RNG request, or Charge-result write;
- the frozen `K=0` adjacent boundaries remain exact (`float32` accepted
  `0x1.f61fea0000000p+98`, rejected `0x1.f61fec0000000p+98`; `float64`
  accepted `0x1.51e4a059b7cf4p+994`, rejected
  `0x1.51e4a059b7cf5p+994`), while the contextual `L=24` float32 pair accepts
  `0x1.f61fd20000000p+98` and rejects `0x1.f61fd40000000p+98`;
- exact positive/negative maximum-radius results are forced through a local
  concrete `CounterRng` test double's protected `_generate_block(...)` hook
  without overriding a public method; the negative result clips to zero, and
  the rejected contextual neighbor fails before the hook or an earlier enabled
  Charge effect can consume words;
- exact `RngKey` fields on the eight stochastic leaf config classes, including
  two fields on each crosstalk config, namespace `0x54445331`, and default
  streams `1` through `10` in the accepted append-only mapping;
- keys participate in config equality and `repr`, exact-key overrides work,
  and no integer/string/`None` coercion is introduced;
- retained and overflow keys differ inside each crosstalk config;
- deterministic, delay, recovery, and composite configs own no key;
- stochastic-capable Charge/noise producers and effects accept
  `CounterRng`, while deterministic producers and preparation helpers omit it;
- exact-zero and disabled paths make no RNG request;
- Charge's scientific Poisson means, current/later-category masses, fixed
  category/address order, final no-draw remainder, complete multinomial
  orchestration, checked count accumulation, and bookkeeping remain
  TensorDSLab-owned; `charge/effects/_counts.py` contains those local concerns
  but no Poisson, inversion, PTRS, binomial, or BTRS implementation;
- the accepted `config.py`, `field.py`, `collection.py`, and private
  `charge/effects` ownership split is complete without compatibility shims;
- TensorCore `require_same_dtype(...)` is used only for Analog inputs and the
  present floating `ReadoutCollection` subset, while raw tensors retain
  operation-specific checks;
- one private `_require_representable_float(...)` helper replaces duplicated
  scalar-to-dtype conversions without absorbing product-specific range,
  ordering, or envelope policy;
- `_RngStream`, `readout/_random.py`, and any replacement `readout/_rng.py`
  are absent; and
- TensorCore independently proves the public distribution mappings, while
  TensorDSLab migration probes prove `uniform`, `gaussian`, `poisson`, and
  `binomial` address/output continuity and completed `NoiseWaveform` and
  `Charge` default-key continuity on the exact recorded numerical-stack
  evidence boundary.

## Closed Maintenance 3 Environment-Qualified Continuity

The focused
[Maintenance 3 work order](implementation/maintenance_3_environment_qualified_stochastic_continuity.md)
is Merged / Closed through exact Review-cleared candidate
`dfe45c96f9cc141f91e29a6a3d81bd7a3e8a49f0` and its five-document Design
closeout. It corrects a test-applicability defect exposed before any accepted
Stage 8 measurement: Maintenance 2's completed stochastic hexadecimal
literals are exact evidence for their recorded macOS 15.7.4 arm64, Python
3.13.11, PyTorch 2.12.1 eager-CPU stack, not a portable CPU bitwise contract.

The closed contract has three distinct validation obligations:

1. Threefry raw words and fixed-point uniforms retain their separately
   documented exact scope. TensorDSLab exercises only public TensorCore
   surfaces and keeps its public fixed-point-uniform literals exact on every
   accepted test stack.
2. On the exact recorded Maintenance 2 stack, the unchanged Gaussian,
   Poisson, binomial, completed-noise, and completed-Charge literals must still
   execute and match exactly.
3. On another accepted stack, those completed requests execute independently
   twice and match exactly within that unchanged stack, while all existing
   structural, invariant, analytic, and statistical checks remain active.

For completed transcendental values, an unchanged stack includes the
OS/architecture, Python/PyTorch build, backend/device implementation, eager
execution mode, dtype, and relevant math settings. Validation must not turn an
observed platform difference into a ULP tolerance or alternate golden table,
nor skip or xfail the fixture. Maintenance 3 changes no production,
dependency, RNG, or scientific byte.

Fixed-commit Validation and independent Review cleared the exact candidate
against freshly reconstructed TensorCore `0.9.0` source and canonical archive
forms. On the recorded macOS stack, each role passed the focused source and
archive module at `9/9/0`, the complete source and archive suites at
`188 run / 176 passed / 12 conditional CUDA skips`, and Pyright with zero
diagnostics. In separate full-A100 Della allocations, each role passed four
fresh-process focused runs at `9/9/0` and the complete source and archive
suites at `188/188/0`; the required replay/literal-branch mutants were
rejected and the frozen environment inventories remained unchanged.

Review's successful Della harness bound the exact frozen CUDA-library package
inventories but did not separately print or assert three redundant descriptive
labels for RHEL, CUDA compatibility, and cuFFT. A duplicate job queued only to
add those literal prints was canceled before allocation and contributes no
evidence. This qualification does not weaken the executed runtime, device,
inventory, source/archive, mutation, or result gates. Maintenance 3 evidence
does not count as Stage 8 evidence or authorize a Stage 8 restart.

## Closed Stage 7 Public-Orchestration Evidence

The focused
[Stage 7 work order](implementation/stage_7_public_readout_orchestration.md)
is Merged / Closed through exact Review-cleared implementation candidate
`6dd55024685013fb9412a7247d3ddde7be1a3177`. It required the public
`rng: CounterRng`, no simultaneous `seed=`, and deterministic closures that
request no RNG values. TensorCore exposes no non-consuming concrete-algorithm
capability query, so Stage 7 does not issue a dummy draw, inspect protected
hooks, or restrict the public boundary to `Threefry4x32`. A real custom
RNG/backend incompatibility remains a dynamic failure at the first genuine
distribution call.

The Stage 7 implementation and fixed-commit evidence prove:

- all 63 nonempty recognized product subsets and one-pass request-iterable
  consumption;
- rejection of empty, duplicate, foreign, base, instance, and non-class product
  requests without adversarial final-leaf subclass policing;
- exact transitive configuration closure and no influence from unrequested
  branches;
- completion of every product-owned preparation plan before the first RNG
  request, producer invocation, or semantic-product/output write;
- closure-wide rejection of one `RngKey` assigned to distinct stochastic
  roles, including structurally present numeric no-op configs and excluding
  absent configs, `ZeroNoiseConfig`, and unrequested branches;
- each required producer executing at most once, exact requested-only retention,
  request-order independence, and common-product equality when retention grows;
- exact source return when requested, source/config immutability, generated-
  product freshness, pairwise output storage independence, and exact source-axis
  reuse;
- one exact product-specific deep postcondition per generated field before
  downstream use or retention, including finite Pure/Noise/Analog values and
  config-bounded Digitized values, plus no escaped field or downstream
  invocation on failure;
- CPU behavior, plus the presence and explicit skipped qualification of
  conditional CUDA checks; CUDA deep-value reductions and scalar extraction
  remain documented as possible current-stream synchronization; and
- no partial collection or semantic product escaping a failed call, without a
  rollback promise for private allocations or completed local prerequisites.

The Stage 7 work order is the exact scope and evidence authority. Validation
and independent Review cleared the same final candidate, and Review
fast-forwarded and reverified unchanged `main`. The source and independently
recreated archive suites each ran 188 tests: 176 passed and 12 conditional
CUDA tests skipped. Pyright `1.1.411` reported zero diagnostics in both forms.
CUDA was unavailable, so this is eager CPU-only evidence and makes no GPU,
cross-backend, fusion, allocation, or performance claim.

## Closed Maintenance 4 Runtime Action Ownership Evidence

The focused
[Maintenance 4 work order](implementation/maintenance_4_runtime_action_ownership.md)
is **Merged / Closed** through exact Review-cleared supplemental candidate
`b3c7c907004741ba67b8b92a54bbdc8c85216dda`, tree
`2d35a0e926b912f3fa846da97726e4e2490c4cc3`. Review cleanly fast-forwarded
`main` from `5fdd3fafe2c44357b09df2a04b88cb121f2d3638` to those exact bytes
without a merge commit. Final Design accepted the merged result and records
closure in this documentation-only direct child.

Maintenance 4 implements the behavior-preserving internal ownership refactor
from product `_produce.py` bundles and `*Plan` records to unexported product
`runtime/{prepare,produce,validate}.py` actions and concrete `*Runtime`
records. It changes no public facade, scientific equation, stochastic address,
result law, dependency, or supported device boundary.

Fixed-commit Validation and independent Review preserved all Stage 7 public,
scientific, RNG, storage, and autograd evidence while proving:

- public facade exports and `simulate_readout(...)` object identity remain
  unchanged; Runtime records/actions remain absent from every public
  `__all__`;
- runtime and effect `__init__.py` modules import and export nothing, while
  direct deep imports are treated as unsupported Python implementation access
  rather than prevented through `hasattr` policing;
- every generated product owns exactly one `prepare_<product>`,
  `produce_<product>`, and `validate_<product>` action in its exact defining
  runtime module; Photoelectrons owns only `validate_photoelectrons`;
- every Runtime is concrete, final, frozen, slotted, non-inheriting,
  unexported, and contains no Config, semantic product, collection, stored
  callable, mutable cache, hidden movement, or generic framework state;
- `prepare_sampling(...)` discovers the public simulation source sample
  dimension exactly once, and temporal ProductRuntime values retain the exact
  same `SamplingRuntime` object where sampling is required;
- the complete requested Runtime closure is prepared before the first RNG
  request, production action, or semantic-output write;
- production modules import no Config or validator, repeat no scientific
  preparation, and perform no product deep publication scan;
- `simulate_readout(...)` executes exact
  `produce -> validate -> descendant` order, so each forced validation failure
  prevents every later production/RNG action and final collection
  construction;
- every generated validator receives the exact produced field and its named
  direct prerequisite products and proves value, axes, shape, dtype, device,
  and fresh-storage publication relationships;
- `validate_charge` performs one terminal deep scan and preserves the accepted
  invalid-generated-result `RuntimeError`; digitized validation receives the
  prepared exact maximum code rather than a Config;
- the convolution-ready PureWaveform kernel is flipped/shaped once during
  preparation without changing same-stack values or gradients; and
- former `_requirements.py`, product `_produce.py`, and `charge/effects/`
  paths are absent without aliases or shims, while closed work orders remain
  unchanged historical evidence.

The final evidence chain begins at Design authority
`36b9cd8338a0a5dad7e6c005d25e70c067a7e66d`, includes Review-returned
Candidate 3 `cab01ca434d5c89f550e960fff8c1684fa7c2a8f`, and ends at the exact
supplemental candidate above with zero remaining Review findings. TensorCore
remains pinned to exact `0.9.0` commit
`4708bf2ca063a1bcd37a30a342733b9e3dbe9f59`. The local and full-A100
totals are recorded below; none changes the public, scientific, RNG,
dependency, performance, or Stage 8 boundary.

Validation and Review also audited duplicate logic. They accepted only narrow
extractions whose exact type, axis, shape, dtype, device, sampling,
representability, finite/range, freshness, numerical-order, RNG, and autograd
semantics are genuinely identical. They rejected a Runtime/Action ABC,
registry, reflection layer, generic dependency graph, decorator framework,
untyped action mapping, or broad `utils.py` / `helpers.py` dumping ground.

Maintenance 4 deliberately adds no `PureWaveformRenderer`, `nn.Module`, public
atomic product action, TensorML dependency, IO/artifact surface, workspace,
optimization, benchmark, or Stage 8 evidence. Renderer validation remains
deferred to its own focused work order.

Documentation-only Design work remains in Design unless the user requests an
independent documentation Validation or Review. At minimum, run:

```bash
git diff --check
```

Also check local Markdown links, heading and code-fence balance, stale `0.6`
names in live documents, accidental placeholder files, and consistency among
architecture, design, decisions, parity, validation, and implementation-stage
records.

## Closed Maintenance 5 Compact-Axis And Sampling Evidence

The closed
[Maintenance 5 work order](implementation/maintenance_5_tensorcore_0_13_compact_axes_and_sampling.md)
adopts exact published TensorCore `0.13.0` commit
`202d8b1bc6259b8453d3d377570417f2480d782b`. The explicitly labeled
Historical Stage 3 sections below preserve closed string-axis and
`SamplingConfig` evidence; they do not define the Maintenance 5 target.

Maintenance 5 Validation and Review independently proved:

- exact dependency commit, parent, tree, version, 30 root exports, 14 installed
  files, direct Git pin, source/archive byte identity, and canonical archive;
- direct final fieldless `ExampleAxis(CountAxis)`,
  `ChannelAxis(LabelAxis)`, and `SampleAxis(RegularAxis)` leaves;
- exact inherited keyword-only constructors and TensorCore-first generic
  validation;
- nonempty example/channel axes, zero-based example ordinals, exact channel
  labels, and range-backed nonmaterializing count/sample coordinates;
- nonnegative sample start, positive step, count at least two, exact
  signed-int64 exclusive-stop boundary, integer `*_ps` properties, lookup,
  equality, and hash behavior;
- arbitrary field axis order and exact generated-product source-axis reuse;
- complete-input `SampleAxis.start == 0` before RNG, production, or final
  collection effects, while a nonzero-start semantic subgrid remains
  constructible;
- `prepare_sampling(photoelectrons)` deriving the existing three-field
  `SamplingRuntime` exactly once from the source axis;
- complete removal of `SamplingConfig`, `common.sampling`,
  `ReadoutConfig.sampling`, old constructors, exports, aliases, and agreement
  checks;
- `ReadoutConfig()` as the valid truth-only config and unchanged generated
  config closure;
- unchanged `simulate_readout(...)`, scientific preparation, RNG
  key/position/ordinal/call behavior, same-stack values, storage, axes,
  dtype/device, validation, and autograd contracts;
- supported exact-type collection lookup and missing-key `KeyError`; and
- deletion without replacement of the off-path
  `collection.field(TensorField)` exception assertion.

Strict Pyright probes cover int coordinates for Example/Sample, string
coordinates for Channel, concrete typed axis/field lookup, new constructors,
retired imports/keywords, and source-derived preparation. Runtime tests own
exact `bool` rejection because Python typing treats `bool` as `int`.

Implementation ran full local source/archive and static gates. Validation and
Review each used a separate fresh full-A100 allocation across supported
PyTorch `2.11` and `2.12`, with no conditional CUDA skips. This evidence is
functional dependency/axis correctness, not Stage 8 performance evidence.

## Maintenance 6 Pint And Runtime-Boundary Gate

The
[Maintenance 6 work order](implementation/maintenance_6_pint_physical_configuration_boundary.md)
is **Merged / Closed** through exact Review-cleared target
`0257fb477ee04556ebbe26351123ae610b5d7925`, tree
`b4f5703ca5b756dc27d876c1dd17ee56cb43b4e8`, against exact clean Maintenance 5
Design closeout `021694b9479d02546405f6a815aedf21c9c831a4`. Candidate 1 is exact commit
`240e1492c466097b3059dfe9911ab338a4dd38a1`, tree
`1e5cae8c0e905c9638eb40e7f9d24fac950fee59`. Validation and independent
Review proved:

- exact Pint `0.25.3` wheel and source identities, dependency resolution,
  isolated source/artifact imports, and retained exact TensorCore `0.13.0`
  commit;
- one private registry, no application-registry mutation, and exactly the two
  new `quantity(...)`/`quantities(...)` facade exports;
- exact scalar magnitude/container acceptance, parser and dimensionality error
  translation, external-registry canonical copying, absence of Pint arrays,
  and one TensorCore `Scalar.require(...)` normalization per physical field;
- all 26 physical Config fields, all 35 unchanged nonphysical fields, exact
  local/cross-field rules, and explicit unhashability of all 22 Configs;
- static typing for wrapper, key, nested-Config, optional, and union
  composition; absence of `require_exact`, `require_optional_exact`, and
  `require_one_of_exact`; and no replacement runtime-diagnostic promise for
  malformed typed composition;
- unit-neutral public physical names and complete absence of compatibility
  aliases for retired suffix-bearing public fields;
- compact integer SampleAxis state, the bounded one-ULP period conversion, and
  exact built-in-int physical accessor magnitudes—including legal values above
  `2**53`—without storing a Quantity or unit on the axis;
- direct integer source-derived `SamplingRuntime` preparation;
- one-time extraction of every active canonical magnitude into plain
  unit-suffixed Runtime facts, with no Config or Pint object recursively
  retained by a Runtime;
- pure numerical helpers receiving plain operands rather than Configs and no
  producer or validator performing unit conversion or Config interpretation;
- removal of only duplicate child-preparer and private Charge
  Runtime/primitive admission policing plus annotation-only Config membership
  checks already owned by the public typed path;
- retention of exact model dispatch, scientific laws, tensor relationships,
  representability, count/envelope/address/allocation limits, axes identity,
  storage freshness, absolute dtype/device rules, and generated-product
  validation;
- exact canonical-operand continuity for deterministic/stochastic products,
  explicit physical-equivalence-versus-binary-equality cases, unchanged RNG
  calls, and exact-zero no-draw behavior; and
- complete local source/artifact, Pyright, mutation, import/privacy, scope,
  protected-byte, and hygiene evidence across the exact TensorCore
  source/archive by Pint wheel/sdist forms.

TensorCore helpers must be used only when their semantics match the package
requirement. `require_axis_signature(...)` cannot replace the unordered
readout-axis set, `require_same_axes(...)` cannot replace exact source-axis
tuple identity, and no TensorCore helper establishes TensorDSLab storage
freshness or scientific value domains.

This gate is functional/API correctness evidence, not Stage 8 performance,
fusion, allocation, persistence, integration, or compatibility evidence.
Under the user-selected evidence schedule, neither role submitted fresh
cluster work for Maintenance 6. The complete local matrix passed with `13`
conditional CUDA skips as an explicit qualification, and no
accelerator-correctness claim follows. Local `main` remains unpushed. Fresh
two-minor CUDA matrices wait until a separate TensorDSLab work order adopts
published
TensorCore `0.15.0` exact commit
`0f974e9e7f52125bbe829e124beb24e69de811d3`; those later package-owned gates
qualify only that exact integrated pairing.

## Maintenance 7 TensorCore 0.15 Adoption Gate

The
[Maintenance 7 work order](implementation/maintenance_7_tensorcore_0_15_adoption.md)
has exact immutable Candidate 1
`68c2f62c2ce354dd6c92fde28b020c0ce71881d6`, which was
**Validation-cleared / Review-returned for a Design-owned package-source
correction**. Its documentation-only correction lineage preserves Candidate
1's production and test bytes. A pre-merge correction target requires
fixed-commit Validation and Review; after an unchanged Review fast-forward, the
same bytes await Design closeout.

For every pre-merge correction target, local fixed-commit Validation and
independent Review must retain Candidate 1's complete gate and additionally
prove:

- exact published TensorCore `0.15.0` commit/tree/version, 34 package-root
  exports, 19 `tensor_core.validation` exports, the exact one-name public
  `tensor_core.validation.random` surface, 25 package files, and exact
  source/archive identity;
- exact Pint `0.25.3` and unchanged TensorDSLab 35/30/5 exports;
- complete retirement of `logical_positions` and local generic
  dtype/floating/representability/allocation/count-domain implementations
  without a wrapper or alias;
- exact `RngPositions` raw value/order agreement for every base, move, select,
  PSD slice, generation offset, timing-jitter offset, and AP-category offset;
- exact same-stack raw words and completed product continuity, no-op no-draw
  behavior, and unchanged global RNG isolation;
- direct delegation to TensorCore's field dtype/layout, eager-float,
  shape-span, allocation, and count-tensor requirements while retaining every
  TensorDSLab readout/scientific/product relationship;
- one non-exported production source for namespace `0x54445331`, with all ten
  streams and every fixed key unchanged, no public Config key fields, and no
  request-time caller-key collision admission;
- synchronized live package sources naming
  `readout/runtime/requirements.py` rather than the retired root module;
- positive pulse-amplitude Config magnitudes, rejection of zero/negative
  quantities, one preparation-owned fixed negative sign, retained
  dtype-rounded-zero rejection, and exact calibrated negative-going kernel and
  waveform continuity;
- unchanged source/axes/storage/autograd/device/dtype/product postconditions;
- zero-diagnostic Pyright plus frozen negative typing for raw-tensor position
  rejection and the adopted validation signatures; and
- full local source/archive, Pint wheel/sdist, artifact, import-isolation,
  privacy, mutation, protected-byte, and hygiene evidence.

Maintenance 7 performs no cluster submission and makes no new accelerator
claim. After its exact local closeout, a separate integrated-CUDA authority
must run the complete TensorCore and TensorDSLab two-Torch-minor matrices
against that exact pairing before TensorDSLab is pushed. Those runs are
functional correctness evidence, not Stage 8 performance evidence.

## Governance Adoption Checks

TensorDSLab adopts Governance Core `0.1.0` through `TDSLAB-GOV-D001`.
Conformance remains `Not evaluated`, Coordination remains `Deferred`, and
Profile B remains `Disabled`. Validate the
[governance index](governance/index.md),
[adoption record](governance/adoption_0_1_0.md),
[overlay](governance/overlay.md),
[semantic rule map](governance/rule_map_0_1_0.md), and `docs/decisions.md`
against these checks:

- verify Governance Core manifest-file SHA-256
  `45292e1d72ab79bb4df68a13b82a4ece1bd1207901cd278cc111fe376da28be8`
  and all eight entries;
- verify Council Charter manifest-file SHA-256
  `343ab10b0ccf54e95fadd70e8cb49ada4480b27149380d39216b2ef1fe9c6916`
  and all three entries;
- verify governed Design base
  `151b61fdc36475498219ee5fe7b045a3a72c2d09`, exact accepted candidate
  `d634401a853915edeb4f83df4a4943b3553deced`, its exact parent, and its
  authorized nine-path scope;
- map `OP-01` through `OP-13` and `ENG-01` through `ENG-12` exactly once and
  retain six `Adopted`, 19 `Stronger local rule`, no whole-rule
  Not-applicable disposition, and no accepted deviation;
- retain exact absence evidence and a focused activation trigger for every
  dormant surface;
- confirm durable files contain no raw task identifier and no `.agents`,
  route, registry, or cache state was created;
- reject claims of conformance, Active Coordination, enabled Profile B,
  deployability, release readiness, backward compatibility, broad
  compatibility, or implemented integration; and
- treat compatibility evidence as exact-baseline evidence only. Same-device
  residency and no-silent-host-materialization Design constraints do not prove
  an implemented package handoff.

Changing the TensorCore dependency and package structure under an authorized
focused package-local work order, including historical Stage 3 or Maintenance
5, does not alter the adopted governance record or create a conformance
finding.

## Boundary-First Validation

The rebuild follows this order:

```text
public constructor/config values
  -> TensorCore constrained scalars and semantic roots
  -> TensorDSLab axes and product fields
  -> complete private request and ProductRuntime preparation
  -> product production
  -> immediate product runtime validation
  -> ReadoutCollection completed result
  -> public simulate_readout(...)
  -> future TensorG4DS, TensorML, and durable boundaries
```

TensorCore validates universal representation invariants before calling a
leaf's `_require()`. TensorDSLab leaf construction validates cheap intrinsic
semantics. Full-device finite/nonnegative/bounded scans belong to explicit
product-owned runtime validators at untrusted ingress and generated-product
publication boundaries; they do not live in `field.py`, run invisibly in every
constructor, or remain embedded in production actions.

## Historical Stage 3 Package And Dependency Checks

Stage 3 must prove:

- the project remains `TensorDSLab` and the import package remains
  `tensor_dslab`;
- `tensor_dslab.common` and `tensor_dslab.readout` remain directly under the
  import root, with product packages directly under `readout`;
- `pyproject.toml` selects exact TensorCore `0.7.0` commit
  `b454d738f6385ce6489d85492a618a3dab139bb6`;
- all TensorCore imports come from the public `tensor_core` package root;
- no retired TensorCore module, compatibility alias, copied helper, local
  fork, or generic TensorCore re-export exists;
- no production import reaches TensorG4DS, TensorML, DSLab, IV-DSLab,
  Projects/dag, G4DS/g4ds11, NumPy, or an IO backend;
- package imports do not transitively load those deferred dependencies;
- every created module owns real Stage 3 behavior; and
- `simulation.py`, `_random.py`, and every `_product.py` remain absent because
  Stage 3 implements no operation or orchestration behavior.

The exact TensorCore public surface used by this stage is limited to
`TensorAxis`, `TensorField`, `TensorCollection`, the accepted constrained
scalars, and public relationship requirements. Retired `Id`, axis/field IDs,
`IdSequence`, `TensorLayout`, selection, movement, reconstruction, output-
buffer, and like-allocation surfaces must be absent from live production and
tests.

## Historical Stage 3 Ordinary-ABC Semantic Leaf Checks

The three axes, six product fields, and `ReadoutCollection` each have
`__bases__ == (matching_tensor_core_root,)`, directly inheriting exactly one
root with no mixin or other base. Static, runtime, and Review evidence must
prove every leaf:

- is decorated with `@final`;
- declares `__slots__ = ()`;
- adds no stored annotation or dataclass field;
- does not reapply `@dataclass`;
- does not override TensorCore construction, `_validate`, equality, hashing,
  or lookup behavior;
- implements `_require(self) -> None`; and
- inherits the exact root constructor signature.

Constructor probes must establish:

```text
ExampleAxis(coordinates: tuple[str, ...])
Photoelectrons(tensor: torch.Tensor, axes: tuple[TensorAxis, ...])
ReadoutCollection(*, fields=...)
```

Use the actual inherited signatures exposed by TensorCore and require the
selected static checker to infer concrete results from each constructor and
from typed `field(...)`, `tensor(...)`, `axis(...)`, and `dimension_of(...)`
calls. Tests must not require a runtime-finality guard or adversarially probe
subclassing of final classes, class mutation, constructor bypass, direct
private calls, or custom Torch dispatch. Those uses are unsupported.

## Historical Stage 3 Axis And Sampling Checks

### `ExampleAxis` And `ChannelAxis`

Tests should prove exact-string coordinate validation is inherited from
TensorCore and each TensorDSLab axis additionally rejects an empty coordinate
tuple. Coordinate tuple order is tensor index order. Coordinate labels are not
RNG hot-path identities.

### `SampleAxis`

Tests should prove:

- at least two coordinates are required;
- timestamps use exact ASCII grammar `^(0|[1-9][0-9]*)ps$`;
- signs, whitespace, leading zeros, decimals, exponents, uppercase or
  alternate units, and values above signed-int64 are rejected;
- timestamps increase strictly at one positive uniform integer-picosecond
  period;
- the derived exclusive stop is at most `2**63 - 1`;
- `start_ps`, `sample_period_ps`, and `stop_ps` are correct;
- direct construction of a regular nonzero-start subaxis remains valid; and
- the coordinate tuple contains left edges only and never the terminal right
  edge.

### `SamplingConfig`

Tests should prove:

- `sample_period_ps` and `sample_count` require exact `PositiveInteger`
  wrappers;
- the count is at least two;
- `sample_period_ps * sample_count <= 2**63 - 1`;
- `window_stop_ps` is exact;
- `build_axis()` produces exactly `sample_count` canonical zero-start left-edge
  timestamps; and
- the returned `SampleAxis` agrees with the configured count, period, and
  exclusive stop.

Stage 3 does not implement PE binning. Boundary fixtures for future binning
must continue to treat bins as left-closed/right-open, include `0` and every
`i * period`, exclude negative time and `window_stop_ps`, and account for
underflow and overflow separately when that bridge is implemented.

## Shared Private Requirement Checks

`tensor_dslab.readout.runtime.requirements` is the unexported owner of the sole
shared readout-domain relationship under Maintenance 7. The former
`tensor_dslab.readout.requirements` and
`tensor_dslab.readout._requirements` paths are removed without shims. Generic
dtype, layout, representability, allocation, shape-span, and count-domain
requirements come directly from exact TensorCore `0.15.0` where their
contracts match. For supported value and container inputs, focused tests prove
the retained `TypeError` distinction for malformed values and `ValueError`
distinction for well-typed values that violate a relationship. Malformed or
off-path typed class-object arguments remain outside that promise. Focused
tests should prove every retained or adopted helper preserves the applicable
supported-input distinction.

The shared readout-structure behavior still requires exactly one
`ExampleAxis`, `ChannelAxis`, and `SampleAxis` in any order and
`torch.strided` storage. It does not require contiguity, a fixed dimension
order, or an exact base `torch.Tensor` type. TensorCore's variadic
`require_field_dtype(...)` supplies the exact one-or-more accepted-dtype
mechanic; TensorDSLab does not wrap it.

Maintenance 5 removes public `common.sampling` and `SamplingConfig` without a
shim. Private
`readout.runtime.sampling.prepare_sampling(photoelectrons)` derives the source
`SampleAxis` relationship once and returns the existing frozen slotted
`SamplingRuntime` containing Python integer count, period, and dimension.
Tests must prove one sample-dimension discovery per public request and exact
`SamplingRuntime` object identity across temporal ProductRuntime values that
retain it. No product preparation may accept or reconstruct a competing
sampling policy.

`readout/runtime/requirements.py` may contain only downstream behavior shared
with exactly identical semantics and not already owned by TensorCore.
Product-specific equations and named validators remain in their owning runtime
packages rather than becoming a registry, global validation layer, scientific
dumping ground, or generic product framework.

## Product Field Checks

Stage 3 defines exactly these direct final leaves:

| Exact type | Constructor invariant | Explicit deep-value invariant |
| --- | --- | --- |
| `Photoelectrons` | `torch.int64`, exact readout axes, `torch.strided` | nonnegative |
| `Charge` | `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite and nonnegative |
| `PureWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `NoiseWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `AnalogWaveform` | exactly `torch.float32` or `torch.float64`, exact readout axes, `torch.strided` | finite |
| `DigitizedWaveform` | `torch.int32`, exact readout axes, `torch.strided` | nonnegative and at most the prepared exact maximum code |

Each `field.py` owns only its exact semantic leaf and cheap intrinsic
`_require()` hook. Explicit full-value checks belong to product
`runtime/validate.py`: `validate_photoelectrons`, `validate_charge`,
`validate_pure_waveform`, `validate_noise_waveform`,
`validate_analog_waveform`, and `validate_digitized_waveform`. The digitized
validator receives the prepared exact maximum code or its ProductRuntime, not
`DigitizedWaveformConfig`. Tests must separate cheap constructor validation
from these explicit runtime-action checks and prove field modules import no
runtime validator.

Photoelectrons validation owns nonnegativity only. Tests must preserve the
public distinction that a truth-only request can contain a nonnegative count
above Charge's per-cell `2**53 - 1` ceiling, while a closure requiring Charge
rejects that source during Charge preparation before RNG or production.

Test at least two valid axis orders, exact shape/axis agreement, missing,
duplicate, or foreign axes, sparse/non-strided rejection, noncontiguous
strided acceptance, correct and incorrect dtypes, CPU tensors, and conditional
CUDA tensors when available. General semantic construction remains
placement-neutral and makes no GPU-kernel claim.

`Photoelectrons` is an already-produced dense truth input. Stage 3 creates no
`PhotoelectronsConfig`, source producer, PE-binning function, or TensorG4DS
adapter. `DigitizedWaveform`, not `DigitalWaveform`, is the accepted product
name; truncation is fixed by the Stage 4 producer and no quantization enum or
sidecar exists.

## Configuration Checks

Every config is a public `@final`, frozen, slotted, keyword-only dataclass.
There is no generic `Config` ABC. Tests should prove exact component wrapper
types, closed exact-class unions, immutable composition, and every local range
or relationship in [Scientific Configuration](architecture/rebuild.md#scientific-configuration).

At minimum, cover:

- `TimingJitterConfig`, `DarkCountConfig`, `FixedDelayConfig`,
  `ExponentialDelayConfig`, `DirectCrosstalkConfig`,
  `DelayedCrosstalkConfig`, `AfterpulseRecoveryConfig`, `AfterpulseConfig`,
  `CorrelatedAvalancheConfig`, `ChargeSmearingConfig`, and `ChargeConfig`;
- after completed Stage 6, exact two-member fixed/exponential crosstalk unions
  and complete `NormalDelayConfig` absence from production, package exports,
  and current package-contract expectations, without a shim;
- `TpcFebSnrPulseConfig`, `VetoPduPulseConfig`, and the exact two-model
  `PureWaveformConfig` union;
- `ZeroNoiseConfig`, `WhiteNoiseConfig`, `PsdNoiseConfig`, and the exact
  three-model `NoiseWaveformConfig` union;
- `AnalogSaturationConfig` and `AnalogWaveformConfig`;
- `DigitizedWaveformConfig`, including bit depth 1 through 16, strict input
  voltage ordering, and gain from 0 through 40 dB; and
- `ReadoutConfig`, containing only optional exact product-config components;
  `ReadoutConfig()` is the valid truth-only configuration.

PSD tests at this structural stage cover tuple type, nonempty equal-length
left-edge/density arrays, zero start, strict edge order, exclusive stop, finite
nonnegative density, and rejection of an all-zero supplied PSD in favor of
`ZeroNoiseConfig`. They do not implement or validate FFT synthesis.

Base classes, foreign objects, and wrong scalar wrappers are unsupported config
values and must fail at the documented public constructor. Subclassing a final
config is itself unsupported and needs no separate adversarial probe. This is
ordinary public-input validation, not a promise to police callers who mutate
classes or bypass construction.

## `ReadoutCollection` Checks

Tests should prove:

- any nonempty subset of the six exact product types is accepted;
- empty or unrecognized membership is rejected;
- duplicate exact product types are rejected by TensorCore;
- membership order has no semantic meaning;
- `field_types` is the exact frozenset of present classes;
- `field(Product)` and `tensor(Product)` infer and return the exact product or
  tensor, while a missing product raises `KeyError`;
- every present field has equal ordered axes and the same exact device;
- all present floating products have one common dtype;
- mixed integer and floating role dtypes remain valid where each leaf permits
  them;
- the collection retains the exact supplied field records and tensor
  references;
- collection membership is immutable and collection equality remains object
  identity; and
- there is one `ReadoutCollection`, no per-product collection subclass,
  canonical-order registry, descendant map, sidecar, lifecycle state, or
  mutation API.

`ReadoutCollection.accepted_field_types()` is the sole class-owned schema
declaration and returns one unordered frozenset containing all six exact
classes. A completed collection may contain only a requested product subset;
it is not an ordered partial-pipeline snapshot.

## Public Surface And Import Checks

Verify deliberate `__all__` values and object identity across:

- each product package root;
- `tensor_dslab.common`;
- `tensor_dslab.readout`; and
- the top-level `tensor_dslab` collaborator API.

After Maintenance 5, the top-level package exposes the three compact semantic
axes, six products, all public product configs, `ReadoutConfig`,
`ReadoutCollection`, and the implemented Stage 7 `simulate_readout`. It does
not export `SamplingConfig` or re-export TensorCore generic classes or
scalars, requirements, Runtime records, preparation, production, validation,
effect actions, retired `0.6` names, or another simulation entry point. The
readout root and package root export the same `simulate_readout` object exactly
once. Runtime/effect `__init__.py` modules import and export nothing; privacy
tests inspect facades and `__all__` rather than assuming Python prevents deep
imports or parent-package attributes.

Maintenance 6 adds only `quantity` and `quantities` to
`tensor_dslab.common` and the package root. The readout facade and every
product facade remain unchanged; the private registry, unit helpers, Runtime
records, action functions, and TensorCore generic `Scalar` remain unexported.

Closed Stage 6 regression checks proved every former product package,
`readout.types`, the readout root, and the package root were acyclic. Closed
Maintenance 2 evidence instead proves fresh-process imports of every product
`config`/`field` module, `readout.config`, `readout.collection`, the readout
root, and the package root, plus absence of the retired `types.py` modules.
Product packages must not import `ReadoutConfig`, `ReadoutCollection`, or
the Stage 7 orchestration layer. The complete product graph may be imported
only by the cross-product composition layer and deliberate export layers.
Under the merged Maintenance 4 implementation, product runtime modules also
must not import `ReadoutRuntime`, simulation, TensorML, TensorG4DS, DAG, or a
private TensorCore module. Internal callers import exact defining runtime
modules and never a runtime-package facade.

## Static Typing Checks

The selected checker must analyze package and tests against the exact
TensorCore pin. Positive probes should require concrete inference for:

- each inherited axis and field constructor;
- `ExampleAxis(count=...) -> ExampleAxis`;
- `ChannelAxis(labels=...) -> ChannelAxis`;
- `SampleAxis(start=..., step=..., count=...) -> SampleAxis`;
- `prepare_sampling(photoelectrons) -> SamplingRuntime`;
- `ReadoutCollection(fields=...) -> ReadoutCollection`;
- `readout.field(Charge) -> Charge`;
- `readout.tensor(Charge) -> torch.Tensor`;
- `charge.axis(SampleAxis) -> SampleAxis`; and
- `charge.dimension_of(SampleAxis) -> int`.

Maintenance 6 adds positive probes for unparameterized Pint `Quantity`
signatures on every migrated physical field, `quantity(...)`,
`quantities(...)`, `SampleAxis.from_period(...)`, and the four physical-time
accessors. Package-private probes require
`type[Scalar[float]].require(...) -> float`, exact plain Runtime field types,
and no Any leakage across Config-to-Runtime preparation. Negative probes cover
raw physical numbers, wrong dimensions, retired suffix-bearing public names,
Quantity use in dimensionless fields, and attempts to treat Runtime facts as
quantities.

The work order must report the exact checker/version or explicitly qualify its
absence. Manual review is not a substitute for the fixed stage-specific
static probes.

## Result Taxonomy, Storage, And Device Scope

Stage 3 introduced no field-returning operation, so TensorCore's operation-
owned exact-return/storage-sharing/guaranteed-fresh taxonomy has no production
operation to classify yet. Construction retains the exact caller tensor by
TensorCore contract; it does not claim a copied or fresh payload. Axis lookup
returns the exact stored axis. Under Maintenance 5 the source bridge constructs
the compact `SampleAxis` directly; private sampling preparation derives
execution facts and returns no field result.

Every later field-returning operation must classify each successful path and
separately document subtype, dtype, device, axes, autograd, synchronization,
failure effects, and output-to-output sharing. No future operation may enqueue
writes after publishing its semantic field. These later requirements do not
authorize Stage 3 to create `out=`, output-buffer, workspace, lease, movement,
selection, or lifecycle APIs.

Stage 3 requires CPU construction tests and conditional CUDA construction
tests. CUDA absence is a recorded skip and no GPU behavior claim. Tests must
prove device mismatches within one collection fail rather than move silently.
No code path may silently call `.cpu()`, `.numpy()`, `.tolist()`, detach, cast,
or import NumPy as a reference implementation.

## Staged Scientific Validation

Stage 3 did not implement scientific producers, RNG, request planning, or
`simulate_readout(...)`. Stage 4 subsequently validated and merged the TPC/Veto
pure, analog, and digitized producers, including their reference equations,
freshness, axes/device/dtype behavior, and accepted autograd contracts. It did
not validate Charge, public orchestration, or GPU fusion. Stage 5 subsequently
validated and merged the private positional Threefry reference, its
noise-consumed fixed-point uniform and Box-Muller behavior, and complete
exact-zero, IID-white, and caller-supplied PSD noise. CUDA was unavailable, so
that closeout is eager CPU-only evidence and makes no GPU execution or
performance claim.

Stage 6 subsequently validated and merged the aggregate multinomial and hybrid
Poisson samplers, dark counts, analytic timing jitter, fixed-generation
correlated avalanches, S1/S2 charge ledgers, recovery weighting, right
overflow, smearing, operation-owned freshness, and stream ordering. The
evidence is eager CPU-only; conditional CUDA checks were skipped and no GPU
fusion or performance claim was made.

Stage 7 subsequently validated and merged product-request closure, complete
preparation, execute-once orchestration, requested-only retention, and the
public result boundary. The remaining acceptance matrix in
[Rebuild Validation Strategy](architecture/rebuild.md#validation-strategy)
therefore covers later CUDA and measured optimization evidence, and future
TensorG4DS, TensorML, Reconstruction, and durable boundaries.

Maintenance 4 subsequently validated and merged the Config-to-Runtime
preparation, execution-only production, and immediate product-runtime
validation ownership split without changing scientific behavior. Against each
exact TensorCore source/archive form, the final candidate passed the local
focused suite at `97 run / 88 passed / 9 conditional CUDA skips`, full
discovery at `198 / 185 / 13`, and Pyright with zero diagnostics. In separate
fresh full-A100 allocations, fixed-commit Validation and independent Review
each passed focused source and archive suites at `97/97/0` and full source and
archive suites at `198/198/0`.

Those A100 runs establish correctness and same-stack replay only. They ran no
Stage 8 benchmark, profiler, threshold, kernel-count, memory, or performance
measurement. Any Stage 8 restart requires a new Design authority after
Maintenance 6.

Stage 5 does not activate Bernoulli, exponential, Poisson, categorical,
multinomial, rejection, source-quantum, iterative-generation, Charge-stream,
compiled, Triton, fusion, or performance validation. Its exact checks cover raw
words, addresses, fixed-point uniform conversion, zero noise, fixed-seed
same-stack repeatability, and deterministic PSD construction. Its stochastic
checks use analytic estimator uncertainty for white/PSD moments and covariance.
The PSD DC coefficient is exact zero; the sample-domain record mean is bounded
by inverse-FFT roundoff rather than required to equal zero exactly.

Stage 6 activated and cleared the aggregate multinomial and hybrid Poisson
contracts below without changing Stage 5 evidence. Its frozen validation
matrix included at least:

- the frozen Stage 6 statistical policy: seeds `0`, `1`,
  `0x0123_4567_89ab_cdef`, and `0xffff_ffff_ffff_ffff`; `M=2**18` for scalar
  and one-parent laws; `M=2**16` for aggregate `Q=32`, small-grid `K<=3`, and
  completed-`Charge` fixtures; examples rather than correlated cells as the
  independent replicates; target-law standard errors; the predeclared
  `8*SE + 64*eps*max(1,ceil(log2(length)))*abs(scale)` gate; and at least 256
  expected hits and misses for every asserted frequency;
- exact separation between TensorDSLab-model conformance and later IV
  equivalence: no fabricated universal donor percentage, no IV runtime test,
  and an observable-specific collaborator/calibration margin before applying
  the documented `abs(delta_mean)+8*combined_SE <= margin` donor rule;

- timing-jitter probability fixtures from an independent scalar oracle using
  at least 100 decimal digits for the analytically integrated latent-uniform
  plus ideal-Gaussian law, covering `sigma / T` values
  `{2**-52, 2**-40, 2**-20, 1e-3, 0.1, 0.625, 1, 4, 16, 64}`;
  sample counts `{2, 3, 8, 64, 512, 8192}`; central cells; offsets bracketing
  standardized distance `z = 8`; named tails near `z = 20, 37, 38`; farthest
  destinations; both window edges; ideal symmetry; exact represented symmetry;
  category/tail/identity absolute error no greater than `1e-12`; and complete
  represented-source-law L1 error no greater than `1e-11`;
- proof that timing preflight evaluates every destination capable of remaining
  inside the window, introduces no arbitrary tail cutoff, produces finite
  monotone nonnegative one-sided tails and finite nonnegative offset masses,
  exercises the direct/asymptotic evaluator boundary and both asymptotic stop
  conditions, prepares stable success/later-category `A`/`B` masses, and
  rejects rather than clips, assigns a residual, or silently renormalizes an
  out-of-contract law;
- exact `sigma == 0` whole-stage identity with no jitter stream access, no
  replacement tensor, and no perturbation of later role addresses; increasing
  target-bin/category addressing through exact stream
  `CHARGE_TIMING_JITTER = 0x0000_0008` for `sigma > 0`; checked
  `2**-52 <= sigma / T <= 64`, `2 <= S <= 8192`, and `S * N <= 2**63`;
  the overflow-safe picosecond conversion/precheck; rejection immediately
  outside every bound before a word request or write;
  the combined drop category as a final no-draw count remainder; exact source-
  count conservation across retained destinations plus drop; and source
  `Photoelectrons` immutability;
- aggregate timing-jitter ensemble agreement with the analytic multinomial
  means, variances, covariance, displacement moments, and drop probability,
  including an explicit per-PE ideal-law oracle used only for validation; plus
  exact repeatability for the same unchanged numerical execution stack and
  statistical CPU/CUDA agreement for completed jitter unless stronger evidence
  is later ratified; and proof that production jitter neither expands PEs nor
  calls the Box-Muller normal primitive;
- fixed-delay scalar-oracle fixtures for exact zero, fractional offsets,
  representable period boundaries and their immediate `nextafter` neighbors,
  exact two-point mass conservation, source-relative overflow values zero,
  `f`, and one, huge finite all-overflow plans, collapse rejection, no signed
  `source + offset` arithmetic, exact integer-ratio
  nanosecond-to-picosecond conversion without boundary-crossing
  multiplication, and no delay-specific RNG request;
- exponential-delay comparison against an independent at-least-100-decimal-
  digit oracle across the complete supported
  `2**-52 <= mean_delay/T <= 2**52` domain, both sides of the frozen `x=0.5`
  central-mass branch, `2 <= S <= 8192`, every retained category and analytic
  right tail, natural far-tail underflow, and immediate out-of-domain
  rejection; require `1e-12` local absolute identities and `1e-11`
  complete-law L1 error, with no cutoff, clipping, residual assignment,
  renormalization, subtraction-derived overflow, or delay-specific RNG;
- exponential AP conditional-binomial `A`/`B` mass fixtures and stable
  recovery fixtures spanning both accepted delay/recovery ratio domains and
  the auxiliary `2**-51 <= x + y <= 2**53` effective-rate domain, including
  both endpoints, and all three frozen log-difference branches; compare
  `rho_bar[d]`, retained `h[d]`, and `h_ap_tail[L]` with the independent
  high-precision oracle; prove finite `0 <= h <= q`,
  `0 <= rho_bar <= 1`, `h[d] + c*q_(x+y)[d] = q_x[d]`, and
  `h_ap_tail[L] + c*R_(x+y)[L] = R_x[L]` within the local tolerance without
  clipping; and prove recovery changes only deposited charge, while
  `recovery=None` is the exact unit-response law;
- contextual identity fixtures proving `K=0` skips every CT/AP delay and
  recovery numerical gate, a zero CT mean skips that mode's kernel, and zero
  AP probability skips its delay and recovery preparation, including
  structurally valid unused config/sampling pairs outside the active kernel's
  numerical domain; each path remains draw-free and begins no producer write;
- exact zero/one/no-count multinomial/binomial paths, fixed category order,
  stable prepared success/later-category masses without repeated remaining-
  probability subtraction, strict `B < A` complement selection, conservation,
  binary64 probability control, the exact forward-CDF/BTRS
  crossover, inversion recurrence and strict comparison, BTRS proposal,
  support check, quick acceptance and log-bound acceptance, fixed word order,
  reflection/complement timing, and deterministic 64-term/attempt exhaustion;
- independent fixed-word scalar binomial oracles covering inversion acceptance
  at `k = 0`, an interior CDF boundary, represented top-lattice
  success-or-exhaustion, the exact crossover on both sides, BTRS support
  rejection, quick acceptance, full log acceptance, 64 rejected attempts,
  `A == B` without reflection, and `B < A` with complement only after
  acceptance;
- algebraic identity between the stabilized and earlier real-arithmetic BTRS
  bounds, plus an at-least-80-decimal-digit sweep over `n` values including
  crossover cases, `2**32 - 1`, `2**52 - 1`, `2**52`, `2**52 + 1`, and
  `2**53 - 1`; `p_star` at and immediately around `n*p_star = 10`, extreme
  accepted probabilities, and `0.5`; support edges and quick/full-accept
  paths; a `1e-6` absolute local stabilized log-bound gate through 25 standard
  deviations; the frozen mixed per-side allowances and decision-separation
  rule over complete support; fixed-word ownership inside the uncertainty
  band; and a regression demonstrating the retired grouping's large-count
  cancellation;
- exact-zero no-draw Poisson behavior; rejection of negative, nonfinite, and
  greater-than-`1e8` means; inverse-CDF fixtures below `10`; PTRS fixtures at
  and above `10`; acceptance of exactly `1e8`; and deterministic 64-attempt
  exhaustion without reseed, clamp, approximation, algorithm switch, or
  `torch.poisson`;
- at-least-80-decimal-digit Poisson mapping oracles: `1e-12` absolute error for
  every represented inversion term/CDF value, and the frozen
  `1e-6 + 64*eps(float64)*max(1,abs(reference_side))` allowance for each PTRS
  full-accept side; exact high-precision/represented decision agreement when
  the reference sides are separated by more than their summed allowances, and
  fixed-word ownership of decisions inside that finite uncertainty band;
- exact values and noncollision for the five selected Poisson streams, dark
  noniterative addressing, retained-destination versus overflow-source
  generation lattices, `source_quantum = 0`, and no active-compaction-derived
  position;
- exact append-only values `0x0000_0009` and `0x0000_000A` for AP and charge
  smearing, AP's coupled generation/offset/source lattice with overflow fixed
  at category `S` and stop as the no-draw remainder, enforcement of
  `K * (S + 1) * N <= 2**63`, and full-grid smearing positions including
  zero-S2 cells with scalar `z0` use and `z1` discard;
- the universal active-Charge per-cell count ceiling `C_max = 2**53 - 1`:
  exact source/working/frontier/diagnostic/cumulative pass at `C_max`, rejection
  at `C_max + 1`, checked-add success at the exact boundary, forced
  `C_max + 1` failure
  without wrap, conditional remainders bounded by their remaining count, and a
  small tensor containing multiple `C_max` cells to prove there is no whole-grid
  or row population cap;
- independent Poisson mean/count boundaries: successful `lambda = 1e8`
  sampling, an injected accepted proposal above `C_max` causing hard failure rather
  than rejection/clipping, exact dark-rate rational boundary checks, and CT
  rate formation that compares the thinning basis before multiplication and
  rechecks the represented rate;
- exact relational address boundaries and their immediate failures for
  `S*N`, `K*N`, and `K*(S+1)*N`; no arbitrary `K` cap on ineffective
  mechanisms; and proof that the implementation never materializes a complete
  address lattice;
- exact eager reference traversal in increasing generation order; direct CT,
  delayed CT, then AP mechanism order; increasing source bins within each CT
  destination; increasing AP source bins and retained offsets before
  overflow/stop; and rejection of repeated-index scatter/atomic accumulation
  with unspecified order in the reference path;
- accumulator-depth boundaries `L = 2**p_d - 1` and `L = 2**p_d` without
  executing that many generations; float32/float64 unit and recovery-weighted
  ledgers against the frozen `gamma_L*T + L*eta_d` oracle; scientific
  `S2 <= S1 <= T`; exact-zero preservation; and rejection of reassociation
  that claims the eager bound without a new depth proof;
- smearing representation guards at the derived dtype-specific sigma boundary
  and its next representable neighbor, positive config values that round to
  zero or infinity, `S2 == 0`, and fixed raw words realizing the greatest
  Box-Muller radius; all successful pre-clipped and clipped results remain
  finite;
- scalar and exact-shaped binary64 Poisson means, rejection of unsupported
  broadcasting and wrong dtypes, exact-shaped fresh nonnegative `int64`
  results, and a mixed zero/inversion/PTRS tensor proving branch masking does
  not compact or perturb another cell's address;
- analytic Poisson mean, variance, zero probability, selected PMF/tails,
  superposition, dark-count cell independence, and separate DiCT/DeCT retained
  and overflow accounting;
- exact same-stack repeatability and integer-history equality across
  float32/float64 Charge requests, with conditional CPU/CUDA statistical
  agreement for completed Poisson values rather than bitwise equality; and
- synthetic forced-exhaustion and checked-overflow fixtures proving that no
  partial semantic `Charge` is returned after failure, including a maximum
  closed-open uniform with a binary64 rate immediately below `10` to lock the
  selected inversion recurrence's success-or-exhaustion result; source/config
  immutability, no fallback or partial diagnostic exposure, deterministic retry
  on the same unchanged numerical execution stack, and stateless RNG behavior
  after both preflight and dynamic failures; and
- allocation-boundary fixtures proving checked package-planned shape-byte
  arithmetic, ordinary managed/backend allocation and OOM propagation without
  a fabricated memory cap or allocation-free claim, raw storage remaining
  unexposed while writable, and absence of a returned semantic object on
  failure.

Stage 6 activated and cleared these private Charge checks on eager CPU.
Conditional CUDA cases were skipped, so CUDA execution, CPU/CUDA agreement, GPU
performance, and fusion remain later focused evidence. Public request
planning/retention and integration checks likewise activate only under their
own work orders. The sole active correlated-avalanche baseline is the fixed-
maximum-generation model in `rebuild.md`; deleted exploratory algorithm
documents are not implementation sources.

## Parity And Donor-Fixture Rules

[IV-DSLab Parity](parity.md) defines the comparison taxonomy, audited donor
baseline, accepted divergences, and operation-level claims. Tests must not
import or execute IV-DSLab or DSLab at runtime. Every promoted fixture must
name its donor source/snapshot, comparison boundary, parity classification,
units, axes, dtype, operation order, RNG or probability contract, edge policy,
acceptance criterion, and intentional divergences.

Golden fixtures remain small, reviewable, and TensorDSLab-owned. One fixture
does not prove distributional parity. Do not preserve donor global state,
unsigned wraparound, CPU-list conversion, singleton-batch assumptions,
condition-DB loading, remote downloads, or apparent bugs merely for literal
parity.

## Ownership And Scope Checks

Validation and Review should reject accidental introduction of:

- source Photoelectrons production, native G4DS parsing, TensorG4DS clustering,
  a TensorG4DS adapter, or PE binning outside a focused bridge work order;
- request planning, `simulate_readout`, workspace, output buffer, stream,
  lease, selection, movement, or lifecycle behavior outside its focused work
  order;
- durable cache, manifest, IO, scheduler, retry, campaign, or DAG surfaces;
- TensorML model/training/evaluation or Reconstruction concepts;
- a public renderer or `nn.Module` before its focused work order;
- global config, field, builder, validation, Runtime, action, or registry
  dumping-ground modules; explicit product-local `runtime/validate.py` modules
  are required and are not an exception for a generic layer;
- Runtime/Action ABCs, protocols, reflection, generic dependency graphs,
  decorators, untyped action mappings, or broad `utils.py` / `helpers.py`;
- a `PhotoelectronsConfig`, generic `Config` or product ABC, per-product
  collection subclass, or compatibility shim;
- placeholder packages or modules;
- generated caches, outputs, or unrelated files; and
- release, deployment, backward-compatibility, broad compatibility,
  conformance, GPU-kernel, zero-copy, or allocation-free claims.

## Stage 3 Command Baseline

The focused work order defines the exact commands. At minimum, run from the
project root against the selected TensorCore source and an independently
archived exact-pin checkout:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/Users/mbedard/Projects/TensorCore python -m unittest discover -s tests -v
pyright
```

Also run fixed public-import, retired-name, dependency/import-isolation,
ordinary-ABC signature, and static inference probes. Report exact Python,
PyTorch, TensorCore, static-checker, and CUDA evidence. A missing required
static checker, dependency mismatch, dirty fixed candidate, or unexplained
conditional skip prevents Stage 3 clearance.
