# Source notes and research questions

Inspected 2026-09-26. These notes distinguish existing foundations from the
work proposed in [the design](design.md). No upstream builds or proof audits
were run for this sketch.

## Local foundations

| Project | Inspected HEAD | Lean pin | Relevant surface |
|---|---|---|---|
| Ix | `58eb0977a3309f41adaa23aec333d72a944e1cdb` | `v4.33.1` | Ixon, canonicalization, content addressing, kernel checking, and certificates. |
| Ant.ix | `33ea5b9df72c2c2d5f79434cac790ed41ae8f141` | `v4.29.0` | Pure AgentM, replay, and Ixon action admission as design precedents. Its Lake file pins an older Ix revision. |
| Lane | `0df4612428a5038c16cdfdd77e7f25baeb01392e` | `v4.33.0` | Semantic proof operations, private logical workspaces, validation, and integration. |
| Compilatr.ix | `67f84cc799bea155f0f9d5c943f368ea0a170fa1` | `v4.33.0` | Addressed compiler artifacts, checked transformations, staged simulations, and selected native/zk targets. |

Relevant Ix sources:

- [Ixon format](../../ix/docs/Ixon.md) and
  [anonymous canonicity](../../ix/docs/ix_canonicity.md): canonical core and
  separate metadata. Canonicalization is not arbitrary semantic equivalence.
- [Ixon definitions](../../ix/Ix/Ixon.lean): `Constant`, `ConstantMeta`, and
  `Named`. Existing metadata is not a complete record of tactic executions;
  source/tactic instrumentation is additional work.
- [Claim definitions](../../ix/Ix/Claim.lean): `eval`, `check`, `checkEnv`,
  `reveal`, `contains`, and `catalog`, including checking assumptions.
- [Merkle construction](../../ix/Ix/Merkle.lean): `merkleRootCanonical` sorts
  and deduplicates leaves. Ordered datasets need a different manifest layer.
- [Identity boundary](../../ix/docs/kernel_identity.md): durable constant and
  blob addresses are distinct from ephemeral kernel-node identities.
- [Collision boundary](../../ix/docs/tc-context-digest-collision-boundary.md):
  cryptographic collision resistance must not become unrestricted mathematical
  injectivity in a theorem.

Relevant Ant.ix sources:

- [AgentM](../../Ant.ix/Antix/AgentM.lean): import-free inductive interaction
  trees; `Runs`, `GasBound`, `interpMetered_eq`, `driveO_gas`,
  `Runs.deterministic`, and spawn conservation, alongside
  [turn replay](../../Ant.ix/Antix/Turn.lean). These are references for the
  laws of Cybernet.ix's new harness, not a decision to copy the implementation.
- [AgentM design](../../Ant.ix/docs/agentm-design.md) and
  [claims ledger](../../Ant.ix/docs/claims.md): distinguish abstract theorems
  from the concrete runtime and its trusted components.
- [Ixon ingress](../../Ant.ix/Antix/Ingress.lean): named or addressed references
  against a pinned environment; direct term resolution without Lean instance
  search, followed by canonical compilation and kernel checking.
- [Agent-step ingress](../../Ant.ix/Antix/AgentStepIngress.lean): exact expected
  type, bounded normalization, structural decoding, and typed rejection.
- [Ix term integration](../../Ant.ix/Antix/IxTerm.lean): compiling, storing,
  checking, and evaluating addressed terms. Full runtime reuse would pull in
  much more than the pure AgentM core.

Relevant Lane sources:

- [README](../../lane/README.md) and [internals](../../lane/docs/internals.md):
  local elaboration, reusable prefixes, private workspaces, validated writes,
  and separate integration authority.
- [Commands](../../lane/docs/commands.md) and
  [verification implementation](../../lane/Lane/Prove/Analysis.lean): policies
  include expected type, no placeholders, no new axioms, no unsafe declaration,
  and affected-module checks. Cybernet.ix still needs its own explicit,
  transitive release policy; comparison with an existing axiom set alone is
  not an axiom allowlist.

Ix is the intended foundational dependency. Ant.ix, Lane, and Pantograph
inform a new Cybernet.ix harness whose execution and training semantics are
defined together. Their existing runtimes are not the product architecture.
The obsolete `cybernetix-whitepaper` is not a design authority for this library;
Ant.ix's VM and deployment subsystems are outside scope.

Relevant Compilatr.ix sources:

- [Compiler design](../../Compilatr.ix/docs/compiler-design.md): semantic
  stages, bounded checking of analysis/optimization proposals, exact artifact
  identity, representation relations, and preservation of checked fallbacks.
- [Roadmap](../../Compilatr.ix/docs/roadmap.md) and
  [README](../../Compilatr.ix/README.md): the current native path covers selected
  scalar and allocating CFG/source families. Broader source coverage,
  floating-point targets, and GPU compilation must not be inferred from those
  results. There is no inspected ready-made tensor-to-PTX backend.
- [Trust ledger](../../Compilatr.ix/docs/trusted-extern-ledger.md): distinguish
  executable interfaces, bootstrap/build tools, target-model assumptions, and
  mathematical interpreter models. Cybernet.ix should use the same explicit
  separation for arithmetic kernels, launch wrappers, and device compilation.

The reuse proposal is an interface and proof pattern, with a future narrow
bridge. It does not require routing a tensor graph through every existing
Compilatr.ix IR. Ixon v3 ingress and the current backend ownership fragments
have different compatibility scopes; a bridge must preserve their actual
contracts instead of treating a shared file format as a correctness theorem.

## Related work to evaluate

The conversation's corpus–interaction–verifier framing is a useful way to
organize the project. The [NEA article](https://www.nea.com/blog/the-ai-neolab-wild-west)
offers that commercial framing; it does not establish the technical feasibility
or performance of this particular system.

| Project | Useful precedent | Adoption question |
|---|---|---|
| [Certigrad](https://github.com/dselsam/certigrad) | Jointly developing a training implementation, specification, and gradient proof. Its README explicitly separates real arithmetic proofs from floating-point execution and Eigen primitives. | Which proof structure should be rebuilt in Lean 4, with explicit numerical and execution obligations? |
| [lean4-mlir](https://github.com/brettkoonce/lean4-mlir) | Lean-defined forward, backward, and optimizer graphs, with real-valued gradient proofs and a stated trusted lowering boundary. | Can its proof and graph interfaces be reused independently of its backend and dependency pins? |
| [lean-transformer](https://github.com/srush/lean-transformer) | Algebraic transformer invariants; its [source](https://github.com/srush/lean-transformer/blob/main/Transformer.lean) uses rational-valued functional tensors for the initial model. | Which invariance statements transfer to Cybernet.ix's chosen numerical semantics? |
| [Hesper](https://github.com/Verilean/hesper) | Lean tensor/shader programming, AD, and GPU backends. Its README identifies the project as alpha software. | What do the actual theorem statements cover, and where do compilation, FFI, and device behavior remain trusted? |
| [FloatLib](https://github.com/lean-dojo/FloatLib) | Executable bit-encoded floating-point operations with specification/refinement and numerical-error results. | Can a compatible, audited subset implement the exact `P32` primitive programs? Deterministic approximations and conditional correctly rounded transcendental results have different coverage. |
| [TorchLean](https://github.com/lean-dojo/TorchLean) | Shape-aware graphs, real-valued reverse-mode proofs, and explicit backend contracts. | Reuse theorem structure without equating a real AD theorem or a same-device CUDA determinism option with portable execution. |
| [PTXLean](https://github.com/lschiemanowski/ptxlean) | A PTX target model and scalar forward/backward examples connecting instruction execution to numerical computations. | Which admitted instruction, memory, and synchronization fragment can implement our finite graph exactly, and what remains beyond PTX? |
| [Pantograph](https://github.com/leanprover/Pantograph) | Interfaces for Lean frontend interaction, proof execution, expression construction, and environment inspection. | Which semantic operations and observations belong in the small Ixon prover protocol? |

These are candidates and precedents, not audited dependencies. Certigrad's
[original paper](https://proceedings.mlr.press/v70/selsam17a.html) is especially
relevant to distinguishing a gradient-estimator theorem from an entire training
run certificate. The supplied Pantograph rationale URL was unavailable during
inspection; the project's GitHub mirror supplied the interface description.

### Small training examples in lean4-mlir

The follow-up source review used commit
[`373059db8e0a32f2af15b27f9e2a4ed8175eece3`](https://github.com/brettkoonce/lean4-mlir/tree/373059db8e0a32f2af15b27f9e2a4ed8175eece3).
Its [Lean pin](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/lean-toolchain)
and [Mathlib dependency](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/lakefile.lean)
are 4.34.0. No upstream proof build or training run was performed.

The most useful examples are linear/MLP MNIST, character bigram, TinyGPT,
and a small Blackjack DQN. The [roadmap](roadmap.md) links their exact entry
points and derives a smaller implementation progression from them. The
shared verified model definitions in
[`NetsCore.lean`](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/LeanMlir/Verified/NetsCore.lean)
are particularly useful: the trainer and corresponding theorem name the
same object. This is a design pattern to adopt before selecting any backend.

The [linear training capstone](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/LeanMlir/Proofs/Nets/Small/LinearFold.lean)
states a single-example result and explicitly excludes the emitted batch
contraction from that statement. Its byte/render tie also has a separate CI
check. These distinctions make it a useful review example: Cybernet.ix needs
to bind the actual batch reduction and consumed artifact, not just the
single-example real derivative.

The [float bridge](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/LeanMlir/Proofs/Float/FloatBridge.lean)
uses a rounding model with hypotheses and a specified fold. Its error bounds
are distinct from executable bit-level conformance. The
[proof guide](https://github.com/brettkoonce/lean4-mlir/blob/373059db8e0a32f2af15b27f9e2a4ed8175eece3/LeanMlir/Proofs/README.md)
describes remaining numerical, op/text, lowering, and runtime assumptions.
Adoption requires checking actual theorem coverage and compatible pins;
no claim of portable GPU inference or complete training-run certification
is inherited merely by reusing these examples.

### System One and generative-model precedents

[Jev's introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
describes schema-conditioned probabilistic decisions over state, including
parallel questions. It is not a public specification of layer counts or a
reason to define System One as a small fixed classifier.
[Laya](https://laya.convaiinnovations.com/) uses bidirectional encoders and
choice, ordinal, and boolean heads; its published calibration and capacity
limitations are useful experiment design inputs. Product latency and claimed
calibration do not become assumptions about Cybernet.ix.

The [proposed System One interface](model-architecture.md) extends these ideas
to supported compositional Ixon schemas, constructor arguments, dependent
fields, and bounded recursion. That extension is our design, not an assertion
that Jev or Laya already handles arbitrary Lean inductives. Output interface,
domain specialization, and parameter count are independent architecture axes.

[Leanstral-2603](https://huggingface.co/mistralai/Leanstral-2603) supplies a
generative proof-engineering precedent: the model card reports 119B total and
6.5B active parameters with 128 experts, four active per token. Its activity
count does not remove the memory cost of the other experts. The 1.13B and
7.11B dense configurations and 131B sparse configuration in our architecture
are independently proposed profiles, not inferred Leanstral implementations.

### Portable arithmetic and compilation

The [Lean reference](https://lean-lang.org/doc/reference/latest/Basic-Types/Floating-Point-Numbers/)
documents platform variation in native floating point. The initial reference
therefore uses an explicit bit representation and executable arithmetic.
Inspection and use of the installed Lean 4.33.1 sources found a pure
`Init.Data.Float.Model.Float32` implementation, including unpacking, arithmetic,
rounding, and packing. The [certified SGD experiment](certified-sgd.md) now uses
it directly and kernel-checks its toy run and primitive fixtures. This is
separate from compiler-replaced native `Float32` execution. Mathlib v4.33.1
also builds with this pin and supplies the real calculus proofs.

FloatLib remains an integration candidate for further numerical proofs and
elementary functions; its inspected
[toolchain](https://github.com/lean-dojo/FloatLib/blob/main/lean-toolchain) is
4.34.0, which must be reconciled with Ix's current pin.

[TorchLean's trust boundaries](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md)
separate graph calculus from native execution. Its deterministic CUDA option
documents a narrower scope than cross-architecture/toolchain bitwise
reproducibility. [Batch-invariant inference work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
also motivates making reduction trees, masks, and serving transformations
explicit. Neither source proves Cybernet.ix's chosen portable profile.

PTXLean was inspected at
[`5e1db82fc6f953882c7a538d11798cbf95415e43`](https://github.com/lschiemanowski/ptxlean/tree/5e1db82fc6f953882c7a538d11798cbf95415e43),
with Lean 4.34.0 and a stated PTX ISA 9.4 target. The web reader could not fetch
the repository; the public GitHub API and raw source supplied the inspection.
No upstream build or GPU execution was run.

- [PtxReluKernel.lean](https://github.com/lschiemanowski/ptxlean/blob/5e1db82fc6f953882c7a538d11798cbf95415e43/integration/torchlean/PtxReluKernel.lean)
  exposes `run_correct`, `pipeline_correct`, `pipeline_exists`, memory safety,
  and frame results for the modeled scalar launch sequence. They concern the
  actual modeled instruction runs under target eligibility and storage/launch
  premises.
- [PtxReluAccuracy.lean](https://github.com/lschiemanowski/ptxlean/blob/5e1db82fc6f953882c7a538d11798cbf95415e43/integration/torchlean/PtxReluAccuracy.lean)
  derives stored forward/backward error bounds. The backward interpretation
  needs a margin keeping rounding from changing the ReLU branch, as well as
  finite/range assumptions. This is neither differentiation of rounding nor
  bitwise equality with an arbitrary framework implementation.
- [The example guide](https://github.com/lschiemanowski/ptxlean/blob/5e1db82fc6f953882c7a538d11798cbf95415e43/docs/foundations/relu-neuron.md)
  separates modeled sequential launches from host runtime and hardware
  conformance. Multithread reductions, broad tensor kernels, and the
  Cybernet.ix-to-PTX bridge remain work to do.

This makes PTXLean a concrete candidate for the target side of
[checked compilation](portable-inference.md), complementing Compilatr.ix's
artifact and preservation discipline. The first question is exact scalar and
reduction refinement; adopting a project name is not a substitute for that
theorem or for concrete execution evidence.

## What survives from the 120B sketch

The [existing note](../verified-120b-v2.md) remains unchanged. Its useful ideas
include separating mathematical and executable semantics, explicit randomness,
parameterized proofs, checkpoint replay, and experiments before large runs.
The following claims need correction or qualification before reuse:

1. **Exact accumulation does not imply whole-step layout invariance.** Rounding
   or rescaling partial sums before combining them changes results. For floor
   rounding, `floor(1/2) + floor(1/2) = 0`, whereas `floor((1+1)/2) = 1`.
   Prove the entire quantization and reduction schedule, including exponent
   alignment and overflow, for each supported layout.
2. **Power-of-two attention rescaling can lose bits.** A right shift is exact
   only under suitable divisibility or representation conditions. Exponent
   range, approximations, retained precision, and final normalization all
   belong in an attention equivalence theorem.
3. **Fixed checksums do not certify every matrix product.** Let
   `v = (1, -2, 1)` and `E = v vᵀ`. This nonzero error matrix has zero checksums
   against both `(1, 1, 1)` and `(1, 2, 3)`, from either side. Exact modular
   checks also need a bound or representation argument to establish equality
   over integers. A randomized checker requires a soundness theorem, a stated
   error probability, and challenges independent of the committed result;
   fixed public checksum vectors do not provide that guarantee.
4. **Elementwise floating point still needs a specification.** Rounding mode,
   contraction, exceptional values, and approximations to elementary functions
   can differ even when there is no parallel reduction.
5. **A surrogate backward rule is not the derivative of a quantizer.** The
   certificate must name the estimator being implemented and the conditions
   of any claim relating it to a mathematical gradient.
6. **Identical arithmetic alone does not make RL on-policy.** Weights, context,
   decoding distribution, rollout timing, and treatment of stale samples must
   also match the objective's policy assumptions.
7. **A transcript is a commitment to a claim about a run.** It becomes evidence
   of correct computation only when a suitable checker, evaluation proof, or
   execution theorem connects the committed inputs and outputs.

No hardware throughput, large-model quality, compute-cost, or legal-threshold
claims from the earlier note are carried into the new architecture.
