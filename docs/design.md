# Cybernet.ix: first-principles design

Draft, 2026-09-26. This document describes proposed interfaces and proof
obligations. Equations and names below are specifications to develop, not
claims that the corresponding Cybernet.ix proofs already exist.

## 1. The system we want

Build models whose training, portable inference, and interaction protocols
can carry Lean proofs certified by Ix. Support System One inference over typed
data, specialized autoformalization, and eventually general models at frontier
scale. Canonicalized Ixon supplies
the corpus; a new Cybernet.ix harness defines both the interaction semantics
and the episodes we train on; the Ix kernel supplies the verification function.
AgentM and Lane/Pantograph are mathematical and interface precedents, rather
than an existing runtime to copy or compose wholesale.

This creates a concrete learning loop:

1. Project an addressed formal corpus and prior interactions into training
   examples with readable names, terms, tactics, and goal context.
2. Train a model using a specified Lean program and publish weights with
   precisely scoped correctness claims and Ix certificates.
3. Run deterministic inference to propose typed Ixon actions in a Cybernet.ix
   interaction tree, with the same observation and action meanings used in
   training.
4. Apply those actions to a Lean workspace, check resulting proof artifacts
   with Ix, and retain the observations and their verification results.
5. Certify the turn's transition and admit selected episodes into the next
   corpus snapshot.

The [model architecture](model-architecture.md) separates output interface,
task specialization, and scale. System One supports typed probabilistic
questions and structured values; a small finite classifier is its first
implementation. The concrete generative targets are dense 1.13B and 7.11B
Transformers, with a sparse 131B-total configuration defining the distributed
scaling direction. Shared graph and proof interfaces cover all these sizes.
Capability, throughput, and certificate cost determine when to fund each run.

Portable deterministic inference is a core requirement: identical complete
semantic inputs produce identical output bytes on every conforming backend.
Start with bit-specified binary32 arithmetic, explicit reduction trees, and
pinned elementary-function programs. Hardware, batch composition, and cache
use cannot change the model's function.

Lean owns the specifications, training control flow, inference semantics, and
interaction programs. C or Rust FFI can supply storage and accelerated numerical
execution behind explicit contracts. Python is not required in the library's
training or execution path.

## 2. One harness for execution and learning

The central object is a versioned Lean `HarnessSpec`. It defines an
interactive environment and the interpretation of its learning episodes.
Model weights implement a policy within that environment. A harness definition
can itself be addressed, checked, and included in a training corpus.

| Component | Shared meaning in execution and training |
|---|---|
| State | Task, formal environment, candidate artifacts, interaction history, and remaining authority/budget. |
| Observation | The selected view of that state available to the model, with its context and symbol table. |
| Proposal and admission | What the model may emit; how it resolves to an admissible typed action or an explicit rejection. |
| Action and response | The dependent tool protocol, including proof attempts, queries, errors, and completion. |
| Transition | The pure state update given an admitted action and its recorded response. |
| Outcome and reward | Task success, failure, termination, and the specified learning signal derived from checked evidence. |
| Cost and delegation | Charges, grants, and the rules governing further interaction. |
| Episode | An ordered derivation of observations, proposals, responses, transitions, and outcomes under this specification. |

A `TaskSpec` instantiates the target and accepted proof policy. A `ViewSpec`
fixes the readable rendering, available premises, symbol resolution, and token
projection. These belong to the same addressed experiment definition as the
harness, model, objective, and numerical profile.

Conceptually, the core supplies:

```text
observe(H, V, state)                         → observation
admit(H, state, proposal)                    → rejection | action
advance(H, state, action, recordedResponse)  → nextState
outcome(H, task, episode)                    → result and reward
project(H, V, episode)                      → training example
```

The transition is deterministic given recorded responses. Proof search and
other effects acquire meaning through handlers with explicit contracts.
Invalid proposals are recorded as rejection events, so failed attempts remain
valid learning episodes without becoming dispatched actions. Rejection
preserves the formal environment while recording feedback and consuming the
specified budget. Tool responses likewise pass their declared validation or
assumption boundary before a state update.

Training examples are projections of this semantics, not independently
assembled chat logs. The same observation renderer and action resolver serve
training and inference. Record roles, loss masks, terminal conditions, and
the exact sampled token sequence. An auxiliary objective may learn proof terms
or tool responses, but it is distinguished from the policy's action objective.
RL likelihoods must specify whether they describe token sequences or canonical
actions; re-rendering an action cannot silently replace the sampled sequence
when computing a probability ratio.

Initial proof obligations connect these uses: observations match their episode
prefixes; rendered actions resolve to the expected terms; admission preserves
authority; transitions preserve state invariants; replay reconstructs the same
state; and the training projection preserves the runtime's action meanings.
These are also the basis for certifying turns and reward computation.

For example, an episode can expose `n : Nat ⊢ n + 0 = n`, receive a proposal
to construct `Nat.add_zero n`, check that term against the fixed target with
Ix, and close the goal. The same record supplies an observation/action pair
for supervised learning, a checked success signal for RL, and a witness for
the turn transition. Its named view is what the model reads; its addressed
terms are what the verifier checks. These interpretations share the episode's
harness and environment identity.

Learned policies, data collectors, trainers, evaluators, and certificate
producers all consume this definition. We can change a model or tool backend
without changing the protocol, or introduce a new harness version with explicit
compatibility evidence. Training on the harness does not grant the model
authority to alter its own active specification.

## 3. What a certificate says

Certification attaches to a proposition about identified artifacts. A model
release needs a manifest of claims rather than an undifferentiated
`certified : Bool`.

| Claim | What must be established |
|---|---|
| Mathematical training correctness | The backward computation implements the stated derivative or explicitly specified gradient estimator, under its preconditions. |
| Numerical correctness | The executable arithmetic implements its finite numerical specification; any relation to ideal real arithmetic has stated bounds and conditions. |
| Training program correctness | The implementation refines the specified state transition, including data selection, randomness, and optimizer state. |
| Training run correctness | The particular final weights are related to the declared initial state, data sequence, and training program by the checked computation. |
| Inference correctness | The particular output follows from the identified weights, model, input, numerical rules, and decoding state. |
| Interaction correctness | The turn is a valid transition of the selected Cybernet.ix harness and interaction program, with its event contracts, capability restrictions, and budget. |
| Proof-task success | The returned proof has the required proposition as its type in the accepted environment. |

These claims compose when their subjects and assumptions match. A derivative
theorem does not establish a GPU execution claim. A checked conversation
transcript does not establish an inference claim. A proof of a generated
proposition does not establish that the proposition captures the user's intent.

### Training code and particular weights

Let `P` identify a training specification. Its executable step is conceptually:

```text
step(P, state, batch, randomTape) : Except NumericalError TrainState
```

The state includes parameters, optimizer accumulators, step number, data
cursor, random-stream position, and any loss-scaling or accumulation state.
The specification fixes the model, loss, optimizer, schedules, and arithmetic.

Training orchestration uses the same addressed program and episode conventions
as model interaction: batch selection, numerical execution, evaluation, and
checkpoint publication have declared inputs and results. Tensor operations
retain their pure numerical semantics. The trainer and the learned policy can
have different capabilities within this common framework.

A successful finite run establishes:

```text
S₀ = initialize(P, initializationInputs)
for every i < N:
    Bᵢ = selectBatch(P, corpus, dataSchedule, i)
    step(P, Sᵢ, Bᵢ, Rᵢ) = ok Sᵢ₊₁
weights(Sₙ) = W
```

The consumer selects the expected plan, corpus commitment, initialization,
step count, and claim policy. A producer cannot substitute a zero-step run,
an arbitrary initial checkpoint, or a weaker proposition and still satisfy
that request. Imported base weights are an explicit initial checkpoint:
certifying their fine-tuning certifies that suffix of training.

Two distinct artifacts support the release:

- A **program certificate** covers reusable theorems about the training
  implementation and its specification.
- A **run certificate** binds the concrete initial state, ordered inputs,
  randomness, and final weights to a valid execution of that implementation.

For a small pure computation, a run witness can be checked through evaluation
or a Lean proof of the concrete relation. At scale, it may compose proofs of
individual steps or sound computation checkers. Connecting those witnesses to
the numerical backend is a central research obligation. A signed log, hashes,
sampled replay, or proof that the trainer typechecks cannot fill that gap.

The mathematical claim is that the weights are the result of the declared
computation. Claims about when or where a physical run occurred require
separate provenance evidence. Neither claim establishes model usefulness.

### Carrying these claims with Ix

The proposed path is:

```text
formal specification + proof term
    → Ixon compilation and addressed dependency closure
    → Ix checking of the required statement and proof
    → Ix certificate for that check
    → consumer verification against expected artifact and claim roots
```

Ix already defines checking, evaluation, membership, and other claim forms.
Cybernet.ix should use these to certify its own Lean propositions, without
inventing a new general proof system. A check of an arbitrary well-typed
constant is insufficient: admission must bind the exact expected proposition
to the checked proof and its environment. Compilation must preserve that
binding. See the [Ix source notes](research-notes.md).

Every claim bundle records the subjects, specification and checker versions,
logical axiom policy, unresolved Ix checking assumptions, proof artifacts, and
any external execution assumptions. Ix's assumptions about dependencies being
well-typed are distinct from Lean's logical axiom dependencies. Both matter.
The default proof policy rejects `sorryAx` and unapproved axioms throughout
the dependency closure. Lean supports auditing those dependencies; native
evaluation and FFI require their own trust review.
[Lean's axiom reference](https://lean-lang.org/doc/reference/latest/Axioms/)
explains why typechecking alone cannot justify arbitrary assumed axioms.

Cryptographic binding and certificate soundness have explicit cryptographic
assumptions. Hashes are not globally injective mathematical functions.
Succinct verification also does not make proof generation inexpensive; the
first prototype must measure both costs.

## 4. Canonical Ixon as a learnable corpus

Ixon is the durable representation. A model's token stream is a deterministic,
versioned projection of that representation. It can present readable Ix
source, explicit kernel terms, recorded tactic scripts, or structured tool
events without asking the model to predict long content hashes.

Three identities need to remain separate:

| Identity | Meaning |
|---|---|
| Statement | A proposition and its dependency environment. |
| Proof artifact | A particular checked term of that proposition. Different proofs can have different Ixon addresses. |
| Presentation or derivation | Source text, naming, tactic history, and elaboration context associated with that artifact. |

Anonymous canonicalization groups terms according to Ix's supported
canonicalization rules. It does not identify every proof of the same theorem,
nor decide arbitrary semantic equivalence. Multiple source presentations can
share an anonymous artifact; multiple proof artifacts can establish one
statement. The corpus should preserve both relationships.

### Names and model-facing references

Use an immutable, versioned symbol registry with one primary readable name per
anonymous address and one address per primary name. Human source aliases and
presentation variants live in separate tables. Names such as `Ix.Nat.add_comm`
are a presentation convention, not a replacement for addresses.

Each example binds its registry and dependency environment. Retrieval can
supply a smaller local table with types and relevant definitions. A generated
reference is resolved through that table and checked before becoming an
admitted Ixon term. Adding a declaration must not silently retarget an existing
name. Cross-registry comparisons use addresses.

Ant.ix's Ixon ingress already accepts named references against a pinned
environment. This is a useful starting point for action generation, although
the larger corpus registry and its projection laws remain new work.

### Tactics, source intent, and elaboration

Kernel terms alone cannot reconstruct which tactics produced them. Record
source and elaboration evidence when compiling: source syntax and locations,
tactic inputs, before/after goals, local contexts, relevant options and imports,
resolved instances and implicits, and the resulting terms. Give these records
their own addresses linked to the anonymous core.

Keep separate acceptance properties for:

- **Term projection:** parsing and resolving a rendered core term reproduces
  its expected anonymous artifact under the pinned environment.
- **Source replay:** a recorded source variant elaborates under its recorded
  toolchain and environment and checks against the intended statement. Exact
  proof-address equality is an additional requirement when claimed.
- **Intent correspondence:** the formal statement represents the informal
  task. This needs evidence beyond proof checking, such as review, examples,
  counterexample search, or a separately formalized source specification.

Typeclass inference, coercions, notation, and tactic extensions belong in the
recorded elaboration context. Begin with instrumentation of the existing
frontend. A modified elaborator is an option to evaluate, not an architectural
prerequisite. An alternative elaborator may generate a different valid proof;
it must still establish the fixed expected statement and label its provenance.

### Corpus snapshots

A snapshot binds the input artifacts, harness/task definitions,
projection/tokenizer versions, registry, sampling rules, and split assignment.
Training sequences retain order and multiplicity, including intentional
repeated samples. Ix's canonical Merkle
tree represents a sorted, deduplicated set, so it cannot alone commit to a
training sequence. Use ordered manifests or index-bearing leaves with checked
length and index coverage.

Group anonymous duplicates, source variants, and known related derivations
before assigning evaluation splits. Record premise availability at each task;
the desired proof or future environment must not leak into the observation.
Canonicalization helps deduplication, but cannot by itself detect all semantic
duplicates or benchmark contamination.

## 5. Training and portable deterministic inference

The numerical foundation needs three connected interpretations:

1. A mathematical model, usually over real-valued tensors, for calculus and
   claims about the objective.
2. An executable numerical model specifying bits, rounding, reduction order,
   quantization, exceptional values, and randomness.
3. A backend implementation with a stated relation to that executable model.

A typed tensor graph should describe shapes, operators, parameters, and
sharing independently of buffer layout. Reverse-mode differentiation carries
proofs for each supported primitive and a composition theorem. Graph rewrites
and buffer layouts have separate preservation obligations. Accelerated kernels
may arrive later without changing what a graph means.

For smooth operations, prove the backward pass implements the appropriate
derivative on a stated domain. For sampling, a theorem about an unbiased
estimator additionally needs a probability model and its analytic conditions.
For quantization, discrete routing, or straight-through estimation, name the
surrogate rule explicitly. The derivative of a real model is not the
derivative of its rounded implementation.

The initial profile is portable binary32, provisionally `P32`. It fixes
rounding, subnormals, exceptional values, individually rounded products,
canonical balanced reduction trees, explicit masks, and finite programs for
elementary functions. RoPE tables are model artifacts. Native floating-point
defaults and ordinary GPU library calls are insufficient contracts.

Inference identity includes weights, graph, tokenizer, context rendering,
output schema, numerical profile, decoding policy, and random state. Begin
with greedy selection and canonical ties. Sampling consumes an explicit tape
under a specified integer-bin categorical rule. Backend identity does not
license a different answer. Caching, batching, incremental decoding, and
device placement require bitwise equivalence to one reference function.

The [portable inference specification](portable-inference.md) defines these
rules and their proof obligations, including the path from a tensor graph to
checked GPU kernels. PTXLean is a candidate target-semantics foundation;
Compilatr.ix supplies a precedent for addressed compiler artifacts, checked
optimization, and staged preservation proofs. A GPU bridge and the downstream
runtime/target evidence remain new work.

Training uses the same finite forward model, with an explicit backward rule
and optimizer transition. The toy demonstration uses SGD; the Transformer
pilot uses specified AdamW updates. Physical microbatching cannot change the
logical batch or its accumulation schedule. Imported weights expose the
certified training boundary. Higher-performance BF16 or integer profiles can
follow with separate identities and proofs while retaining portable inference.

## 6. The Cybernet.ix interaction calculus

Define a new harness around the shared `HarnessSpec`. Take the pure ideas from
AgentM as a starting point: effect signatures, interaction trees, trace
semantics, replay, gas bounds, and delegation. Establish the corresponding laws
for Cybernet.ix's own learning and execution semantics. Ant.ix's existing tool
vocabulary, runtime, VM, deployment, and host-management layers are outside
this design.

The model proposes first-order action data. A Cybernet.ix interaction program
determines how to observe the state, admit a proposal, perform its effect,
incorporate its response, and produce the next decision point. Those decision
points are exactly the ones projected into training examples. This keeps the
learned policy replaceable while the protocol has a fixed meaning. Larger
certified plans can be added after the single-action interface.

Possible initial tool families, with names still provisional:

| Operation | Observation or result |
|---|---|
| Inspect target | Statement, goals, local context, and workspace version. |
| Retrieve premises | Addressed declarations and their types from the allowed environment. |
| Attempt proof | Candidate artifact or positioned diagnostics, with the before/after state. |
| Check candidate | Result for a fixed statement, dependency closure, and proof policy. |
| Fork or finish workspace | A versioned private workspace or immutable candidate bundle. |
| Request integration | A candidate for the integrating authority to check against the current base. |

Operations and their dependent response types form a signature in the same
mathematical sense as `AgentM.Sig`. The new harness also specifies their state,
observation, and learning relationships. Payloads, responses, model
observations, and events have Ixon representations.
Source or tactic text is data inside a typed operation. Conversation text can
also be represented as Lean data; storing it in Ixon does not make its contents
true or executable.

The admission design takes the following lesson from Ant.ix: bounded parse
and name resolution, canonical compilation, exact expected-type checking,
kernel checking, normalization where required, structural decoding, and capability/state
validation before dispatch. Grammar-constrained output helps syntax; it cannot
replace semantic checks or prove a generated proof argument.

The budget theorems should retain AgentM's quantification over all possible
model responses. Gas is a cost function on abstract operations. Inference,
parsing, failed proof attempts, checking, and delegation must all be charged
in the selected plan. Physical time or memory bounds need an implementation
contract beyond the abstract theorem. A timeout is an explicit failed attempt.

### Lane and Pantograph concepts

Use semantic interaction with Lean: inspect goals and premises, try proofs
against a reusable prefix, receive structured diagnostics, and validate the
containing module and affected dependencies. Private workspaces isolate
experiments; compare-and-swap updates prevent stale candidates from replacing
newer work. Integration authority is distinct from proof-search authority.

Lane supplies a concrete precedent for that workflow. Pantograph supplies a
precedent for a machine-facing Lean frontend. Cybernet.ix's tools should act on
its addressed formal states and candidate artifacts; file views and frontend
sessions are implementation choices. A Lane or Pantograph adapter may provide
a handler, but its command set does not define the harness. The handler's
correspondence to the shared semantics needs its own evidence.

Keep two task types: a proof task fixes the formal proposition before search;
an autoformalization task first proposes a proposition for an informal input.
Proof checking rewards the former. The latter retains an additional statement
alignment obligation even if the proposed statement has a valid proof.

The corpus retains accepted and rejected attempts with their labels, context,
model version, action, checker policy, diagnostics, and resource observations.
An RL reward is a versioned function of this evidence. Successful proof reward
requires the original target under the accepted axiom policy; changing the
goal to `True` or adding the conclusion as an axiom cannot earn it. Rollouts
pin the behavior model and sampling policy so later updates can account for
stale or off-policy data.

Models may help prove the library's training theorems. Their outputs still pass
through the independently pinned verification boundary; the learner does not
select its own checker or acceptance policy.

## 7. Certified turns and checkpoints

A turn manifest binds the parent state, interaction program and its grant,
model release, rendered observation, decoding/random state, ordered events,
resulting state, and evidence dependencies. Conceptually its proof establishes:

```text
ValidTurn(T, start, trace, finish) :=
    the selected Cybernet.ix program follows trace from start to finish
    ∧ every admitted action satisfies the selected capability policy
    ∧ model responses satisfy their declared inference relation
    ∧ successful prover results satisfy their required statements
    ∧ abstract cost is within the selected bound
```

The actual formalization must specify how state transitions and per-event
contracts compose with the new interaction-tree execution relation, following
the role of `AgentM.Runs`. Existing AgentM theorems are a reference for this
development; they do not automatically prove the new harness or handlers correct.
External observations remain explicitly assumed unless supported by a checked
relation. Diagnostics and timing observations are not mathematical claims
merely because they appear in the trace.

Publish the addressed turn, its proof, and an Ix certificate for the exact
turn proposition. Determinism enables replay and a well-defined inference
relation; evidence still has to establish that relation. A certificate for
protocol conformance with an assumed model response must be distinguishable
from one that also certifies the neural computation.

Linking turns by their parent commits to a history. A checkpoint can commit to
an ordered prefix and compose available certificates. A consumer needs the
expected specification, parent, claim policy, and verification parameters;
data availability is a separate concern. These artifacts can be stored on a
central server, exchanged directly, or later anchored on a blockchain. The
library does not require a consensus protocol or token.

## 8. Library boundaries

Proposed package name: `cybernetix`; Lean namespace: `Cybernetix`.

| Module area | Responsibility |
|---|---|
| `Harness` | The shared state, observation, action, episode, reward, and interaction-tree semantics used by learning and execution. |
| `Artifact`, `Certificate` | Typed references, manifests, expected statements, proof policies, and Ix integration. |
| `Corpus` | Canonical registries, source/term/trace projections, tokenizer, splits, and ordered datasets. |
| `Tensor`, `Numeric` | Shape-aware graphs, arithmetic semantics, layouts, and reference evaluators. |
| `Model` | Parameterized backbones and output heads, schema-directed prediction, and concrete model profiles. |
| `Training` | Harness-derived objectives and episodes, differentiation, optimizers, training transitions, and run composition. |
| `Inference` | Forward evaluation, caches, decoding, and deterministic sampling. |
| `Formalization` | Goal observations, private workspaces, Lean tool adapters, and reward definitions. |
| `Backend`, `Compilation` | CPU/GPU implementations, checked graph/kernel transformations, target semantics, and artifact-specific refinement contracts. |

Small schemas, programs, and proofs can be Ixon constants. Large tensor payloads
need addressed byte chunks and typed manifests fixing dtype, shape, order,
encoding, and chunk layout. A loader must establish the bytes-to-tensor
representation relation; wrapping an address in a Lean type does not validate
its contents. Storage identity and semantic tensor equality remain distinct.

Pin one compatible Lean/Ix environment and selected dependencies for each
release. The inspected references use different versions: Ix uses Lean 4.33.1,
Ant.ix 4.29.0, Lane and Compilatr.ix 4.33.0, and inspected PTXLean/FloatLib
sources use 4.34.0. Design precedents do not automatically become package
dependencies. Resolve pins and validate narrow adapters before integration.
The new harness and training core will be developed together in Cybernet.ix.
The [Lean/Rust build scaffold](development.md) is implemented; integration
of the numerical, harness, and certificate dependencies remains future work.

## 9. Implementation progression

The [implementation roadmap](roadmap.md) owns the detailed milestones and
acceptance gates. Begin with shared harness/artifact definitions and an Ix
claim experiment, then a complete toy SGD training/inference/turn certificate.
Linear MNIST and a small MLP test useful learning before introducing sequence
models. A byte bigram and a 426,624-parameter Transformer exercise the language
path well before the proposed 44M encoder or billion-parameter provers.

Each small task uses the same output, episode, and training interfaces. MNIST
is a finite-choice System One instance, while source/term/trace experiments
develop the richer typed and generative interfaces. The bounded Ixon corpus
and Lean tool adapter can develop once the harness contracts exist, alongside
the numerical work. A preexisting model can bootstrap episodes with explicit
external inference/training assumptions.

The first complete run certificate covers a computation small enough to check
directly. Larger training examples must report their actual program, backend,
and concrete-run coverage separately. Useful MNIST accuracy does not establish
a certificate for every training step, and a gradient theorem does not certify
a GPU runtime.

The main feasibility gates are portable numerical throughput, useful learning
from the canonical corpus, and economical evidence for concrete training and
inference. The [architecture](model-architecture.md) retains the larger
configurations and resource accounting; the roadmap makes the experiments
that must justify scaling explicit.
