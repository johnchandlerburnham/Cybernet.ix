# Model architecture

Proposed architecture, 2026-09-26. These are concrete starting configurations
and implementation targets, not trained models or completed proofs. The
[system design](design.md) defines the shared harness and certificate claims;
[portable inference](portable-inference.md) fixes the numerical contract.
The [implementation roadmap](roadmap.md) owns delivery order: toy SGD,
linear MNIST, a small MLP, byte bigram, and a tiny Transformer precede the
larger reference configurations below.

## 1. Model interfaces, specialization, and scale

Cybernet.ix should support System One inference over typed Ixon data,
specialized autoformalization, and general models at frontier scale. These
describe different axes: output interface, task specialization, and capacity.
A System One model can be large and general; an autoformalization model can
have both typed prediction heads and a token decoder. They share a tensor
language, numerical semantics, artifacts, training machinery, and harness.

| Use | Larger reference implementation | Output | Intended role |
|---|---|---|---|
| System One | Bidirectional Transformer with schema-conditioned prediction heads; finite-choice head first | Typed predictions, distributions, scores, and structured values under a declared output schema | Learned functions over program state, including multiple questions and structured tool decisions |
| Prover | Dense causal Transformer with grouped-query attention | Tokens encoding source, proof terms, tactics, and typed tool proposals | Specialized autoformalization and proof engineering |
| General | Causal Transformer, extending the dense core with sparse mixture-of-experts feed-forward layers | General text/code and typed tool proposals | A path to large general models, retaining the same inference and certificate contracts |

System One means direct, schema-conditioned probabilistic inference. Its
interface is not defined by a small parameter count or a fixed label set.
Jev emphasizes structured state, typed probabilistic answers, and parallel
questions; Laya implements choice, ordinal score, and boolean predictions.
Our extension to compositional Ixon types is a Cybernet.ix design proposal.
Their product descriptions do not specify our network, and schema validity
does not establish factual correctness.
[Jev introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[Laya architecture](https://laya.convaiinnovations.com/).

Leanstral is a useful precedent for the prover's repository-and-tool workflow.
Its March 2026 model card describes 119B total parameters, 6.5B active, and
128 experts with four active per token. A specialized formalizer therefore
need not be small; our dense pilot deliberately starts at a more accessible
scale. These configurations are our proposals, not reproductions of Leanstral.
[Leanstral-2603 model card](https://huggingface.co/mistralai/Leanstral-2603).

## 2. Concrete reference configurations

All counts below include embeddings, normalization scales, and output heads.
`d` is residual width, `m` feed-forward width, and `Q/KV` attention head counts.
Token budgets include the complete model-visible sequence. They are supported
limits to implement, not demonstrated long-context quality.

| Profile | Layers | d | m | Q/KV | Vocabulary | Context limit | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| SystemOne-44M, choice baseline | 8 | 512 | 1,536 | 8/8 | 32,768 | 2,048 observation; 256 per candidate | 44,311,040 |
| Prover-1B | 24 | 2,048 | 5,504 | 16/4 | 32,768 | 8,192 | 1,130,465,280 |
| Prover-7B | 32 | 4,096 | 14,336 | 32/8 | 32,768 | 32,768 | 7,113,805,824 |
| General-MoE | 40 | 4,096 | 2,048 per expert | 32/8 | 131,072 | 32,768 initially | 131,084,914,688 total; 6,262,427,648 active-count proxy |

General-MoE uses 128 experts per layer and selects four per token. The active
count includes the complete tied embedding/output table and selected experts;
it is a rough computation proxy, not a resident-memory figure or exact FLOP
count. Every expert still contributes to storage and training state.

Parameter accounting is explicit. With `L` layers, vocabulary `V`, `q` query
heads, `h` KV heads, and two normalization vectors per block:

```text
attention parameters per block A = 2 d² (1 + h/q)
SwiGLU parameters per block    F = 3 d m
normalization parameters       N = (2L + 1) d
dense decoder parameters         = Vd + L(A + F) + N
decision parameters              = Vd + L(A + F) + N + d²
MoE total parameters             = Vd + L(A + E F + dE) + N
MoE active-count proxy           = Vd + L(A + k F + dE) + N
```

The first certification experiment uses toy SGD, followed by linear MNIST
(7,850 parameters), an MLP (101,770), a byte bigram (65,536), and a two-layer
causal Transformer (426,624). Their configurations and acceptance gates are
in the [roadmap](roadmap.md). Prover-1B is a later dense-prover pilot after
the small-model, corpus, arithmetic, and backend gates pass. Prover-7B and
General-MoE make the scaling requirements concrete; they are not commitments
to immediately fund runs of those sizes.

## 3. The shared Transformer block

Choose pre-normalized residual blocks, RMSNorm, rotary position embeddings
(RoPE), SwiGLU feed-forward layers, and bias-free linear maps. Use no dropout
in these initial profiles. The encoder uses bidirectional attention; the
decoders use full causal attention and grouped-query attention (GQA).

In mathematical notation, for a sequence of residual vectors `x`:

```text
u  = RMSNorm(x, gamma_attn)
x' = x + Attention(RoPE(u Wq), RoPE(u Wk), u Wv) Wo
v  = RMSNorm(x', gamma_ffn)
y  = x' + (SiLU(v Wgate) * (v Wup)) Wdown
```

There is a final RMSNorm before pooling or the language-model head. Each
normalization has a learned scale and no additive bias. Epsilon is the binary32
rounding of the rational `1/1,000,000`. Decoder embeddings and the final token
projection share one parameter tensor, including its gradient contributions.
There are no learned absolute position embeddings or separate output biases.

Attention scales QK scores by `1/sqrt(headWidth)`. Query heads map to contiguous
groups of KV heads. RoPE rotates adjacent even/odd coordinates over the full
head width, starting positions at zero. Start with theta 10,000 for Decision
and Prover-1B, and 1,000,000 for Prover-7B and General-MoE. Position tables are
immutable bit patterns in the model artifact. Increasing context or changing
RoPE scaling creates a new model version and requires quality evaluation.

These equations describe the real-valued model. Their executable meaning also
fixes each primitive, cast, intermediate rounding, mask, and reduction tree.
For example, the executable softmax and SwiGLU use the selected finite `exp`
program; their gradient specifications must state how they relate to the real
functions. Algebraically equivalent expressions are not interchangeable under
the [portable numerical profile](portable-inference.md).

### Tokens and addressed data

Use a byte-fallback BPE tokenizer with 32,768 total tokens, including all 256
byte values and reserved protocol tokens, for the initial decision and prover
profiles. Fit it to the selected mixture of named Ix terms, recorded Lean
source and tactics, natural language, and code. Store the vocabulary, merge
ranks, boundary rules, and special-token IDs as addressed artifacts. Resolve
equal-priority merges by leftmost position. Do not rely on host regex or Unicode
normalization defaults. Decoding ordinary token spans must recover their bytes.

Special role and field tokens are inserted by the harness renderer, not
recognized as authority-bearing delimiters inside arbitrary user text. Symbols
appear through the pinned readable registry and local reference table. The
model can emit a short reference such as a local premise index; admission
resolves it to the addressed term. It need not learn to predict content hashes.

The larger general profile uses a separate 131,072-token vocabulary trained
for its broader corpus. The library supports different vocabularies; the first
two families sharing a tokenizer does not require all future models to do so.
Changing tokenization changes model identity and generally requires training
or a checked conversion, not just swapping a configuration file.

## 4. System One: typed inference over program state

Separate a `BackboneSpec` from its `OutputSpec` and training objective. The
backbone extracts representations of a state and its questions. The output
specification describes the space being predicted, the distribution's
factorization, and the procedure for producing a typed answer. A larger
encoder or a backbone shared with a generative model can implement the same
interface. A finite classifier is its first useful special case.

Conceptually, a query names an input state, a schema with interpretation `α`,
instructions about the desired answer, and a prediction policy:

```text
predict(model, observation, QuerySpec α)
    -> Prediction α

Prediction α records:
    the represented distribution or score semantics
    a decoded value, abstention, or explicit failure
    the schema, calibration, and numerical identities
```

The same state can support several questions, with independent questions
evaluated in parallel and dependent questions following a declared graph.
The interface supports novel language-described questions and schemas within
its implemented schema universe. A new question should not inherently require
a new fixed classification layer or a separately trained model.

### Compositional output schemas

| Schema form | Prediction mechanism | Required meaning |
|---|---|---|
| Boolean, enum, or candidate reference | Categorical head | Probability mass over the declared alternatives and a specified selection rule. |
| Ordinal or bounded numeric score | Ordinal distribution, discretized numeric head, or explicitly specified regression head | Range, precision, uncertainty interpretation, and decoding rule; an expected score need not itself be an enum value. |
| Record or product | Several field heads, sharing the state representation | Independent marginal predictions or an explicit joint factorization; marginal calibration does not establish joint calibration. |
| Inductive sum with fields | Constructor head followed by the selected constructor's argument heads | Inactive branches contribute neither fields nor training loss. |
| Bounded list or recursive value | Length/shape prediction and typed field expansion with explicit fuel | Termination, size bounds, and the order of conditional decisions. |
| Indexed or proof-bearing value | Typed construction plus a checker or evidence-producing procedure | Satisfaction of the actual index/predicate, with rejection if evidence cannot be established. |

Choose a schema-directed graph of heads as the next extension after finite
choice. Compile each supported schema into constructor, field, and reference
decisions. Evaluate conditionally independent fields in parallel; condition
dependent fields on their declared predecessors. This avoids enumerating the
Cartesian product of record fields. It also avoids claiming that independently
predicting each field recovers correlations between them.

Recursive structures need bounded expansion, and arbitrary Lean types are not
automatically executable output schemas. Initial schemas are a supported
universe of data types with checked Ixon encoders/decoders. Refinements and
proof obligations require additional evidence. The result can be assembled as
typed data and serialized to Ixon directly, without first generating source
text. Generating an unrestricted proof or novel source program remains a
possible task for a generative head using the same model and harness.

For example, a Lean task can ask jointly for promising premise references,
estimated branch success, a tactic-family choice, and structured arguments.
The harness can use the scores directly in its search policy or dispatch an
admitted action. Those question schemas, answers, and downstream verification
results become training records. This is substantially broader than selecting
one of several complete tool calls.

Training uses proper scoring objectives where appropriate: categorical or
joint negative log likelihood, Brier loss, and declared losses for numeric
predictions. Missing fields and inactive branches are masked explicitly.
Learning from interaction can optimize the quality of these predictions and
the workflows that consume them. Calibration remains a measured property on a
specified task distribution, not a consequence of Lean typing or a portable
arithmetic theorem.

### Encoder pilot: finite-choice head

The SystemOne-44M choice configuration ranks a finite, nonempty list of fully
constructed values of a specified Lean type. For a state-dependent tool protocol, these
can be values of `Action state`, after checking the relevant context and
admission conditions. Conceptually:

```text
candidates : Fin n -> CheckedValue environment expectedType
scores     : Observation -> candidates -> Vector Scalar n
choose     : scores -> Fin n
result     = candidates (choose scores)
```

Boolean decisions are two candidates; an ordinal score can be an inductive
with fixed bins. A distribution or expected numeric score is additional
output, not a proof that the chosen label is true. Types with unbounded fields
such as `Nat`, `String`, or expressions cannot be covered by a finite
constructor classifier. A constructor tag alone also omits its arguments.
For those tasks, supply complete candidate values, extend the schema-directed
heads, use a bounded recursive decision protocol, or invoke a generative head.
Proof-bearing arguments
must actually pass checking before the action can be admitted.

Use one shared bidirectional encoder for observation and candidate descriptors.
Insert a pooling token at the beginning of each sequence and take its final
normalized vector. Encode the observation once and each candidate separately:

```text
h  = encode(observation)
ci = encode(descriptor(candidate_i))
zi = dot(h Wscore, ci) / sqrt(d)
p  = softmax(z / temperature)
```

`Wscore` is one `d × d` matrix without bias. Candidate descriptors contain
readable type/schema information, constructor name, arguments, and relevant
preconditions. Candidate embeddings can be cached under the complete model,
tokenizer, and descriptor identity. This makes the first classifier usable for
different finite candidate sets without a new output matrix for every schema.

Limit this first choice profile to 64 candidates, 2,048 observation tokens, and 256
tokens per descriptor. Canonical candidate IDs determine order and ties;
container insertion order cannot affect the answer. Duplicates are resolved
before inference under a specified identity policy. Exceeding a limit returns
a typed error or invokes an explicitly defined hierarchical policy. It never
silently discards choices. Abstention or escalation is an ordinary candidate
when the task allows it.

This architecture trades cross-attention between observation and candidate
tokens for reuse of candidate encodings. Measure that tradeoff against a joint
encoder before increasing size. Start with supervised cross-entropy on checked
episodes, including rejected alternatives where labels are available. Distill
successful decisions from larger prover rollouts. Fit a positive temperature
on a held-out calibration set; report calibration and decision quality
separately from type validity. The finite softmax vector approximates a
probability distribution; exact sampling normalization is specified separately.

## 5. A generative prover and autoformalizer

After the tiny language-model and harness experiments, use Prover-1B for
the first billion-parameter dense pilot. Its observation contains the
task, available premises, local context, goal state, prior tool responses, and
remaining budget, projected by the shared `HarnessSpec` and `ViewSpec`.
Retrieval is part of this protocol, with an addressed result set and explicit
ordering. Long proof projects require retrieval and successive tool calls;
they do not depend on fitting the entire project into the context window.

The causal decoder proposes one typed action envelope at a time. Its payload
may contain source, a tactic script, an explicit term, or a reference to a
candidate artifact. Use a versioned grammar mask for the envelope where useful.
Bound parsing and generation, and record malformed proposals as failed
attempts. Semantic admission still resolves references, checks the expected
type, and validates the action against the current harness state.

Train in three steps using the same runtime projections:

1. **Representation learning.** Next-token training on paired named kernel
   terms, preserved source/tactics, and explanatory text, with explicit task
   tags and provenance. Splits group presentations of the same artifact before
   sampling. Auxiliary term prediction teaches logic without making every
   policy response emit a long kernel term.
2. **Supervised interaction.** Learn proposal spans from checked episodes.
   The default policy loss masks observations and tool replies. Auxiliary
   response or proof-term losses have separate weights in the training plan.
   Keep the actual proposal bytes and tokens, including failed attempts.
3. **Learning from verified interaction.** Begin with a specified episodic
   policy-gradient estimator with an action-independent baseline, before more
   complicated policy updates. Pin behavior weights and sampling distribution;
   use fresh rollouts for each initial update. Rewards use fixed-target proof
   checking, declared costs, and budget rules from `HarnessSpec`. A baseline
   estimated from the same trajectories needs its own bias analysis.

The initial real-valued objective, finite sampling distribution, executed
backward rule, and any approximation between them need separate statements.
Rounded probabilities or integer sampling bins do not inherit an exact
unbiased-gradient theorem for an ideal real softmax. An imported model can
bootstrap data, and an imported checkpoint can be fine-tuned, but its earlier
training remains outside the certificate for that fine-tuning run.

Use SGD through the initial MNIST experiments and introduce AdamW for the
tiny Transformer: first and second moments, bias correction, decoupled weight
decay, and global gradient-norm clipping. Select
initial beta values 0.9 and 0.95, epsilon `1e-8`, and clipping threshold 1.
Learning rate, warmup, decay schedule, weight-decay coefficient, loss mixture,
batch schedule, initialization, and update count belong to an explicit run
plan. They should be tuned in the pilot rather than inferred from parameter
count. Update moments once per logical batch. Data-parallel workers cannot
silently turn one planned update into several updates.

The release criterion includes valid-action rate, fixed-target proof success,
autoformalization statement alignment, and cost under equal task budgets.
Stronger proof search does not by itself establish better informal-to-formal
translation. Prover-7B is the next dense scaling experiment if corpus quality,
learning curves, backend throughput, and certificate costs justify it.

## 6. General models and sparse experts

Keep dense causal models as a supported family at larger widths and depths.
Add sparse experts as an explicit graph operator for the General-MoE profile.
Each block retains the same attention and replaces its single SwiGLU with
128 SwiGLU experts of width 2,048, using a learned `d × 128` router.

For each token independently:

1. Compute all router logits in the selected portable profile.
2. Select the four highest logits, breaking equal scores by expert index.
3. Compute normalized weights over those four selected logits.
4. Evaluate all four experts, then combine their weighted outputs using
   canonical expert-index order and the specified reduction tree.

Use dropless routing. Capacity quotas, worker availability, and other tokens
in a serving batch cannot change which experts evaluate a token. Scheduling
may queue work or report an operational failure, but cannot substitute experts
or return a different successful answer. Training may include a separately
specified load-balancing loss over a fixed logical batch. Hard top-k routing
is a discrete operation; its backward rule differentiates selected branches
or names a surrogate explicitly, with no claim of a global smooth derivative.

The first general profile is text/code plus tools. Natural language and general
code broaden the corpus beyond formal Lean data; formal tasks and interaction
traces remain part of the mixture. Other modalities can enter through typed
encoders/projectors and versioned observations later. There is no mathematical
reason the certificate interfaces must stop at formalization models.

There is also no reason to treat Ix as an oracle for arbitrary real-world
claims: general tools supply observations with their own evidence contracts.
Training and execution can be certified independently of whether a generated
answer is true. Broad capability and frontier competitiveness are empirical
goals requiring suitable data, research, evaluation, and resources.

Design distributed execution now, implement it after the dense pilot. The
graph identifies logical tensor axes and global reduction indices; a separate
placement plan assigns data, tensor, pipeline, and expert shards to devices.
That plan must preserve the graph's bit-level semantics. Network arrival order
cannot determine accumulation order. A changed world size requires a supported
placement proof, not a promise that ordinary all-reduce will agree.

## 7. Lean implementation and proof structure

Define the model in a shape-indexed graph with symbolic sequence/batch lengths,
named parameter tensors, explicit sharing, and a finite primitive vocabulary.
Keep its mathematical interpretation, executable finite interpretation,
reverse-mode transformation, and storage/placement interpretation separate.
Parameterize reusable theorems by dimensions and primitive contracts so larger
configurations instantiate proofs rather than require a proof per weight.

The Transformer primitive set is embedding/gather, linear maps, pointwise
arithmetic, canonical reductions, masking, normalization, positional rotation, softmax,
SwiGLU, and structured selection. Add sparse dispatch after the dense graph is
working. The harness uses these graphs through a policy interface with both
decision and autoregressive implementations.

The selected reuse strategy is:

| Layer | Initial choice | Evidence still needed |
|---|---|---|
| Numerical semantics | Lean 4.33.1's pure `Float32.Model` supplies the first SGD bit model; evaluate FloatLib for further numerical proofs and elementary functions. | Complete `P32` error/reduction rules, general refinement, and the exact chosen transcendental programs. |
| Graph and differentiation | Own a small Cybernet.ix graph; adapt proven graph/VJP patterns from TorchLean and lean4-mlir where their statements fit. | Shape and sharing laws, reverse-mode composition, and numerical interpretations for our precise operator set. |
| Reference execution | Pure Lean bit-level evaluator, followed by a C/Rust CPU implementation behind a refinement contract. | Equality to reference semantics and a concrete computation-checking path. |
| Acceleration | PTX as the first GPU target to investigate, using PTXLean semantics for an admitted fragment; MLIR can be an additional producer. | Exact arithmetic, memory, synchronization, reduction, and target/runtime refinement. |
| Compilation | Addressed kernel artifacts and checked transformations following Compilatr.ix's compiler contracts. | A new tensor/kernel bridge, artifact-specific preservation, and evidence for each downstream stage. |
| Interaction and artifacts | New Cybernet.ix harness and policy/training interfaces; Ix for checking and certificates. | Shared episode semantics, source projection, run composition, and consumer verification. |

[FloatLib](https://github.com/lean-dojo/FloatLib) provides executable bit-level
semantics and associated proofs. Its general approximation programs must be
distinguished from correctly rounded elementary-function results with specific
conditions. [TorchLean's trust-boundary documentation](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md)
distinguishes real AD theorems from concrete backend behavior; its same-device
deterministic CUDA option is weaker than our portability requirement.
[lean4-mlir](https://github.com/brettkoonce/lean4-mlir) is relevant to Lean-owned
training graphs and VJP proofs, while its lowering/runtime boundary needs
separate evidence. None supplies an automatic certificate for our whole stack.

The [first SGD implementation](certified-sgd.md) uses Lean core's bit model
and Mathlib v4.33.1 for calculus. The other libraries above remain reuse
decisions at the interface level, not dependency installations.
The inspected FloatLib and PTXLean toolchains are Lean 4.34.0, local Ix is
4.33.1, and Compilatr.ix is 4.33.0. Resolving those pins and checking the actual
theorem dependencies is an M1/M2 task in the [roadmap](roadmap.md).
The project should not acquire multiple incompatible numerical authorities
merely by combining libraries. See the [source notes](research-notes.md).

The [compilation route](portable-inference.md) connects the finite graph to a
checked kernel plan, PTX semantics, and a device execution contract. PTXLean is
a target-semantics candidate; Compilatr.ix's existing compiler supplies a
preservation and artifact-checking pattern. Neither is assumed to contain a
ready-made certified Transformer backend.

## 8. Resources and scaling gates

Portable binary32 is the initial reference profile. Its memory cost makes the
scaling challenge visible. Decimal GB below count parameter arrays only:

| Profile | FP32 weights, 4 bytes/parameter | FP32 weights + gradients + two Adam moments, 16 bytes/parameter |
|---|---:|---:|
| SystemOne-44M, choice baseline | 0.18 GB | 0.71 GB |
| Prover-1B | 4.52 GB | 18.09 GB |
| Prover-7B | 28.46 GB | 113.82 GB |
| General-MoE | 524.34 GB | 2,097.36 GB |

Activations, attention buffers, KV caches, temporary workspaces, replicas,
communication, data storage, and certificate witnesses are additional. Sparse
activation saves compute but does not remove inactive expert weights or their
optimizer state. A future BF16 storage profile halves weight storage; a mixed
precision trainer may still need FP32 master weights and moments. It is a
separate specified model profile with explicit casts and its own proofs.

For these GQA decoders, a binary32 KV cache alone costs
`2 × layers × tokens × KVHeads × headWidth × 4` bytes per sequence. Prover-1B
at 8,192 tokens therefore needs about 0.81 GB of KV storage. Long contexts and
concurrent sessions must be budgeted even when the weights fit.

Use MNIST and the tiny Transformer to measure accelerator execution and CPU
verification before the larger decision model. The 1B pilot needs an
accelerator memory budget beyond its 18 GB of training arrays, or explicit
sharding/recomputation. Full 7B training
and the MoE plan require distributed resources; adapter fine-tuning can reduce
trainable state but does not remove the base model's forward cost.

As a rough planning proxy, dense linear forward/backward work scales like
`6 × activeParameters × trainingTokens`. Attention, routing, optimizer work,
communication, proof search, and certificate generation add costs. This is
not a device-time estimate: strict portable kernels may be substantially
slower than conventional mixed-precision kernels.

Scale only after measuring the corresponding gate:

| Gate | Required evidence |
|---|---|
| Tiny complete example | Correct arithmetic, a training witness, portable inference, a harness turn, and independently checked Ix claims. |
| System One baseline and structured extension | Useful typed predictions and calibration, exact output agreement across CPU and GPU, measured certificate cost, then constructor/field heads with explicit joint semantics. |
| Dense prover | Improved fixed-target proof success per budget, sound corpus splits, reproducible action generation, and a viable accelerated training path. |
| Larger dense model | Learning curves justify additional data/compute; checkpoint and backend evidence remain practical. |
| Sparse general model | Stable training, exact routing across batch layouts, distributed refinement, broader evaluation, and a funded data/compute plan. |

The immediate research question is whether portable arithmetic and practical
execution evidence can support useful small models economically. The library
architecture should already express the larger models, while that experiment
determines how quickly we can build them.
