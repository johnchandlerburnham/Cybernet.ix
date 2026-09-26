# v2: A Deterministic, Verified Training Pipeline for a Competitive 120B-Class Model

*Design notes, September 2026. Builds on v1, "A Maximally Verified Lean 4 Pipeline for Training a GPT-2-Class Model." The v1 design remains the right choice below roughly 1B parameters; this document covers everything above that.*

## Summary

v1 made training bit-exact by fixing the order of every floating-point reduction and giving up tensor cores. At GPT-2 scale that cost a few extra GPU-days. At 120B scale it would cost months to years of cluster time, so v2 changes the arithmetic contract instead of paying that price.

The core change is that every matrix multiply uses int8 operands with **exact integer accumulation on tensor cores**. Integer addition is associative, so a matmul's result no longer depends on tile order, split-K, the tensor core's internal ordering, or the order in which GPUs combine partial sums. Three consequences drive the rest of the design:

- **Tensor cores come back.** An exact-integer run should cost roughly what a standard bf16 run costs, rather than 10–15× more.
- **Parallelism becomes a theorem.** "The sharded training step equals the single-device step" becomes an equation over the integers, provable once from associativity and commutativity. In floating point the same statement is false.
- **Determinism pays for itself at scale.** Exact integer checksums can verify every GEMM at runtime, bit-exact restarts make hardware failures harmless to the training trajectory, and bit-identical trainer and sampler numerics make reinforcement learning truly on-policy.

On size: roughly 120B total parameters is the right class for a competitive open model in 2026, but the model should be a **sparse mixture-of-experts with about 8–12B active parameters**, not a dense 120B (Section 1).

The largest risk in this plan is not verification. It is whether exact-int8 training matches bf16 quality at this scale; published evidence reaches only about 1.5B parameters. The plan therefore climbs a ladder of proving runs with explicit go/no-go gates before committing to the flagship (Section 11).

This pipeline proves that the training run did exactly what it claims. Whether the resulting model is competitive is an empirical question, decided mostly by data, and no proof addresses it.

---

## 1. Is 120B the right size?

### 1.1 The competitive set is sparse

The open models in this size class are mixtures of experts. OpenAI's gpt-oss-120b has 116.8B total parameters but activates about 5.1B per token, using 36 layers, 128 experts, and top-4 routing. Upstage's Solar Open has 102B total parameters with 12B active. A dense 120B model would compete against these on quality while spending roughly ten times more compute per training token, and would be much more expensive to serve.

### 1.2 Compute

| Run | Total / active params | Tokens | Training compute | H100-hours (approx.) | Example wall clock |
|---|---|---|---|---|---|
| Proving run A (dense) | 1B / 1B | 100B | 6×10²⁰ | ~330 | ~2 days on 8 GPUs |
| Proving run B (MoE) | ~20B / ~2B | 1T | 1.2×10²² | ~10k | ~3 days on 128 GPUs |
| Flagship MoE | ~120B / ~10B | 15T | 9×10²³ | ~0.7M | ~1 month on 1,024 GPUs |
| Flagship MoE, longer | ~120B / ~10B | 25T | 1.5×10²⁴ | ~1.2M | ~7 weeks on 1,024 GPUs |
| Dense alternative | 120B / 120B | 15T | 1.1×10²⁵ | ~6M | ~4 months on 2,048 GPUs |

All figures are rough planning estimates:

- Training compute is estimated as 6 × active parameters × tokens. This ignores attention, which understates long-context runs by roughly 10–50%.
- Dense exact-int8 throughput is taken as about 500 effective TOPS per H100. That is about a quarter of the ~2,000 TOPS int8 dense peak, a conservative allowance for the integer epilogues in Section 3.
- MoE throughput is taken as about 350 effective TOPS per H100. This is anchored on DeepSeek-V3, which trained 37B active parameters on 14.8T tokens in about 2.8M H800-hours, or roughly 330 effective TFLOPS per GPU.

Two things stand out. First, the MoE flagship costs roughly a tenth of the dense alternative. Second, the dense 120B run at a typical modern token budget lands at about 1.1×10²⁵ FLOPs, just over the EU AI Act's 10²⁵ threshold. Above that threshold a general-purpose model is presumed to carry systemic risk, and for open-weights models the Act's open-source exemptions from documentation duties fall away. That is the status as of August 2026 and is not legal advice. The MoE plan stays roughly an order of magnitude below the threshold.

### 1.3 Verification cost does not scale with size

The proofs are about the architecture, parameterized by its dimensions. The same theorems that cover the 1B proving run cover the 120B flagship. What does scale is:

- the risk that low-precision exact training behaves differently at scale, and
- the distributed-systems machinery: parallelism, fault tolerance, and post-training infrastructure.

So size should be chosen for budget and ML risk, and the proofs amortized across a ladder of runs.

### 1.4 Recommendation

Target about 120B total parameters as a sparse MoE with about 8–12B active, trained on 15–25T tokens, reached through proving runs at 1B dense and ~20B-total MoE. If a dense model is required, a dense model in the 8–30B range is a more sensible verified flagship than a dense 120B.

---

## 2. What changed from v1

| | v1 (GPT-2) | v2 (120B-class MoE) |
|---|---|---|
| Matmul arithmetic | fp32 on CUDA cores, fixed reduction order | int8 × int8 → exact int32/int64 on tensor cores |
| Source of bit-exactness | Freezing every reduction order | Exact integer accumulation; order is irrelevant |
| Other reductions | Fixed order | Exact integer sums, max/min, or all-gather plus canonical local sum |
| Elementwise operations | fp32, explicit rounding | Same |
| Executable spec | Spec F (binary32, via FloatLib) | Spec ℤ (integers and fixed point) plus a small fp32 elementwise surface |
| Gradient theorem | Backward pass equals the true gradient | True gradient except at explicit quantizers and routing decisions |
| Hardware | One GPU | 1,000+ GPUs of the H100/H200/B200 class |
| Parallelism | Avoided | Proved equivalent to single-device execution |
| Kernel assurance | Proofs (research-grade) or bit-exact testing | Exact runtime checksums on every GEMM; proofs added progressively |
| Post-training | Out of scope | SFT and RL with a bit-identical sampler |

Unchanged from v1:

- the two-level structure: a real-valued spec for gradient correctness and an executable spec that defines the bits
- the graph IR and its denotation proofs
- the tokenizer and data-order proofs
- the static memory plan and the checkpoint round-trip proofs
- the hash-chained training transcript
- most of the trusted computing base

---

## 3. The arithmetic contract

### 3.1 The rule

Every reduction in the training step must be one of three kinds:

1. an exact integer sum, with a proved bound showing it cannot overflow;
2. a maximum or minimum, which is associative and commutative in any arithmetic;
3. a small reduction performed by all-gathering the partial values and summing them locally in a canonical order.

Elementwise operations may use IEEE fp32 with explicit rounding modes. They contain no reduction, so they are already bit-reproducible under any parallel schedule. A floating-point reduction whose order depends on scheduling anywhere in the step is a bug in the spec.

### 3.2 Matrix multiplies

Operands are int8 with power-of-two scales, and products accumulate exactly in int32 on tensor cores. Each int8 product has magnitude at most 2¹⁴, so an int32 accumulator cannot overflow for inner dimensions below 2¹⁷ (131,072), which covers every hidden size in this design.

Where the scales live determines whether the result stays a single exact sum:

- **Scales on the outer dimensions** (per row of the left operand, per column of the right) factor out of the sum. Each output is one exact integer sum times a power of two.
- **Scales along the inner dimension** (per block of the contraction dimension, as in Jetfire or DeepSeek-V3's 1×128 activation tiles) give each block its own exponent. Each block's int32 partial sum is shifted to a common exponent and added into an int64 accumulator. This is exact as long as the spread of block exponents is bounded, which the kernel checks.

The second pattern has the same structure as DeepSeek-V3's FP8 kernels, which periodically copy partial sums out of the tensor cores into FP32 registers on the CUDA cores. The difference is that here the second-level accumulator is an exact integer.

Each linear layer runs three GEMMs: forward, input gradient, and weight gradient. Each contracts over a different dimension, so a tensor scaled per row for one GEMM has its scales along the contraction dimension in another. The spec therefore quantizes each tensor per use, in the layout that GEMM needs, and each of those quantizations is an explicit function in the spec.

**Why not FP8?** Hopper's FP8 tensor cores accumulate with limited precision. DeepSeek reports that products are aligned by the maximum exponent and only the top 13–14 bits are kept; the rest are truncated. That behavior was characterized by users rather than specified by NVIDIA, which rules FP8 accumulation out of a bit-exact contract. Integer tensor-core accumulation is exact by construction.

### 3.3 When int8 is not precise enough: limb splitting

Some matmuls may need more than 8 bits of operand precision. The likely candidates are attention, the output head, and the router. Instead of falling back to floating point, split the operand into 8-bit limbs:

- A 16-bit value is `hi · 2⁸ + lo`, so its product with an int8 matrix is two exact 8-bit GEMMs combined with a shift.
- Two 16-bit operands take four GEMMs.

This is the same idea used to emulate FP64 matrix multiplication on integer tensor cores (the Ozaki scheme). It keeps the contract exact at 2–4× the cost of the affected GEMMs. It is the v2 counterpart of the block-level precision fallback that made INT8 training work for a 1.5B Llama model.

### 3.4 Requantization and rounding

After each GEMM, results are shifted and rounded back to int8 for the next operation. Rounding must be deterministic and explicit. There are two good options:

- NITI's pseudo-stochastic rounding, which uses the discarded low-order bits as its source of randomness;
- stochastic rounding driven by a counter-based hash of (step, tensor, element index).

Both are pure functions and go directly into the spec.

### 3.5 Operations kept wider

DeepSeek-V3 kept the embedding, output head, MoE gating, normalization, and attention in higher precision than its FP8 GEMMs. v2 follows the same instinct inside the integer contract:

| Operation | Arithmetic in v2 |
|---|---|
| Linear layers (attention projections, expert FFNs) | int8 × int8, exact int32/int64 |
| Embedding lookup and its gradient | int32 fixed point; the gradient scatter-add is an exact integer sum |
| Router | 16-bit via limb splitting; exact logits |
| Output head | 16-bit via limb splitting |
| Attention scores and weighted sum | int8, 16-bit limbs where needed; softmax in base-2 fixed point |
| RMSNorm | int32/int64 fixed point with an integer square root |
| SwiGLU gating, residual adds, applying RoPE | fp32 elementwise, or fixed point |
| Loss (softmax cross-entropy) | Fixed point, max-subtracted, base 2 |
| AdamW update | fp32 elementwise, sharded |
| Global gradient norm | Per-shard partial sums, all-gathered, summed in canonical order |

These are the starting point for the proving-run ablations, not conclusions.

### 3.6 Attention and softmax

Softmax is unchanged when a constant is subtracted from all its inputs. So v2 computes it in base 2 (FlashAttention implementations already fold log₂e into the score scale) and rounds each running maximum to an integer. Every online-softmax rescaling factor then becomes an exact power of two. Tiled attention within a GPU and ring attention across GPUs combine their partial results exactly, in any order.

Attention is a large share of compute in a sparse model. Its cost scales with context length and model width, not with the number of active parameters. For a model of this shape with about 10B active parameters, attention is on the order of 10% of training compute at 8k context and approaches half at 32k. Attention therefore has to run on tensor cores too.

---

## 4. The model and its specifications

### 4.1 Architecture (illustrative)

The model is a standard modern sparse decoder:

- pre-norm RMSNorm
- grouped-query attention with RoPE, with context extension added late in training
- SwiGLU experts with top-k routing, optionally plus shared experts
- untied embeddings and no biases

gpt-oss-120b is a useful reference point: 36 layers, residual width 2,880, 128 experts, and top-4 routing. The exact dimensions should come from scaling ablations at the proving-run sizes.

### 4.2 Spec ℝ, with explicit quantizers

The real-valued spec is the model with every quantizer written in as an explicit function Q. Every operation other than Q gets a backward-pass proof against Mathlib's derivative, as in v1. RoPE, RMSNorm, SwiGLU, attention, and GQA are all smooth.

Each quantizer's backward pass is a straight-through estimator: the gradient passes through as if Q were the identity. That is a definition, not a derivative, and the spec says so explicitly. The top-level gradient theorem therefore has this shape (a sketch, not working code):

```lean
-- surrogateGrad: exact VJPs for every operation, the identity VJP at each
-- quantizer, and the gradient of the selected branch at each router.
theorem backwardR_eq_surrogate (θ : Params) (b : Batch) :
    backwardR θ b = surrogateGrad lossR θ b
```

Because the proofs quantify over dimensions, one set of theorems covers every run on the ladder.

### 4.3 Spec ℤ

The executable spec is now over integers and fixed point, with a small fp32 surface for elementwise operations. Most of it lives in Lean's integer and bit-vector types, where automation such as `omega` and `bv_decide` does much of the proof work. FloatLib is needed only for the fp32 elementwise operations and their error analysis.

Overflow bounds are theorems:

```lean
theorem gemm_int32_exact (A : Mat Int8 m k) (B : Mat Int8 k n) (hk : k < 2^17) :
    ∀ i j, |∑ l, (A i l : ℤ) * (B l j : ℤ)| < 2^31
```

### 4.4 Routing

Router logits are exact integers, so top-k selection is a deterministic function of the input. Two consequences need care.

**Ties are real.** With integer logits, two experts can tie with nonzero probability. The spec breaks ties by lowest expert index, and the gradient theorem is stated for the selected branch. This parallels how lean4-mlir handles ReLU kinks with conditional theorems, except that here the branch choice is fully specified rather than assumed away.

**No order-dependent drops.** Routing is dropless. If a capacity limit is used instead, overflow tokens are dropped by canonical token index, never by the order in which tokens arrive over the network.

Load balancing uses either an auxiliary loss, which is differentiable and covered by the gradient proofs, or DeepSeek-V3-style bias adjustment, which updates router biases outside the gradient. Either way the update is a pure function in the spec.

Expert-parallel all-to-all only moves data. Each token's expert outputs are combined in canonical expert order.

---

## 5. Parallelism as a theorem

With exact integer reductions, splitting a computation across GPUs only changes which integers get added where. "The sharded step equals the single-device step" becomes an equation over ℤ, provable once from associativity and commutativity, for every supported layout:

```lean
theorem sharded_eq (L : Layout) (s : State) (b : Batch) :
    unshard L (stepSharded L (shard L s) b) = stepZ s b
```

How each form of parallelism fits:

- **Data parallel and FSDP/ZeRO.** Weight gradients are integer sums over tokens. The ranks agree on a common exponent per tensor with a max all-reduce, which is exact. Each rank shifts its partial sum to that exponent, then they reduce-scatter in int64. The shift discards low bits deterministically, and the spec defines exactly how. The optimizer update on each shard is elementwise, so sharding it changes nothing.
- **Tensor parallel.** Column-parallel layers need no reduction. Row-parallel layers all-reduce exact integer partial sums.
- **Pipeline parallel.** No arithmetic changes. Gradient accumulation across microbatches is an exact integer sum.
- **Expert parallel.** All-to-all is a permutation.
- **Context parallel.** Ring attention combines partial results with exact power-of-two rescaling (Section 3.6).

NCCL remains trusted code, but its reduction order no longer affects the result. This enables a cheap and powerful test: run the same steps under two different parallel layouts and require bit-identical results.

### 5.1 Batch invariance comes for free

Thinking Machines showed that most LLM inference nondeterminism comes from kernels whose per-sequence results depend on batch size, and built batch-invariant kernels to fix it. In v2, batch invariance is structural as long as one rule holds: **no activation scale may depend on other rows of the batch.** With per-token activation scales and exact accumulation, each token's result is a function of its own row alone.

```lean
theorem batch_invariant (θ : Params) (xs : Batch) (i : Fin xs.size) :
    (forwardZ θ xs)[i] = forwardZ θ (singleton xs[i])
```

Per-tensor activation scales would break this property, so they are excluded from the forward pass.

---

## 6. Kernels

### 6.1 The GEMM

The integer GEMM is the easiest kernel to reason about arithmetically and still hard to make fast. It needs int8 tensor-core instructions, a pipelined memory schedule, and the int64 epilogue for inner-dimension block scales. DeepSeek's DeepGEMM is encouraging evidence on size: its FP8 kernel, which uses the same two-level accumulation structure, is described as a single core kernel function of about 300 lines.

### 6.2 Exact checksums on every GEMM

Algorithm-based fault tolerance (ABFT) checks a matrix product with checksums: the column sums of C must equal the column sums of A multiplied by B.

- In floating point this check needs a tolerance and misses small errors.
- In integer arithmetic it is an exact equality. Computed with wrapping int64 arithmetic, it stays exact modulo 2⁶⁴ with no overflow analysis needed.
- Using both an all-ones and a weighted checksum vector catches any single corrupted element and nearly all multi-element corruptions.
- Each check costs a matrix-vector product, which is negligible next to the GEMM.

Running this on every GEMM turns an unverified kernel into a checked one. Any kernel bug or hardware fault that corrupts an output element is caught in the step where it happens. At launch, before refinement proofs exist for the production kernels, this is the primary assurance for the large majority of the compute.

### 6.3 Other kernels

Attention, normalization, routing, and the optimizer are checked by sampled audits against Spec ℤ, as in v1, and by replay. Refinement proofs using a formal PTX semantics (PTXLean) remain the long-term goal. Start with the elementwise and row-reduction kernels.

### 6.4 Hardware targets

H100, H200, and B200 offer int8 and FP8 tensor-core throughput at a 1:1 ratio. Blackwell Ultra (B300) does not:

- its published INT8 throughput is roughly 1/30 of its FP8 throughput;
- the PTX ISA does not expose its new integer tensor-core instruction;
- NVIDIA's CUTLASS library skips generating INT8 kernels for it.

This design is a bet on hardware with first-class integer tensor cores, and NVIDIA's roadmap is moving away from that. Confirm integer tensor-core support on any target, including AMD parts, before procurement.

---

## 7. Failures, silent corruption, and replay

At this scale hardware failure is routine:

- Meta's Llama 3 405B run on 16K H100s saw 466 job interruptions in a 54-day window. 419 were unexpected, and about 78% of those were attributed to confirmed or suspected hardware issues.
- Silent data corruption (SDC) is the more dangerous case. Meta attributed six interruptions to it in that window, and Google has estimated an SDC event every week or two during Gemini training.

Determinism turns these from threats to the training trajectory into operational nuisances:

- **Bit-exact resume.** A restart from a checkpoint continues exactly the trajectory that would have happened without the failure. By Section 5, this holds even if the restart uses a different node count or parallel layout.
- **Exact GEMM checksums.** Compute corruption is caught in the step where it occurs.
- **Replay.** A small fraction of spare capacity continuously re-executes randomly chosen past steps from checkpoints and compares bits. This catches corruption in memory, the interconnect, and non-GEMM kernels.
- **Transcript.** Each checkpoint is hash-chained to the previous one, together with the hashes of every batch consumed, the spec and code versions, and the hardware used.

---

## 8. Data

Competitiveness depends mostly on data, and verification does not make data good. The proof boundary starts at a fixed, hashed corpus manifest. Everything upstream of it is outside the boundary: crawling, filtering, deduplication, quality classifiers, and synthetic data generation. Publish that code and its hashes so it can be audited, but don't claim it is verified.

Inside the boundary, v1's approach scales directly:

- **Tokenizer.** A byte-level BPE with a vocabulary around 200k (gpt-oss uses the o200k family), implemented in pure Lean with round-trip and merge-fidelity proofs. Tokenizing 15–25T tokens in Lean is a one-time, embarrassingly parallel job, but measure its throughput early. The pre-tokenizer regex again needs either a verified matcher or a simpler, fully specified replacement.
- **Mixture and order.** The data-mixture schedule and the batch order are pure functions of a seed, with proofs that the ordering is a bijection. Every batch hash goes into the transcript.
- **Decontamination.** Benchmark decontamination is a specified function. The pipeline can prove it was run as specified, but not that it was sufficient.

---

## 9. Post-training

Supervised fine-tuning reuses the pretraining pipeline unchanged.

Reinforcement learning needs a sampler, and here v2 has a structural advantage. Thinking Machines found that nondeterministic inference quietly turns on-policy RL into off-policy RL. With bit-identical sampler and trainer numerics, the KL divergence between them is exactly zero and training is stable. In v2 the sampler runs the same exact integer kernels as the trainer and is batch-invariant by construction. Sampler and trainer therefore agree bit for bit, however requests are batched.

The same property means the deployed model is exactly the trained model. The int8 weights and activation quantization used in the training forward pass are the serving format, so there is no separate post-training quantization step and no quantization gap.

Other sources of randomness and reward fit into the spec as follows:

- **Sampling randomness** comes from a counter-based generator keyed by (step, prompt, position).
- **Programmatic rewards** are specified functions. This covers unit tests, math answer checking, and Lean itself for formal mathematics.
- **Learned reward models** are models, trained by the same pipeline.
- **External tool environments** are logged and hashed, not verified.

---

## 10. What is proved, checked, and trusted

| Category | Items |
|---|---|
| Proved (in Lean) | Tokenizer round-trip and merge fidelity; data order; backward pass equals the specified surrogate gradient; integer overflow bounds; sharded step equals single-device step for every supported layout; batch invariance; memory plan; checkpoint round-trip; optimizer semantics |
| Checked at runtime | Every GEMM, via exact checksums; sampled audits of other kernels against Spec ℤ; replay of random past steps; bit-identity across parallel layouts and restarts |
| Trusted | Lean's kernel and its standard axioms; the specs themselves; the Lean and C compilers; ptxas and the CUDA driver; NCCL code (its ordering no longer matters); GPU hardware (mitigated by checksums and replay); everything upstream of the corpus manifest |
| Empirical | Training quality relative to bf16; competitiveness of the final model |

---

## 11. Build plan

| Phase | Run | Purpose | Gate to proceed |
|---|---|---|---|
| 0 | None | Spec ℝ and Spec ℤ; gradient, overflow, parallelism, and batch-invariance proofs, all parametric in dimensions | Proofs close using only the standard axioms |
| 1 | Two 1B dense runs, 100B tokens each | A/B comparison of exact-int8 against bf16 on identical data and order | Loss curves within a pre-registered tolerance; no divergence; limb-splitting rate measured |
| 2 | ~20B-total MoE (~2B active), 1T tokens, 64–128 GPUs | Routing, expert parallelism, all parallel layouts, checksums, replay | Bit-identical results across layouts and restarts; loss parity with a bf16 reference |
| 3 | Same scale as phase 2 | Long-context attention numerics; RL loop with the exact sampler | Attention ablations pass; sampler and trainer bit-identical during RL |
| 4 | ~120B-total MoE (~10B active), 15–25T tokens, ~1,000 GPUs | Flagship pretraining | Checksums and replay stay clean; loss tracks the scaling-law prediction from phases 1–2 |
| 5 | Post-training and release | SFT, RL, evaluation; publish specs, proofs, transcript, and weights | — |

Phase 1 is the most important experiment in the plan and one of the cheapest. Two runs of a few hundred H100-hours each provide the first real evidence of whether the arithmetic contract can train a good model at all.

---

## 12. Cost

The main-run compute from Section 1.2 is roughly 0.7–1.2M H100-hours for the MoE flagship, versus about 6M for a dense 120B. At an illustrative $2 per H100-hour, that is roughly $1.5–2.5M for the flagship run itself.

Budget several times that for the whole program: proving runs, ablations, restarts after failed experiments, long-context extension, post-training, evaluation, and data.

Overheads specific to v2, all to be measured in Phase 1:

- the int64 epilogue for inner-dimension block scales;
- limb splitting for the sensitive matmuls;
- GEMM checksums, expected to be small;
- replay capacity, likely a few percent of the fleet.

---

## 13. Risks and open problems

- **Exact-int8 training quality at scale.** Published evidence reaches about 1.5B parameters, and only with block-level fallback to higher precision. Nothing guarantees it holds at 120B total; the ladder exists to find out cheaply.
- **Activation outliers.** SwiGLU-style layers widen activation distributions early in training. That is what broke Jetfire's INT8 data flow on a 1.5B Llama. Limb splitting and per-token scales are the planned mitigations.
- **Integer attention in training.** This is less established than integer attention for inference.
- **Hardware.** Integer tensor cores are being deprioritized, as B300 shows.
- **Kernel proofs.** At launch, GEMMs are checked rather than proved. PTX-level refinement proofs remain research-grade.
- **Routing ties and discrete load balancing.** Both are fully specified, but they make the surrogate-gradient theorem less like a classical gradient statement.
- **Regulation.** A dense 120B at modern token budgets crosses the EU's 10²⁵ FLOP presumption threshold. The MoE plan does not, but check the rules in force at the time of training.

---

## References

All v1 references carry over (TorchLean, lean4-mlir, Hesper, FloatLib, PTXLean, Certigrad, SciLean, TensorLib, llm.c).

- OpenAI, gpt-oss-120b and gpt-oss-20b model card: https://arxiv.org/abs/2508.10925
- Solar Open technical report: https://arxiv.org/abs/2601.07022
- Wang et al., NITI: Training Integer Neural Networks Using Integer-only Arithmetic: https://arxiv.org/abs/2009.13108
- Xi et al., Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization: https://arxiv.org/abs/2403.12422
- Accurate INT8 Training Through Dynamic Block-Level Fallback: https://arxiv.org/abs/2503.08040
- Dettmers et al., LLM.int8(): https://arxiv.org/abs/2208.07339
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware: https://arxiv.org/abs/2505.09343
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8 Availability on NVIDIA Blackwell Ultra: https://arxiv.org/abs/2608.11693
- Llama Team, The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Understanding Silent Data Corruption in LLM Training: https://arxiv.org/abs/2502.12340
- He et al., Defeating Nondeterminism in LLM Inference: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- EU AI Act, high-level summary: https://artificialintelligenceact.eu/high-level-summary/
