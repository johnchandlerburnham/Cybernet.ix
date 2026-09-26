# Certified SGD: first implementation

The first implementation trains a three-parameter affine regressor in Lean,
proves its minibatch gradient, and checks a complete eight-update run in the
Lean kernel. It also executes the same update graph with a pure binary32
model and proves that every parameter checkpoint in this particular run
agrees with exact arithmetic.

This is the initial training and arithmetic part of roadmap M2. Ix certificate
export, the shared interaction harness, and the full portable inference
profile remain separate implementation work.

## Run it

```sh
nix develop
lake build                    # Runtime library, proofs, and axiom audit
lake test                     # Replay/rejection, binary32, and FFI tests
lake exe cybernetix-sgd        # Print every loss checkpoint and final parameters
```

Lake resolves the locked Mathlib dependencies on first use; `lake update` is
the explicit dependency setup command. Outside the development shell:

```sh
nix run .#sgd
nix build .#proofs
nix flake check
```

The mathematical proofs import Mathlib at the same Lean 4.33.1 pin as Ix.
Executable modules use Lean core, without importing Mathlib. The
[development guide](development.md) describes the build and dependency cache.

## The experiment

The model is `prediction = w₀*x₀ + w₁*x₁ + bias`. Its target function is
`2*x₀ - 3*x₁ + 1`; the objective is the mean squared residual of a nonempty
minibatch. The concrete recipe lives in
[`Training/Example.lean`](../Cybernetix/Training/Example.lean).

| Input | Value |
|---|---|
| Initial parameters and step counter | `(0, 0, 0)`, step 0 |
| Batch A: `(x₀, x₁, target)` | `(-1, -1, 2)`, `(1, 1, 0)` |
| Batch B | `(-1, 1, -4)`, `(1, -1, 6)` |
| Ordered updates | A, B, A, B, A, B, A, B |
| Learning rate | `1/8` at every update |
| Optimizer | Plain minibatch SGD, without momentum |
| Evaluation set | All four examples |

The kernel checks the literal final state:

```text
cybernetix.exact-rational-sgd.v1
steps=8
w0=15/8
w1=-45/16
bias=58975/65536
```

Full-dataset MSE starts at `14` and ends at `261150529/4294967296`, which is
less than `1/16`. Prediction at `(3, -2)` is `796255/65536`. These are concrete
theorems about this recipe, including the initial state and example order.
They do not assert generalization quality or convergence of arbitrary SGD.

An SGD run is deterministic when conditioned on its ordered minibatches.
Here that schedule is an explicit list, rather than sampled internally.
There is no probabilistic assumption or unbiased-estimator theorem yet.

## One update definition, three interpretations

[`Training/Linear.lean`](../Cybernetix/Training/Linear.lean) defines parameters,
samples, nonempty batches, prediction, loss, gradient, and the SGD update using
only arithmetic operations. Instantiating those definitions gives:

| Interpretation | Role |
|---|---|
| `Rat` | Executable exact reference and concrete run checker |
| Mathlib `ℝ` | Differentiation and loss-change theorems |
| Lean `Float32.Model` | Executable bit-level rounded reference |

The tests also instantiate the update with native `Float32`, to compare a
compiled backend against the reference. There is no independent handwritten
native gradient formula to keep synchronized.

For batch size `n`, residual `rᵢ`, and augmented input `zᵢ = (x₀ᵢ, x₁ᵢ, 1)`:

```text
L(p)    = (Σᵢ rᵢ²) / n
g(p)    = (Σᵢ 2*rᵢ*zᵢ) / n
p_next  = p - η*g(p)
E(v)    = (Σᵢ (v · zᵢ)²) / n
```

The real proof establishes, for every direction `v`:

```text
L(p + t*v) = L(p) + t*(g(p) · v) + t²*E(v)
d/dt L(p + t*v) at t=0 = g(p) · v
L(p_next) = L(p) - η*(g(p) · g(p)) + η²*E(g(p))
```

Consequently `L(p_next) ≤ L(p)` when `η ≥ 0` and
`η*E(g(p)) ≤ g(p) · g(p)`. This is a sufficient step-size condition for the
current batch. It does not promise improvement on another batch or the full
dataset. `sgdStep_cast` and `run_cast` connect the executable rational update
and run to these same definitions over the reals.

The batch type contains a first example and a list of remaining examples,
so an empty mean cannot be constructed. Evaluation order is explicit:
prediction adds the two products before the bias, batch sums associate to
the right with a final zero, each gradient component is divided by the batch
size, and rate multiplication precedes parameter subtraction.

## Run checking and checkpointing

[`Training/SGD.lean`](../Cybernetix/Training/SGD.lean) defines the exact state,
update schedule, executable fold, and inductive `Runs` relation. A state holds
the parameters and update count. The remaining schedule is an explicit input
when resuming; it is not inferred from the counter.

`checkRun initial expectedSchedule claimedFinal` replays the computation and
compares the entire final state. Its theorem states that acceptance is
equivalent to `Runs initial expectedSchedule claimedFinal`. The consumer
supplies the expected initialization and schedule. This checker does not
authenticate a producer-supplied recipe or prove which physical process
produced the claimed weights.

The run laws prove deterministic results, exact update counts, and agreement
between a single run and checkpoint/resume at any schedule split. Equal
accepted final states have identical versioned UTF-8 encodings. This encoding
is a small reference format for state; an addressed experiment manifest,
decoder, and Ix claim bundle have not been implemented.

## What the binary32 experiment establishes

The pinned Lean toolchain already contains pure bit-level arithmetic in
`Init.Data.Float.Model.Float32`. It represents a valid 32-bit encoding and
implements operations through unpacking, arithmetic, rounding, and packing.
`Cybernetix.Numeric.Binary32` calls this model directly. Ordinary native
`Float32` operations are replaced during compilation and have a separate
execution boundary.

This allows the first finite reference without a toolchain upgrade or a new
floating-point dependency. FloatLib remains a candidate for additional
numerical proofs and elementary functions; it is not a dependency of this
implementation.

The experimental profile is
`cybernetix.binary32-sgd-toy.v1.lean-4.33.1`. The toolchain pin is part of its
definition. Rational inputs convert by rounding the numerator and denominator
separately, then performing binary32 division. This is an explicit conversion
program, not a general correctly rounded rational conversion. The toy inputs
are small integers and dyadic rationals.

The kernel proves that the finite parameter values at all nine checkpoints
(initialization and eight updates) equal their exact rational counterparts.
It separately proves these final words and their little-endian byte encoding:

```text
w₀       0x3ff00000
w₁       0xc0340000
bias     0x3f665f00
bytes    00 00 f0 3f  00 00 34 c0  00 5f 66 3f
```

The finite-to-rational comparison identifies both signs of zero with rational
zero. Bit patterns remain distinct in word/byte comparisons and primitive
fixtures. NaNs and infinities have no rational value.

Nine additional primitive fixtures are kernel-checked: both ties-to-even
directions, subnormal addition, crossing into the normal range, underflow to
each sign of zero, overflow, rounded division, and cancellation. Another
theorem proves that reassociating the binary32 sum of `2^24`, `1`, and `-2^24`
changes the result from zero to one. Reduction order is part of the model.

The compiled tests compare native `Float32` against the bit model after every
toy update and on every primitive fixture. Passing on the current x86-64 Linux
host is conformance evidence for those cases. A proof about `Float32.Model`
does not certify the native compiler, runtime, FFI, or another CPU/GPU.

This profile deliberately has a narrower scope than
[`P32`](portable-inference.md): its sums associate to the right, special
values propagate, and it has no tensor graph or elementary-function programs.
The balanced reductions, finite-input validation, deterministic error policy,
and accelerated implementation required by `P32` remain to be built.

## Proof coverage and trust boundary

| Source | Main theorem roots | Claim |
|---|---|---|
| [Gradient](../CybernetixProofs/Gradient.lean) | `meanGradient_correct`, `sgdStep_loss`, `sgdStep_descent` | Actual mean-batch derivative, exact loss change, conditional descent over reals |
| [Exact](../CybernetixProofs/Exact.lean) | `sgdStep_cast`, `run_cast` | Rational execution agrees with the real update program |
| [SGD](../Cybernetix/Training/SGD.lean) | `checkRun_iff`, `run_append`, `run_steps`, `Runs.encoded_result_eq` | Run acceptance, resume, update count, identical encoded results |
| [Exact](../CybernetixProofs/Exact.lean) | `example_valid`, loss and prediction theorems | Concrete final state, evaluation loss, and inference |
| [Binary32](../CybernetixProofs/Binary32.lean) | `toy_prefixes_exact`, `toy_final_words`, `toy_final_bytes`, `cases_correct`, `reassociation_changes_answer` | Concrete rounded run, encoding, and arithmetic fixtures |

Concrete proofs use kernel reduction via `decide +kernel`. The
[axiom audit](../CybernetixProofs/Audit.lean) traverses the dependencies of
19 selected theorem roots and fails the build if any axiom occurs outside
`propext`, `Classical.choice`, and `Quot.sound`. It excludes admitted proofs
and native-evaluation axioms. The audit runs as part of the default Lake
build and the Nix proof check.

These are Lean theorem artifacts. They are not yet exported Ix certificates,
zero-knowledge proofs, or an independent certificate consumer. The logical
run-checker theorem also does not certify its compiled machine-code execution.
For this tiny recipe, a concrete kernel-checked proof establishes the result
without relying on that compiled checker.

## Next implementation boundaries

1. Export the checked recipe/run claim and its dependencies through Ix, with
   admission against consumer-selected inputs and an explicit axiom policy.
2. Implement finite validation, deterministic errors, and canonical reductions
   for a versioned `P32` subset. Preserve this toy profile as a distinct fixture.
3. Connect inference and training examples to the new shared harness and its
   episode projection; then add tensor shapes and a coarse Rust numerical API.
4. Measure execution and certificate costs before the MNIST pipeline and
   finite softmax/cross-entropy programs enlarge the computation.

The [roadmap](roadmap.md) retains the complete M1/M2 acceptance gates. This
implementation establishes the first mathematical and concrete training
claims needed by those gates.
