import Cybernetix.Tensor.Buffer
import Cybernetix.Numeric.Reduction

/-! A shape-indexed affine classifier. The profile fixes row-major weights,
tree reductions, bias-after-reduction, and lowest-index argmax ties. This is
the inference subset; it makes no claim about a training run or a compiler.
-/

namespace Cybernetix.Model.LinearClassifier

open Tensor

def profile : String := "cybernetix.linear-f32-tree.v1.lean-4.33.1"

structure Parameters (inputs outputs : Nat) where
  weights : Buffer (inputs * outputs)
  bias : Buffer outputs

inductive InputKind where
  | image | weights | bias
  deriving Repr, DecidableEq

inductive Error where
  | nonFiniteInput (kind : InputKind) (index : Nat)
  | nonFiniteDot (output : Nat)
  | nonFiniteBiasAdd (output : Nat)
  deriving Repr, DecidableEq

@[specialize] def validateScalars (kind : InputKind) (get : Fin n → Float32) : Except Error Unit := do
  for i in [:n] do
    if h : i < n then
      unless (get ⟨i, h⟩).isFinite do
        throw (.nonFiniteInput kind i)

/-- Validation order is image, row-major weights, bias, then output index. -/
def validate (kind : InputKind) (buffer : Buffer n) : Except Error Unit :=
  if small : buffer.bytes.size < USize.size then
    validateScalars kind fun i => buffer.getFast i small
  else validateScalars kind buffer.get

def referenceDot (p : Parameters inputs outputs) (x : Buffer inputs)
    (j : Fin outputs) : Float32.Model :=
  Numeric.Reduction.reference inputs fun i =>
    x.view i * p.weights.view (Buffer.matrixIndex i j)

@[inline] def machineIndex (i : USize) (_hi : i.toNat < inputs) (j : Fin outputs)
    (columns : outputs < USize.size) (_size : inputs * outputs < USize.size) : USize :=
  i * USize.ofNatLT outputs columns + USize.ofNatLT j.val (by omega)

theorem machineIndex_value (i : USize) (hi : i.toNat < inputs) (j : Fin outputs)
    (columns : outputs < USize.size) (size : inputs * outputs < USize.size) :
    (machineIndex i hi j columns size).toNat = i.toNat * outputs + j.val := by
  have hindex := (Buffer.matrixIndex (⟨i.toNat, hi⟩ : Fin inputs) j).isLt
  change i.toNat * outputs + j.val < inputs * outputs at hindex
  simp only [machineIndex, USize.toNat_add, USize.toNat_mul, USize.toNat_ofNatLT]
  rw [Nat.mod_eq_of_lt (show i.toNat * outputs < USize.size by omega),
    Nat.mod_eq_of_lt (show i.toNat * outputs + j.val < USize.size by omega)]

@[inline] def machineLeaf (p : Parameters inputs outputs) (x : Buffer inputs) (j : Fin outputs)
    (hx : x.bytes.size < USize.size) (hw : p.weights.bytes.size < USize.size)
    (columns : outputs < USize.size) (i : USize) : Float32 :=
  have hinputs : inputs < USize.size := by have := x.size_eq; omega
  if h : i < USize.ofNatLT inputs hinputs then
    have hi : i.toNat < inputs := by simpa [USize.lt_iff_toNat_lt] using h
    have hweights : inputs * outputs < USize.size := by have := p.weights.size_eq; omega
    let index := machineIndex i hi j columns hweights
    have hindex : index.toNat < inputs * outputs := by
      rw [machineIndex_value]
      exact (Buffer.matrixIndex (⟨i.toNat, hi⟩ : Fin inputs) j).isLt
    x.getU i hi hx * p.weights.getU index hindex hw
  else .ofBits 0

@[inline] def nativeDot (p : Parameters inputs outputs) (x : Buffer inputs)
    (j : Fin outputs) : Float32 :=
  if hx : x.bytes.size < USize.size then
    if hw : p.weights.bytes.size < USize.size then
      if columns : outputs < USize.size then
        if span : 2 ^ Numeric.Reduction.depth inputs < USize.size then
          Numeric.Reduction.nativeTreeU (machineLeaf p x j hx hw columns)
            (Numeric.Reduction.depth inputs) 0
            (USize.ofNatLT (2 ^ Numeric.Reduction.depth inputs) span)
        else Numeric.Reduction.native inputs fun i => x.get i * p.weights.matrixGet i j
      else Numeric.Reduction.native inputs fun i => x.get i * p.weights.matrixGet i j
    else Numeric.Reduction.native inputs fun i => x.get i * p.weights.matrixGet i j
  else Numeric.Reduction.native inputs fun i => x.get i * p.weights.matrixGet i j

def referenceLogit (p : Parameters inputs outputs) (x : Buffer inputs)
    (j : Fin outputs) : Except Error Float32.Model :=
  let dot := referenceDot p x j
  if dot.isFinite then
    let result := dot + p.bias.view j
    if result.isFinite then .ok result else .error (.nonFiniteBiasAdd j.val)
  else .error (.nonFiniteDot j.val)

@[inline] def nativeLogit (p : Parameters inputs outputs) (x : Buffer inputs)
    (j : Fin outputs) : Except Error Float32 :=
  let dot := nativeDot p x j
  if dot.isFinite then
    let result := dot + p.bias.get j
    if result.isFinite then .ok result else .error (.nonFiniteBiasAdd j.val)
  else .error (.nonFiniteDot j.val)

/-- Indexed result retains the output shape without a second size check. -/
def reference (p : Parameters inputs outputs) (x : Buffer inputs) :
    Except Error (Vector UInt32 outputs) := do
  validate .image x
  validate .weights p.weights
  validate .bias p.bias
  Vector.ofFnM fun j => return (← referenceLogit p x j).toBits

def forward (p : Parameters inputs outputs) (x : Buffer inputs) :
    Except Error (Vector UInt32 outputs) := do
  validate .image x
  validate .weights p.weights
  validate .bias p.bias
  Vector.ofFnM fun j => return (← nativeLogit p x j).toBits

/-- Monotone binary32 encoding for finite values, identifying the two zeros.
Numerical validation precedes this ordering; NaNs are not classifier scores. -/
@[inline] def scoreKey (word : UInt32) : Nat :=
  if word &&& 0x7fffffff == 0 then 0x80000000
  else if word &&& 0x80000000 == 0 then (word ^^^ 0x80000000).toNat
  else (~~~word).toNat

/-- Greedy scan updates only for strict improvement, preserving the first tie. -/
def argmaxPrefix (score : Fin (n + 1) → Nat) : (count : Nat) → count ≤ n → Fin (n + 1)
  | 0, _ => 0
  | count + 1, h =>
    let best := argmaxPrefix score count (by omega)
    let next : Fin (n + 1) := ⟨count + 1, by omega⟩
    if score best < score next then next else best

def argmax (scores : Fin (n + 1) → UInt32) : Fin (n + 1) :=
  argmaxPrefix (fun i => scoreKey (scores i)) n (Nat.le_refl n)

def predict (p : Parameters inputs (n + 1)) (x : Buffer inputs) :
    Except Error (Fin (n + 1)) := do
  let logits ← forward p x
  return argmax fun i => logits[i]

abbrev MNIST := Parameters 784 10

end Cybernetix.Model.LinearClassifier
