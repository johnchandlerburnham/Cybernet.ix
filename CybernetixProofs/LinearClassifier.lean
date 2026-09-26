import Cybernetix.Model.LinearClassifier
import Init.Data.Vector.Lemmas
import Init.Data.Nat.Lemmas
import CybernetixProofs.Tensor

namespace Cybernetix.Numeric.Reduction

theorem depth_covers (length : Nat) : length ≤ 2 ^ depth length := by
  unfold depth
  split
  · simpa using ‹length ≤ 1›
  · have hne : length - 1 ≠ 0 := by omega
    have h := (Nat.log2_lt hne).mp (Nat.lt_succ_self (length - 1).log2)
    omega

theorem depth_minimal (length height : Nat) (h : length ≤ 2 ^ height) :
    depth length ≤ height := by
  unfold depth
  split
  · exact Nat.zero_le _
  · have hne : length - 1 ≠ 0 := by omega
    have : length - 1 < 2 ^ height := by omega
    exact Nat.succ_le_of_lt ((Nat.log2_lt hne).mpr this)

@[simp] theorem add_model (a b : Float32) : (a + b).toModel = a.toModel + b.toModel := rfl
@[simp] theorem mul_model (a b : Float32) : (a * b).toModel = a.toModel * b.toModel := rfl
@[simp] theorem finite_model (a : Float32) : a.isFinite = a.toModel.isFinite := rfl

/-- Logical refinement only: compiled native instructions have a separate contract. -/
theorem nativeTree_model (leaf : Nat → Float32) (height start : Nat) :
    (nativeTree leaf height start).toModel =
      referenceTree (fun i => (leaf i).toModel) height start := by
  induction height generalizing start with
  | zero => rfl
  | succ height ih =>
    rw [nativeTree, referenceTree]
    simp only [apply_ite, finite_model, add_model, ih]
    split <;> simp_all

theorem native_model (length : Nat) (leaf : Fin length → Float32) :
    (native length leaf).toModel = reference length (fun i => (leaf i).toModel) := by
  simp only [native, reference, nativeTree_model]
  congr 1
  funext i
  split <;> rfl

theorem nativeTreeSpan_eq (leaf : Nat → Float32) (height start : Nat) :
    nativeTreeSpan leaf height start (2 ^ height) = nativeTree leaf height start := by
  induction height generalizing start with
  | zero => rfl
  | succ height ih =>
    rw [nativeTreeSpan, nativeTree]
    have hhalf : 2 ^ (height + 1) / 2 = 2 ^ height := by
      simp [Nat.pow_succ]
    simp only [hhalf, ih]

theorem nativeSpan_model (length : Nat) (leaf : Fin length → Float32) :
    (nativeSpan length leaf).toModel = reference length (fun i => (leaf i).toModel) := by
  simp only [nativeSpan, nativeTreeSpan_eq]
  exact native_model length leaf

theorem nativeTreeU_eq (leaf : USize → Float32) (height : Nat) (start span : USize)
    (region : start.toNat + span.toNat ≤ USize.size) :
    nativeTreeU leaf height start span =
      nativeTreeSpan (fun i => if i < USize.size then leaf i.toUSize else .ofBits 0)
        height start.toNat span.toNat := by
  induction height generalizing start span with
  | zero => simp [nativeTreeU, nativeTreeSpan, start.toNat_lt_size]
  | succ height ih =>
    have hhalf : (span / 2).toNat = span.toNat / 2 := by simp
    have hnext : (start + span / 2).toNat = start.toNat + span.toNat / 2 := by
      rw [USize.toNat_add, hhalf]
      apply Nat.mod_eq_of_lt
      have := start.toNat_lt_size
      change start.toNat + span.toNat / 2 < USize.size
      omega
    rw [nativeTreeU, nativeTreeSpan]
    rw [ih start (span / 2) (by rw [hhalf]; omega)]
    rw [ih (start + span / 2) (span / 2) (by rw [hhalf, hnext]; omega)]
    simp only [hhalf, hnext]

end Cybernetix.Numeric.Reduction

namespace Cybernetix.Model.LinearClassifier

theorem validate_eq (kind : InputKind) (buffer : Tensor.Buffer n) :
    validate kind buffer = validateScalars kind buffer.get := by
  unfold validate
  split
  · congr 1
    funext i
    simp only [Tensor.Buffer.getFast, Tensor.Buffer.get, Tensor.Buffer.getWordFast_eq]
  · rfl

theorem machineLeaf_model (p : Parameters inputs outputs) (x : Tensor.Buffer inputs)
    (j : Fin outputs) (hx : x.bytes.size < USize.size)
    (hw : p.weights.bytes.size < USize.size) (columns : outputs < USize.size) (i : Nat) :
    (if i < USize.size then machineLeaf p x j hx hw columns i.toUSize
      else Float32.ofBits 0).toModel =
    (if h : i < inputs then
      x.view ⟨i, h⟩ * p.weights.view (Tensor.Buffer.matrixIndex ⟨i, h⟩ j)
    else Float32.Model.ofBits 0) := by
  have hinputs : inputs < USize.size := by have := x.size_eq; omega
  by_cases hsmall : i < USize.size
  · have hi : i.toUSize.toNat = i := Nat.mod_eq_of_lt hsmall
    simp only [if_pos hsmall, machineLeaf, USize.lt_iff_toNat_lt,
      USize.toNat_ofNatLT, hi]
    split
    · simp only [Numeric.Reduction.mul_model, Tensor.Buffer.getU_model,
        machineIndex_value, hi, Tensor.Buffer.matrixIndex]
    · rfl
  · have hlarge : ¬i < inputs := by omega
    simp only [if_neg hsmall, dif_neg hlarge]
    rfl

theorem nativeDot_model (p : Parameters inputs outputs) (x : Tensor.Buffer inputs)
    (j : Fin outputs) : (nativeDot p x j).toModel = referenceDot p x j := by
  unfold nativeDot referenceDot
  split
  · split
    · split
      · split
        · rw [Numeric.Reduction.nativeTreeU_eq _ _ _ _ (by simp; omega)]
          simp only [USize.toNat_ofNatLT, USize.reduceToNat, Numeric.Reduction.nativeTreeSpan_eq,
            Numeric.Reduction.nativeTree_model]
          unfold Numeric.Reduction.reference
          congr 1
          funext i
          exact machineLeaf_model p x j _ _ _ i
        · rw [Numeric.Reduction.native_model]; rfl
      · rw [Numeric.Reduction.native_model]; rfl
    · rw [Numeric.Reduction.native_model]; rfl
  · rw [Numeric.Reduction.native_model]; rfl

theorem nativeLogit_model (p : Parameters inputs outputs) (x : Tensor.Buffer inputs)
    (j : Fin outputs) : (nativeLogit p x j).map Float32.toModel = referenceLogit p x j := by
  simp only [nativeLogit, referenceLogit, Numeric.Reduction.finite_model,
    Numeric.Reduction.add_model, nativeDot_model]
  split
  · split <;> simp_all [Except.map, Numeric.Reduction.add_model, nativeDot_model,
      Tensor.Buffer.get, Tensor.Buffer.view, Float32.ofBits]
  · rfl

theorem nativeLogit_words (p : Parameters inputs outputs) (x : Tensor.Buffer inputs)
    (j : Fin outputs) :
    (nativeLogit p x j).map Float32.toBits = (referenceLogit p x j).map (·.toBits) := by
  rw [← nativeLogit_model]
  cases nativeLogit p x j <;> rfl

/-- Equal result words and equal logical errors for every shape and input buffer. -/
theorem forward_eq_reference (p : Parameters inputs outputs) (x : Tensor.Buffer inputs) :
    forward p x = reference p x := by
  have h (j : Fin outputs) :
      (do return (← nativeLogit p x j).toBits : Except Error UInt32) =
        (do return (← referenceLogit p x j).toBits : Except Error UInt32) := by
    exact nativeLogit_words p x j
  simp only [forward, reference, h]

/-- Maximum and smallest-index tie rule, established for every admitted score vector. -/
theorem argmaxPrefix_correct (score : Fin (n + 1) → Nat) (count : Nat) (hc : count ≤ n) :
    (argmaxPrefix score count hc).val ≤ count ∧
    ∀ i : Fin (n + 1), i.val ≤ count →
      score i ≤ score (argmaxPrefix score count hc) ∧
      (score i = score (argmaxPrefix score count hc) →
        (argmaxPrefix score count hc).val ≤ i.val) := by
  induction count with
  | zero =>
    simp only [argmaxPrefix]
    constructor
    · exact Nat.le_refl 0
    · intro i hi
      have heq : i = 0 := Fin.ext (by simpa using Nat.eq_zero_of_le_zero hi)
      subst i
      simp
  | succ count ih =>
    have hprev : count ≤ n := by omega
    obtain ⟨hbound, hmax⟩ := ih hprev
    let best := argmaxPrefix score count hprev
    let next : Fin (n + 1) := ⟨count + 1, by omega⟩
    simp only [argmaxPrefix]
    change (if score best < score next then next else best).val ≤ count + 1 ∧
      ∀ i, i.val ≤ count + 1 → score i ≤ score (if score best < score next then next else best) ∧
      (score i = score (if score best < score next then next else best) →
        (if score best < score next then next else best).val ≤ i.val)
    split
    · rename_i hbetter
      constructor
      · exact Nat.le_refl _
      · intro i hi
        change score i ≤ score next ∧ (score i = score next → next.val ≤ i.val)
        by_cases heq : i = next
        · subst i; simp
        · have hi' : i.val ≤ count := by
            have : i.val ≠ next.val := fun h => heq (Fin.ext h)
            dsimp [next] at this
            omega
          obtain ⟨hle, _⟩ := hmax i hi'
          change score i ≤ score best at hle
          constructor <;> omega
    · rename_i hnotbetter
      constructor
      · exact Nat.le_trans hbound (Nat.le_succ count)
      · intro i hi
        change score i ≤ score best ∧ (score i = score best → best.val ≤ i.val)
        by_cases heq : i = next
        · subst i
          constructor
          · omega
          · intro _; exact Nat.le_trans hbound (Nat.le_succ count)
        · have hi' : i.val ≤ count := by
            have : i.val ≠ next.val := fun h => heq (Fin.ext h)
            dsimp [next] at this
            omega
          exact hmax i hi'

theorem argmax_correct (scores : Fin (n + 1) → UInt32) (i : Fin (n + 1)) :
    scoreKey (scores i) ≤ scoreKey (scores (argmax scores)) ∧
      (scoreKey (scores i) = scoreKey (scores (argmax scores)) →
        (argmax scores).val ≤ i.val) :=
  (argmaxPrefix_correct (fun j => scoreKey (scores j)) n (Nat.le_refl n)).2 i (by omega)

end Cybernetix.Model.LinearClassifier
