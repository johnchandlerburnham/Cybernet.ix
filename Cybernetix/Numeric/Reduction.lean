import Init.Data.Float.Float32
import Init.Data.Nat.Log2

/-! Canonical binary trees. Failed subtrees return their first nonfinite value;
the enclosing tensor operation assigns its logical error location. Success
therefore requires finite leaves and finite results at every tree node.
-/

namespace Cybernetix.Numeric.Reduction

def depth (length : Nat) : Nat :=
  if length ≤ 1 then 0 else (length - 1).log2 + 1

/-- Consecutive leaves, adjacent pairs, and separately rounded additions. -/
def referenceTree (leaf : Nat → Float32.Model) : Nat → Nat → Float32.Model
  | 0, start => leaf start
  | height + 1, start =>
    let left := referenceTree leaf height start
    if left.isFinite then
      let right := referenceTree leaf height (start + 2 ^ height)
      if right.isFinite then left + right else right
    else left

/-- Concrete scalar types keep the recursive result unboxed in generated C. -/
@[specialize] def nativeTree (leaf : Nat → Float32) : Nat → Nat → Float32
  | 0, start => leaf start
  | height + 1, start =>
    let left := nativeTree leaf height start
    if left.isFinite then
      let right := nativeTree leaf height (start + 2 ^ height)
      if right.isFinite then left + right else right
    else left

/-- Carry the subtree width rather than recomputing a power at every node. -/
@[specialize] def nativeTreeSpan (leaf : Nat → Float32) : Nat → Nat → Nat → Float32
  | 0, start, _ => leaf start
  | height + 1, start, span =>
    let half := span / 2
    let left := nativeTreeSpan leaf height start half
    if left.isFinite then
      let right := nativeTreeSpan leaf height (start + half) half
      if right.isFinite then left + right else right
    else left

@[inline] def nativeSpan (length : Nat) (leaf : Fin length → Float32) : Float32 :=
  nativeTreeSpan (fun i => if h : i < length then leaf ⟨i, h⟩ else .ofBits 0)
    (depth length) 0 (2 ^ depth length)

/-- Machine-sized tree traversal; the caller proves the complete span fits. -/
@[specialize] def nativeTreeU (leaf : USize → Float32) : Nat → USize → USize → Float32
  | 0, start, _ => leaf start
  | height + 1, start, span =>
    let half := span / 2
    let left := nativeTreeU leaf height start half
    if left.isFinite then
      let right := nativeTreeU leaf height (start + half) half
      if right.isFinite then left + right else right
    else left

def reference (length : Nat) (leaf : Fin length → Float32.Model) : Float32.Model :=
  referenceTree (fun i => if h : i < length then leaf ⟨i, h⟩ else .ofBits 0)
    (depth length) 0

@[inline] def native (length : Nat) (leaf : Fin length → Float32) : Float32 :=
  nativeTree (fun i => if h : i < length then leaf ⟨i, h⟩ else .ofBits 0)
    (depth length) 0

end Cybernetix.Numeric.Reduction
