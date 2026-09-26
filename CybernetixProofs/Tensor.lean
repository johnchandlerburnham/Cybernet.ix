import Cybernetix.Tensor.Buffer
import Init.Data.UInt.Lemmas

namespace Cybernetix.Tensor

/-- All words round-trip, without a native-evaluation axiom. -/
theorem assembleWord_wordByte (w : UInt32) :
    assembleWord (wordByte w 0) (wordByte w 1) (wordByte w 2) (wordByte w 3) = w := by
  apply UInt32.eq_of_toBitVec_eq
  apply BitVec.eq_of_getLsbD_eq
  intro i
  simp [assembleWord, wordByte]
  intro hi
  have hb (x : BitVec 32) (j : Nat) :
      (x % 256#32).getLsbD j = (decide (j < 8) && x.getLsbD j) := by
    exact Nat.testBit_mod_two_pow x.toNat 8 j
  simp only [hb, BitVec.getLsbD_ushiftRight]
  by_cases h0 : i < 8
  · have h1 : i < 16 := by omega
    have h2 : i < 24 := by omega
    simp [hi, h0, h1, h2]
  · by_cases h1 : i < 16
    · have h2 : i < 24 := by omega
      have h3 : i - 8 < 8 := by omega
      have h4 : 8 + (i - 8) = i := by omega
      simp [hi, h0, h1, h2, h3, h4]
    · by_cases h2 : i < 24
      · have h3 : ¬i - 8 < 8 := by omega
        have h4 : i - 16 < 8 := by omega
        have h5 : 16 + (i - 16) = i := by omega
        simp [hi, h0, h1, h2, h3, h4, h5]
      · have h3 : ¬i - 8 < 8 := by omega
        have h4 : ¬i - 16 < 8 := by omega
        have h5 : i - 24 < 8 := by omega
        have h6 : 24 + (i - 24) = i := by omega
        simp [hi, h0, h1, h2, h3, h4, h5, h6]

namespace Buffer

theorem getWord_ofFn (f : Fin n → UInt32) (i : Fin n) : (ofFn f).getWord i = f i := by
  have h0 : (4 * i.val) / 4 = i.val := by omega
  have h1 : (4 * i.val + 1) / 4 = i.val := by omega
  have h2 : (4 * i.val + 2) / 4 = i.val := by omega
  have h3 : (4 * i.val + 3) / 4 = i.val := by omega
  have m0 : (4 * i.val) % 4 = 0 := by omega
  have m1 : (4 * i.val + 1) % 4 = 1 := by omega
  have m2 : (4 * i.val + 2) % 4 = 2 := by omega
  have m3 : (4 * i.val + 3) % 4 = 3 := by omega
  simpa [getWord, ofFn, ByteArray.getElem_eq_getElem_data,
    h0, h1, h2, h3, m0, m1, m2, m3] using
    assembleWord_wordByte (f i)

theorem view_ofFn (f : Fin n → UInt32) (i : Fin n) :
    (ofFn f).view i = Float32.Model.ofBits (f i) := by
  simp only [view, getWord_ofFn]

theorem get_model (b : Buffer n) (i : Fin n) : (b.get i).toModel = b.view i := rfl

theorem getWordFast_eq (b : Buffer n) (i : Fin n) (small : b.bytes.size < USize.size) :
    b.getWordFast i small = b.getWord i := by
  have hbound : 4 * i.val + 4 < USize.size := by have := b.size_eq; omega
  have hindex (k : Fin 4) :
      (USize.ofNat (4 * i.val) + k.val.toUSize).toNat = 4 * i.val + k.val := by
    have h0 : 4 * i.val < USize.size := by omega
    have hk : k.val < USize.size := by omega
    change ((4 * i.val) % USize.size + k.val % USize.size) % USize.size = _
    rw [Nat.mod_eq_of_lt h0, Nat.mod_eq_of_lt hk, Nat.mod_eq_of_lt (by omega)]
  have hu (bytes : ByteArray) (index : USize) (h : index.toNat < bytes.size) :
      bytes.uget index h = bytes[index.toNat] := rfl
  simp only [getWordFast, hu]
  have h0 : (USize.ofNat (4 * i.val)).toNat = 4 * i.val := by
    change (4 * i.val) % USize.size = _
    exact Nat.mod_eq_of_lt (by omega)
  have h1 := hindex ⟨1, by decide⟩
  have h2 := hindex ⟨2, by decide⟩
  have h3 := hindex ⟨3, by decide⟩
  change (USize.ofNat (4 * i.val) + 1).toNat = 4 * i.val + 1 at h1
  change (USize.ofNat (4 * i.val) + 2).toNat = 4 * i.val + 2 at h2
  change (USize.ofNat (4 * i.val) + 3).toNat = 4 * i.val + 3 at h3
  simp only [USize.ofNatLT_eq_ofNat, h0, h1, h2, h3]
  rfl

theorem getFast_model (b : Buffer n) (i : Fin n) (small : b.bytes.size < USize.size) :
    (b.getFast i small).toModel = b.view i := by
  simp only [getFast, Float32.ofBits, getWordFast_eq, view]

theorem getWordU_eq (b : Buffer n) (i : USize) (hi : i.toNat < n)
    (small : b.bytes.size < USize.size) : b.getWordU i hi small = b.getWord ⟨i.toNat, hi⟩ := by
  have h4 : 4 < USize.size := by have := b.size_eq; omega
  have hoff : (i * 4).toNat = 4 * i.toNat := by
    change (i.toNat * (4 % USize.size)) % USize.size = _
    rw [Nat.mod_eq_of_lt h4, Nat.mod_eq_of_lt (by have := b.size_eq; omega), Nat.mul_comm]
  have hindex (k : Fin 4) : (i * 4 + k.val.toUSize).toNat = 4 * i.toNat + k.val := by
    change ((i * 4).toNat + k.val % USize.size) % USize.size = _
    rw [hoff, Nat.mod_eq_of_lt (show k.val < USize.size by omega),
      Nat.mod_eq_of_lt (show 4 * i.toNat + k.val < USize.size by have := b.size_eq; omega)]
  have hu (bytes : ByteArray) (index : USize) (h : index.toNat < bytes.size) :
      bytes.uget index h = bytes[index.toNat] := rfl
  have h1 := hindex ⟨1, by decide⟩
  have h2 := hindex ⟨2, by decide⟩
  have h3 := hindex ⟨3, by decide⟩
  change (i * 4 + 1).toNat = 4 * i.toNat + 1 at h1
  change (i * 4 + 2).toNat = 4 * i.toNat + 2 at h2
  change (i * 4 + 3).toNat = 4 * i.toNat + 3 at h3
  simp only [getWordU, hu, hoff, h1, h2, h3]
  rfl

theorem getU_model (b : Buffer n) (i : USize) (hi : i.toNat < n)
    (small : b.bytes.size < USize.size) : (b.getU i hi small).toModel = b.view ⟨i.toNat, hi⟩ := by
  simp only [getU, Float32.ofBits, getWordU_eq, view]

theorem ofBytes_roundtrip (b : Buffer n) : ofBytes n b.bytes = some b := by
  simp [ofBytes, b.size_eq]

theorem ofBytes_reject (bytes : ByteArray) (h : bytes.size ≠ 4 * n) :
    ofBytes n bytes = none := by simp [ofBytes, h]

end Buffer
end Cybernetix.Tensor
