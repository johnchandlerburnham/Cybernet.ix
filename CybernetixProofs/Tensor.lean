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

theorem ofBytes_roundtrip (b : Buffer n) : ofBytes n b.bytes = some b := by
  simp [ofBytes, b.size_eq]

theorem ofBytes_reject (bytes : ByteArray) (h : bytes.size ≠ 4 * n) :
    ofBytes n bytes = none := by simp [ofBytes, h]

end Buffer
end Cybernetix.Tensor
