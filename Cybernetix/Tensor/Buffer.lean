import Init.Data.ByteArray.Lemmas
import Init.Data.Float.Float32
import Init.Omega

/-! Packed, little-endian binary32 storage. Shape proofs erase at runtime.
The storage layer preserves words, including signed zero and NaN payloads;
numerical consumers separately validate their admissible values.
-/

namespace Cybernetix.Tensor

@[inline] def wordByte (word : UInt32) (byte : Nat) : UInt8 :=
  (word >>> UInt32.ofNat (8 * byte)).toUInt8

@[inline] def assembleWord (a b c d : UInt8) : UInt32 :=
  a.toUInt32 ||| (b.toUInt32 <<< 8) ||| (c.toUInt32 <<< 16) ||| (d.toUInt32 <<< 24)

structure Buffer (length : Nat) where
  bytes : ByteArray
  size_eq : bytes.size = 4 * length
  deriving DecidableEq

namespace Buffer

def ofBytes (length : Nat) (bytes : ByteArray) : Option (Buffer length) :=
  if h : bytes.size = 4 * length then some ⟨bytes, h⟩ else none

@[inline] def getWord (buffer : Buffer n) (i : Fin n) : UInt32 :=
  let offset := 4 * i.val
  assembleWord
    (buffer.bytes[offset]'(by have := buffer.size_eq; omega))
    (buffer.bytes[offset + 1]'(by have := buffer.size_eq; omega))
    (buffer.bytes[offset + 2]'(by have := buffer.size_eq; omega))
    (buffer.bytes[offset + 3]'(by have := buffer.size_eq; omega))

@[inline] def get (buffer : Buffer n) (i : Fin n) : Float32 :=
  Float32.ofBits (buffer.getWord i)

def view (buffer : Buffer n) (i : Fin n) : Float32.Model :=
  Float32.Model.ofBits (buffer.getWord i)

/-- Construction is outside the hot numerical loop. -/
def ofFn (f : Fin n → UInt32) : Buffer n :=
  ⟨⟨Array.ofFn fun (byte : Fin (4 * n)) =>
    wordByte (f ⟨byte.val / 4, by omega⟩) (byte.val % 4)⟩, by simp [ByteArray.size]⟩

def ofWords (words : Array UInt32) : Buffer words.size :=
  ofFn fun i => words[i]

def words (buffer : Buffer n) : Array UInt32 := Array.ofFn buffer.getWord

/-- A row-major offset into a statically shaped matrix. -/
@[inline] def matrixIndex (row : Fin rows) (column : Fin columns) : Fin (rows * columns) :=
  ⟨row.val * columns + column.val, by
    have h := Nat.mul_le_mul_right columns (Nat.succ_le_of_lt row.isLt)
    simp only [Nat.succ_mul] at h
    omega⟩

@[inline] def matrixGet (buffer : Buffer (rows * columns))
    (row : Fin rows) (column : Fin columns) : Float32 :=
  buffer.get (matrixIndex row column)

end Buffer
end Cybernetix.Tensor
