import Cybernetix.Tensor.Buffer

/-! MNIST's byte representation and shared inference/training preprocessing.
IDX headers are big-endian; packed model tensors are little-endian binary32.
Decoding is pure and does not imply that a file matches the release manifest.
-/

namespace Cybernetix.Corpus.MNIST

open Tensor

def preprocessingProfile : String := "cybernetix.mnist-u8-div255.v1.lean-4.33.1"

inductive FileKind where
  | images | labels
  deriving Repr, DecidableEq

inductive Error where
  | header (kind : FileKind) (required actual : Nat)
  | magic (kind : FileKind) (actual : Nat)
  | dimensions (rows columns : Nat)
  | length (kind : FileKind) (expected actual : Nat)
  | countMismatch (images labels : Nat)
  | label (index value : Nat)
  deriving Repr, DecidableEq

structure Images where
  count : Nat
  pixels : ByteArray
  size_eq : pixels.size = count * 784

structure Labels where
  count : Nat
  values : Vector (Fin 10) count

structure Dataset where
  images : Images
  labels : Vector (Fin 10) images.count

/-- Natural-number arithmetic avoids overflow in untrusted length fields. -/
def readBE32 (bytes : ByteArray) (offset : Nat) (h : offset + 4 ≤ bytes.size) : Nat :=
  bytes[offset].toNat * 16777216 + bytes[offset + 1].toNat * 65536 +
    bytes[offset + 2].toNat * 256 + bytes[offset + 3].toNat

def decodeImages (bytes : ByteArray) : Except Error Images := do
  if h : 16 ≤ bytes.size then
    let magic := readBE32 bytes 0 (by omega)
    unless magic == 2051 do throw (.magic .images magic)
    let count := readBE32 bytes 4 (by omega)
    let rows := readBE32 bytes 8 (by omega)
    let columns := readBE32 bytes 12 (by omega)
    unless rows == 28 && columns == 28 do throw (.dimensions rows columns)
    if hsize : bytes.size = 16 + count * 784 then
      return ⟨count, bytes.extract 16 bytes.size, by simp [ByteArray.size_extract, hsize]⟩
    else throw (.length .images (16 + count * 784) bytes.size)
  else throw (.header .images 16 bytes.size)

def decodeLabels (bytes : ByteArray) : Except Error Labels := do
  if h : 8 ≤ bytes.size then
    let magic := readBE32 bytes 0 (by omega)
    unless magic == 2049 do throw (.magic .labels magic)
    let count := readBE32 bytes 4 (by omega)
    if hsize : bytes.size = 8 + count then
      let values ← Vector.ofFnM fun (i : Fin count) =>
        let value := (bytes[8 + i.val]'(by omega)).toNat
        if hvalue : value < 10 then .ok (⟨value, hvalue⟩ : Fin 10)
        else .error (.label i.val value)
      return ⟨count, values⟩
    else throw (.length .labels (8 + count) bytes.size)
  else throw (.header .labels 8 bytes.size)

def decode (imageBytes labelBytes : ByteArray) : Except Error Dataset := do
  let images ← decodeImages imageBytes
  let labels ← decodeLabels labelBytes
  if h : labels.count = images.count then
    return ⟨images, h ▸ labels.values⟩
  else throw (.countMismatch images.count labels.count)

def Images.pixel (images : Images) (sample : Fin images.count) (feature : Fin 784) : UInt8 :=
  images.pixels[sample.val * 784 + feature.val]'(by
    have h := Nat.mul_le_mul_right 784 (Nat.succ_le_of_lt sample.isLt)
    simp only [Nat.succ_mul] at h
    have := images.size_eq
    omega)

def pixelReference (pixel : UInt8) : Float32.Model :=
  Float32.Model.ofUInt8 pixel / Float32.Model.ofBits 0x437f0000

@[inline] def pixelWord (pixel : UInt8) : UInt32 :=
  (pixel.toFloat32 / Float32.ofBits 0x437f0000).toBits

/-- Exact byte conversion, then one separately rounded division by 255. -/
def Images.input (images : Images) (sample : Fin images.count) : Buffer 784 :=
  Buffer.ofFn fun feature => pixelWord (images.pixel sample feature)

inductive Split where
  | train | validation | test
  deriving Repr, DecidableEq

def Split.size : Split → Nat
  | .train => 55000
  | .validation => 5000
  | .test => 10000

/-- Indices refer to the named release, without reshuffling or split leakage. -/
def Split.releaseIndex (split : Split) (i : Fin split.size) : Nat :=
  match split with
  | .train => i.val
  | .validation => 55000 + i.val
  | .test => i.val

end Cybernetix.Corpus.MNIST
