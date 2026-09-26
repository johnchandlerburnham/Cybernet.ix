import Cybernetix.Corpus.MNIST
import Cybernetix.Model.LinearArtifact

namespace Cybernetix.Tests.MNIST

open Corpus.MNIST Model.LinearClassifier Tensor

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw <| IO.userError message

private def be32 (n : Nat) : ByteArray :=
  let w := n.toUInt32
  ⟨#[(w >>> 24).toUInt8, (w >>> 16).toUInt8, (w >>> 8).toUInt8, w.toUInt8]⟩

private def imageHeader (count : Nat) (rows : Nat := 28) (columns : Nat := 28) : ByteArray :=
  be32 2051 ++ be32 count ++ be32 rows ++ be32 columns

private def images (count : Nat) : ByteArray :=
  imageHeader count ++ ⟨Array.ofFn fun (i : Fin (count * 784)) => UInt8.ofNat i.val⟩

private def labels (values : Array UInt8) : ByteArray :=
  be32 2049 ++ be32 values.size ++ ⟨values⟩

private def error? : Except Corpus.MNIST.Error α → Option Corpus.MNIST.Error
  | .ok _ => none
  | .error e => some e

private def rejected : Except ε α → Bool
  | .ok _ => false
  | .error _ => true

def run : IO Unit := do
  let raw := images 2
  let dataset ← match decode raw (labels #[3, 7]) with
    | .ok result => pure result
    | .error e => throw <| IO.userError s!"Valid IDX rejected: {repr e}"
  check (dataset.images.count == 2) "big-endian image count"
  check (dataset.labels.toArray.map Fin.val == #[3, 7]) "IDX labels"
  if h : 1 < dataset.images.count then
    check (dataset.images.pixel ⟨0, by omega⟩ 255 == 255) "pixel layout"
    check (dataset.images.pixel ⟨1, h⟩ 0 == 16) "sample stride"
    let input := dataset.images.input ⟨0, by omega⟩
    check (input.getWord 0 == 0 && input.getWord 255 == 0x3f800000) "pixel endpoints"
    check (input.getWord 128 == 0x3f008081) "pixel division by 255"
    let p : Parameters 784 10 := ⟨Buffer.ofFn fun _ => 0,
      Buffer.ofFn fun i => if i.val == 7 then 0x3f800000 else 0⟩
    match predict p input with
    | .ok digit => check (digit.val == 7) "IDX to typed prediction"
    | .error _ => throw <| IO.userError "IDX inference failed"
    let encoded := p.encode
    check (encoded.size == 31400) "model payload length"
    match decodeParameters 784 10 encoded with
    | .ok decoded => check (decoded.encode == encoded) "model payload round trip"
    | .error _ => throw <| IO.userError "model payload rejected"
    check (rejected (decodeParameters 784 10 (encoded.push 0))) "model trailing byte accepted"
    check (rejected (decodeParameters 784 10 (encoded.extract 0 31399))) "short model accepted"
    let nanModel : Parameters 1 1 := ⟨Buffer.ofWords #[0x7fc00000], Buffer.ofWords #[0]⟩
    check (rejected (decodeParameters 1 1 nanModel.encode)) "nonfinite model accepted"
  else throw <| IO.userError "Wrong dataset count"

  for i in [:256] do
    let byte := UInt8.ofNat i
    check (pixelWord byte == (pixelReference byte).toBits) s!"pixel native/reference mismatch at {i}"
  check (error? (decodeImages ⟨#[]⟩) == some (.header .images 16 0)) "short image header"
  check (error? (decodeLabels ⟨#[0, 0, 8, 1]⟩) == some (.header .labels 8 4)) "short label header"
  check (error? (decodeImages (be32 2049 ++ raw.extract 4 raw.size)) == some (.magic .images 2049))
    "wrong image magic"
  check (error? (decodeLabels (be32 2051 ++ be32 0)) == some (.magic .labels 2051)) "wrong label magic"
  check (error? (decodeImages (imageHeader 1 27)) == some (.dimensions 27 28)) "wrong image dimensions"
  check (error? (decodeImages (raw.extract 0 (raw.size - 1))) ==
    some (.length .images 1584 1583)) "truncated image data"
  check (error? (decodeImages (raw.push 0)) == some (.length .images 1584 1585)) "trailing image data"
  check (error? (decodeImages (imageHeader 4294967295)) ==
    some (.length .images (16 + 4294967295 * 784) 16)) "overflowing untrusted byte length"
  check (error? (decodeLabels (be32 2049 ++ be32 2 ++ ⟨#[0]⟩)) ==
    some (.length .labels 10 9)) "truncated labels"
  check (error? (decodeLabels ((labels #[0]).push 0)) == some (.length .labels 9 10)) "trailing labels"
  check (error? (decode raw (labels #[0])) == some (.countMismatch 2 1)) "image/label count mismatch"
  check (error? (decodeLabels (labels #[0, 10, 255])) == some (.label 1 10)) "invalid label/error order"
  check (error? (decodeLabels (labels #[255])) == some (.label 0 255)) "large invalid label"
  IO.println "MNIST IDX admission, byte preprocessing, split shapes, and model artifact tests passed."

end Cybernetix.Tests.MNIST
