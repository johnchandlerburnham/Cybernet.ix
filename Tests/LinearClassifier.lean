import Cybernetix.Model.LinearClassifier

namespace Cybernetix.Tests.LinearClassifier

open Tensor Model.LinearClassifier

deriving instance DecidableEq for Except

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw <| IO.userError message

private def compare (p : Parameters inputs outputs) (x : Buffer inputs) : IO Unit :=
  check (decide (forward p x = reference p x)) "dense native/reference mismatch"

def run : IO Unit := do
  let words : Array UInt32 := #[0, 0x80000000, 0x3f800000, 0x00000001,
    0x7f7fffff, 0x7f800000, 0x7fc01234, 0xffffffff]
  let packed := Buffer.ofWords words
  check (packed.bytes.size == 32) "packed storage byte count"
  check (packed.words == words) "packed storage lost scalar bits"
  check (packed.bytes.extract 8 12 == ⟨#[0, 0, 128, 63]⟩) "packed storage endianness"
  check ((Buffer.ofBytes 7 packed.bytes).isNone) "accepted wrong tensor shape"
  check ((Buffer.ofBytes 8 packed.bytes).isSome) "rejected valid tensor shape"

  -- Asymmetric weights distinguish row-major from transposed layouts.
  let p : Parameters 3 2 := ⟨Buffer.ofWords #[0x3f800000, 0x40000000,
    0x40400000, 0x40800000, 0x40a00000, 0x40c00000],
    Buffer.ofWords #[0x3f800000, 0xbf800000]⟩
  let x := Buffer.ofWords #[0x3f800000, 0x40000000, 0x40400000]
  compare p x
  check (decide (forward p x = .ok #v[0x41b80000, 0x41d80000])) "row-major affine logits"
  check (decide (predict p x = .ok 1)) "typed dense prediction"

  let biasOnly : Parameters 0 3 := ⟨Buffer.ofWords #[],
    Buffer.ofWords #[0xbf800000, 0xc0000000, 0xbf800000]⟩
  compare biasOnly (Buffer.ofWords #[])
  check (decide (predict biasOnly (Buffer.ofWords #[]) = .ok 0)) "negative argmax tie"
  check ((argmax (n := 2) fun _ => 0x3f800000).val == 0) "all-equal argmax tie"
  check ((argmax (n := 1) fun i => if i.val == 0 then 0x80000000 else 0).val == 0)
    "signed-zero argmax tie"
  check ((argmax (n := 1) fun i => if i.val == 0 then 0 else 0x00000001).val == 1)
    "positive subnormal ordering"

  let cancel : Parameters 3 1 := ⟨Buffer.ofWords #[0x4b800000, 0x3f800000, 0xcb800000],
    Buffer.ofWords #[0]⟩
  let ones := Buffer.ofFn (n := 3) fun _ => 0x3f800000
  compare cancel ones
  check (decide (forward cancel ones = .ok #v[0])) "canonical cancellation tree"
  let negZeros := fun (_ : Fin 3) => Float32.ofBits 0x80000000
  check ((Numeric.Reduction.native 1 (fun _ => Float32.ofBits 0x80000000)).toBits == 0x80000000)
    "singleton changed signed zero"
  check ((Numeric.Reduction.native 3 negZeros).toBits == 0) "odd tree did not pad with positive zero"

  let badImage := Buffer.ofWords #[0x7fc01234, 0x7f800000, 0]
  compare p badImage
  check (decide (forward p badImage = .error (.nonFiniteInput .image 0)))
    "nonfinite image error order"
  let badWeights : Parameters 3 2 := { p with
    weights := Buffer.ofFn fun i => if i.val == 2 then 0xff800000 else 0 }
  compare badWeights x
  check (decide (forward badWeights x = .error (.nonFiniteInput .weights 2)))
    "nonfinite weight error location"
  let badBias : Parameters 3 2 := { p with bias := Buffer.ofWords #[0, 0x7fc00000] }
  compare badBias x
  check (decide (forward badBias x = .error (.nonFiniteInput .bias 1))) "nonfinite bias admission"

  let overflow : Parameters 2 1 := ⟨Buffer.ofFn fun _ => 0x7f7fffff, Buffer.ofWords #[0]⟩
  let twoOnes := Buffer.ofFn (n := 2) fun _ => 0x3f800000
  compare overflow twoOnes
  check (decide (forward overflow twoOnes = .error (.nonFiniteDot 0))) "reduction overflow"
  let overflowBias : Parameters 1 1 := ⟨Buffer.ofWords #[0x7f7fffff], Buffer.ofWords #[0x7f7fffff]⟩
  let one := Buffer.ofWords #[0x3f800000]
  compare overflowBias one
  check (decide (forward overflowBias one = .error (.nonFiniteBiasAdd 0))) "bias overflow"

  -- Full MNIST dimensions with nonzero, alternating-sign, exact dyadic values.
  let full : MNIST := ⟨Buffer.ofFn fun i =>
      if i.val % 3 == 0 then 0x3d800000 else if i.val % 3 == 1 then 0xbd800000 else 0,
    Buffer.ofFn fun i => (UInt8.ofNat i.val).toFloat32.toBits⟩
  let image := Buffer.ofFn (n := 784) fun i =>
    if i.val % 2 == 0 then 0x3f000000 else 0x3e800000
  compare full image
  check ((full.weights.bytes.size + full.bias.bytes.size) == 31400) "MNIST parameter bytes"
  IO.println "Packed tensors, deterministic dense inference, errors, and argmax tests passed."

end Cybernetix.Tests.LinearClassifier
