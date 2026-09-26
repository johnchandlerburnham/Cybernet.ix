import Cybernetix.Model.LinearClassifier

namespace Cybernetix.Model.LinearClassifier

inductive ArtifactError where
  | length (expected actual : Nat)
  | numerical (error : Error)
  deriving Repr, DecidableEq

/-- The v1 payload is row-major weights followed by bias, all little-endian.
Its shape and numerical profile must be bound by the enclosing claim. -/
def Parameters.encode (p : Parameters inputs outputs) : ByteArray :=
  p.weights.bytes ++ p.bias.bytes

def decodeParameters (inputs outputs : Nat) (bytes : ByteArray) :
    Except ArtifactError (Parameters inputs outputs) := do
  if h : bytes.size = 4 * (inputs * outputs) + 4 * outputs then
    let weights : Tensor.Buffer (inputs * outputs) :=
      ⟨bytes.extract 0 (4 * (inputs * outputs)), by simp [ByteArray.size_extract, h]⟩
    let bias : Tensor.Buffer outputs :=
      ⟨bytes.extract (4 * (inputs * outputs)) bytes.size, by simp [ByteArray.size_extract, h]⟩
    validate .weights weights |>.mapError ArtifactError.numerical
    validate .bias bias |>.mapError ArtifactError.numerical
    return ⟨weights, bias⟩
  else throw (.length (4 * (inputs * outputs) + 4 * outputs) bytes.size)

end Cybernetix.Model.LinearClassifier
