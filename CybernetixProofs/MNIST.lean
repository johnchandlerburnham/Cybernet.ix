import Cybernetix.Corpus.MNIST
import CybernetixProofs.Tensor

namespace Cybernetix.Corpus.MNIST

theorem pixelWord_eq (pixel : UInt8) : pixelWord pixel = (pixelReference pixel).toBits := rfl

theorem input_word (images : Images) (sample : Fin images.count) (feature : Fin 784) :
    (images.input sample).getWord feature = (pixelReference (images.pixel sample feature)).toBits := by
  simp only [Images.input, Tensor.Buffer.getWord_ofFn, pixelWord_eq]

set_option maxRecDepth 8192 in
set_option maxHeartbeats 4000000 in
/-- Exhaustive kernel checking of the 256 byte inputs, without native evaluation. -/
theorem pixels_finite : ∀ i : Fin 256,
    (pixelReference i.val.toUInt8).isFinite = true ∧
    (pixelReference i.val.toUInt8).toBits.toNat ≤ 0x3f800000 := by
  decide +kernel

theorem train_index (i : Fin Split.train.size) : Split.train.releaseIndex i < 55000 := i.isLt

theorem validation_index (i : Fin Split.validation.size) :
    55000 ≤ Split.validation.releaseIndex i ∧ Split.validation.releaseIndex i < 60000 := by
  have := i.isLt
  simp only [Split.releaseIndex, Split.size] at *
  omega

theorem train_validation_disjoint (train : Fin Split.train.size)
    (validation : Fin Split.validation.size) :
    Split.train.releaseIndex train ≠ Split.validation.releaseIndex validation := by
  have := train_index train
  have := validation_index validation
  omega

end Cybernetix.Corpus.MNIST
