import Cybernetix.Model.LinearArtifact

namespace Cybernetix.Model.LinearClassifier

theorem decodeParameters_encode (p : Parameters inputs outputs)
    (hw : validate .weights p.weights = .ok ()) (hb : validate .bias p.bias = .ok ()) :
    decodeParameters inputs outputs p.encode = .ok p := by
  have hsize : p.encode.size = 4 * (inputs * outputs) + 4 * outputs := by
    simp [Parameters.encode, p.weights.size_eq, p.bias.size_eq]
  have hleft : p.encode.extract 0 (4 * (inputs * outputs)) = p.weights.bytes := by
    rw [← p.weights.size_eq]
    exact ByteArray.extract_append_eq_left rfl
  have hright : p.encode.extract (4 * (inputs * outputs)) p.encode.size = p.bias.bytes := by
    rw [← p.weights.size_eq]
    exact ByteArray.extract_append_eq_right rfl (ByteArray.size_append ..)
  simp only [decodeParameters, dif_pos hsize, hleft, hright, hw, hb]
  rfl

end Cybernetix.Model.LinearClassifier
