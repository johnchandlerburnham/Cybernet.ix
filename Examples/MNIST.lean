import Cybernetix.Corpus.MNISTIO
import Cybernetix.Model.LinearArtifact

open Cybernetix Cybernetix.Corpus.MNIST Cybernetix.Model.LinearClassifier

private def usage : String := "Usage:\n\
  cybernetix-mnist fetch [data-directory]\n\
  cybernetix-mnist inspect [data-directory]\n\
  cybernetix-mnist init WEIGHTS\n\
  cybernetix-mnist predict WEIGHTS SHA256 train|validation|test INDEX [data-directory]"

private def directory (args : List String) : IO System.FilePath :=
  match args with
  | [] => pure defaultDirectory
  | [path] => pure path
  | _ => throw <| IO.userError usage

def main (args : List String) : IO Unit := do
  match args with
  | "fetch" :: rest => fetch (← directory rest)
  | "inspect" :: rest =>
    let dir ← directory rest
    let train ← loadRelease dir
    let test ← loadRelease dir true
    IO.println s!"training release: {train.images.count} images; train [0,55000), validation [55000,60000)"
    IO.println s!"test release: {test.images.count} images; held out"
  | ["init", path] =>
    if ← (System.FilePath.mk path).pathExists then
      throw <| IO.userError s!"Refusing to overwrite existing weights: {path}"
    let p : MNIST := ⟨Tensor.Buffer.ofFn fun _ => 0, Tensor.Buffer.ofFn fun _ => 0⟩
    let bytes := p.encode
    IO.FS.writeBinFile path bytes
    IO.println s!"profile: {profile}\nzero-initialized weights: {path}\nsha256: {← sha256 bytes}"
  | "predict" :: weights :: digest :: splitName :: indexText :: rest =>
    let split ← match splitName with
      | "train" => pure Split.train
      | "validation" => pure Split.validation
      | "test" => pure Split.test
      | _ => throw <| IO.userError usage
    let some index := indexText.toNat? | throw <| IO.userError "INDEX must be a natural number"
    let splitIndex : Fin split.size ←
      if h : index < split.size then pure ⟨index, h⟩
      else throw <| IO.userError s!"INDEX must be below {split.size}"
    let wordBytes ← readIdentified weights digest
    let parameters ← match decodeParameters 784 10 wordBytes with
      | .ok p => pure p
      | .error e => throw <| IO.userError s!"Invalid weights: {repr e}"
    let dataset ← loadRelease (← directory rest) (split == .test)
    let releaseIndex := split.releaseIndex splitIndex
    if h : releaseIndex < dataset.images.count then
      let input := dataset.images.input ⟨releaseIndex, h⟩
      let logits ← match forward parameters input with
        | .ok value => pure value
        | .error e => throw <| IO.userError s!"Inference failed: {repr e}"
      let digit := argmax (n := 9) fun i => logits[i]
      let imageFile := if split == .test then testImages else trainImages
      IO.println s!"profile: {profile}\nweights: {digest}\nimages: {imageFile.payloadSHA256}"
      IO.println s!"preprocessing: {preprocessingProfile}"
      IO.println s!"release index: {releaseIndex}\nlogit words: {logits.toArray}\ndigit: {digit.val}"
    else throw <| IO.userError "Release index out of range"
  | _ => throw <| IO.userError usage
