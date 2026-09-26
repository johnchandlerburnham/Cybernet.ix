/-! SHA-256 identities observed from the CVDF mirror on 2026-09-26.
Source: https://github.com/cvdfoundation/mnist . Digests identify bytes;
they are not proofs of the dataset's annotations or acquisition software.
-/

namespace Cybernetix.Corpus.MNIST

structure FileIdentity where
  name : String
  compressedSHA256 : String
  payloadSHA256 : String
  count : Nat
  payloadBytes : Nat

def FileIdentity.url (file : FileIdentity) : String :=
  s!"https://storage.googleapis.com/cvdf-datasets/mnist/{file.name}.gz"

def trainImages : FileIdentity := ⟨"train-images-idx3-ubyte",
  "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
  "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db", 60000, 47040016⟩
def trainLabels : FileIdentity := ⟨"train-labels-idx1-ubyte",
  "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
  "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5", 60000, 60008⟩
def testImages : FileIdentity := ⟨"t10k-images-idx3-ubyte",
  "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
  "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7", 10000, 7840016⟩
def testLabels : FileIdentity := ⟨"t10k-labels-idx1-ubyte",
  "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
  "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2", 10000, 10008⟩

def files : Array FileIdentity := #[trainImages, trainLabels, testImages, testLabels]

end Cybernetix.Corpus.MNIST
