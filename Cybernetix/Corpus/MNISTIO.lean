import Cybernetix.Corpus.MNIST
import Cybernetix.Corpus.MNISTManifest

/-! Native acquisition/admission boundary. Curl, gzip and sha256sum are external
utilities; their execution is not certified. Hash the exact bytes returned to
the caller, and keep dataset interpretation in the pure decoder.
-/

namespace Cybernetix.Corpus.MNIST

def defaultDirectory : System.FilePath := "plans/mnist/data"

def sha256 (bytes : ByteArray) : IO String := do
  let child ← do
    let (stdin, child) ← (← IO.Process.spawn {
      cmd := "sha256sum", stdin := .piped, stdout := .piped }).takeStdin
    stdin.write bytes
    stdin.flush
    pure child
  let output ← child.stdout.readToEnd
  unless (← child.wait) == 0 do throw <| IO.userError "sha256sum failed"
  let digest := (output.splitOn " ").headD ""
  unless digest.length == 64 && digest.toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) do
    throw <| IO.userError s!"Invalid sha256sum output: {output}"
  return digest

def checkDigest (bytes : ByteArray) (expected : String) (name : String) : IO Unit := do
  let actual ← sha256 bytes
  unless actual == expected do
    throw <| IO.userError s!"{name}: SHA-256 mismatch; expected {expected}, got {actual}"

def readIdentified (path : System.FilePath) (expected : String) : IO ByteArray := do
  let bytes ← IO.FS.readBinFile path
  checkDigest bytes expected path.toString
  return bytes

def readPayload (directory : System.FilePath) (file : FileIdentity) : IO ByteArray := do
  let bytes ← readIdentified (directory / file.name) file.payloadSHA256
  unless bytes.size == file.payloadBytes do
    throw <| IO.userError s!"{file.name}: expected {file.payloadBytes} bytes, got {bytes.size}"
  return bytes

def fetch (directory : System.FilePath := defaultDirectory) : IO Unit := do
  IO.FS.createDirAll directory
  for file in files do
    let payload := directory / file.name
    let compressed := directory / s!"{file.name}.gz"
    if ← compressed.pathExists then
      let _ ← readIdentified compressed file.compressedSHA256
    if ← payload.pathExists then
      let _ ← readPayload directory file
    else
      unless ← compressed.pathExists do
        let temporary := directory / s!"{file.name}.gz.download"
        let _ ← IO.Process.run { cmd := "curl", args := #["--fail", "--location", "--retry", "3",
          "--output", temporary.toString, file.url] }
        let _ ← readIdentified temporary file.compressedSHA256
        IO.FS.rename temporary compressed
      let child ← IO.Process.spawn {
        cmd := "gzip"
        args := #["--decompress", "--stdout", compressed.toString]
        stdout := .piped }
      let bytes ← child.stdout.readBinToEnd
      unless (← child.wait) == 0 do throw <| IO.userError s!"gzip failed: {compressed}"
      checkDigest bytes file.payloadSHA256 file.name
      unless bytes.size == file.payloadBytes do throw <| IO.userError s!"Wrong length: {file.name}"
      let temporary := directory / s!"{file.name}.decoded"
      IO.FS.writeBinFile temporary bytes
      IO.FS.rename temporary payload
    IO.println s!"verified {file.name}: {file.payloadSHA256}"

def loadRelease (directory : System.FilePath) (test : Bool := false) : IO Dataset := do
  let imageFile := if test then testImages else trainImages
  let labelFile := if test then testLabels else trainLabels
  let imageBytes ← readPayload directory imageFile
  let labelBytes ← readPayload directory labelFile
  let .ok dataset := decode imageBytes labelBytes
    | throw <| IO.userError "Manifest-matching IDX data failed decoding"
  unless dataset.images.count == imageFile.count do
    throw <| IO.userError "Release count does not match manifest"
  return dataset

end Cybernetix.Corpus.MNIST
