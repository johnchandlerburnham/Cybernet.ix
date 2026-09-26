import Cybernetix

open Cybernetix.FFI

namespace Cybernetix.Tests.FFI

private def checkBytes (label : String) (actual expected : ByteArray) : IO Unit := do
  unless actual.data == expected.data do
    throw <| IO.userError s!"{label}: unexpected bytes ({actual.size} bytes, expected {expected.size})"

def run : IO Unit := do
  let empty := ByteArray.empty
  checkBytes "empty array" (copyBytes empty) empty

  let payload := ByteArray.mk #[0, 1, 127, 128, 254, 255]
  let copied := copyBytes payload
  checkBytes "binary payload" copied payload
  let extended := copied.push 42
  checkBytes "copy can be extended" extended (ByteArray.mk #[0, 1, 127, 128, 254, 255, 42])
  checkBytes "borrowed input remains usable" payload (ByteArray.mk #[0, 1, 127, 128, 254, 255])

  let large := ByteArray.mk <| (List.range 65536).toArray.map UInt8.ofNat
  for _ in [:32] do
    checkBytes "repeated allocated copy" (copyBytes large) large

  checkBytes "zero word" (u64ToLEBytes 0) (ByteArray.mk #[0, 0, 0, 0, 0, 0, 0, 0])
  checkBytes "byte order" (u64ToLEBytes 0x0123456789abcdef)
    (ByteArray.mk #[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01])
  checkBytes "maximum word" (u64ToLEBytes 0xffffffffffffffff)
    (ByteArray.mk #[255, 255, 255, 255, 255, 255, 255, 255])
  IO.println "Lean/Rust FFI smoke tests passed."

end Cybernetix.Tests.FFI
