/-! Minimal foreign-function interface used to validate the build and Lean ABI.
These declarations are runtime operations; no refinement theorem is asserted.
-/

namespace Cybernetix.FFI

/-- Copy borrowed bytes into a new Lean-owned byte array. -/
@[extern "cybernetix_copy_bytes"]
opaque copyBytes : @& ByteArray → ByteArray

/-- Encode a 64-bit word as eight little-endian bytes. -/
@[extern "cybernetix_u64_to_le_bytes"]
opaque u64ToLEBytes : UInt64 → ByteArray

end Cybernetix.FFI
