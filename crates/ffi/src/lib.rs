//! Minimal Lean/Rust boundary. Lean retains ownership of all returned objects.

use lean_ffi::object::{LeanBorrowed, LeanByteArray, LeanOwned};

/// Copy a borrowed Lean byte array into a fresh Lean allocation.
#[unsafe(no_mangle)]
pub extern "C" fn cybernetix_copy_bytes(
    bytes: LeanByteArray<LeanBorrowed<'_>>,
) -> LeanByteArray<LeanOwned> {
    LeanByteArray::from_bytes(bytes.as_bytes())
}

/// Exercise the scalar-to-object ABI with an explicitly little-endian encoding.
#[unsafe(no_mangle)]
pub extern "C" fn cybernetix_u64_to_le_bytes(value: u64) -> LeanByteArray<LeanOwned> {
    LeanByteArray::from_bytes(&value.to_le_bytes())
}
