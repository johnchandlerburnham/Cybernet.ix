# Development

The initial scaffold builds a Lean library linked to a Rust static archive.
Its Nix setup follows Ix: `lean4-nix` supplies Lean/Lake, Fenix supplies Rust,
and Crane builds and caches Rust dependencies. The retained flake input
revisions come from Ix's lockfile.

| Component | Pin |
|---|---|
| Lean | `leanprover/lean4:v4.33.1` |
| Rust | `1.98.1`, default profile including rustfmt and clippy |
| Rust Lean bindings | `argumentcomputer/lean-ffi` at `93c7e52952ae94546be08313f4ff3922984c84d5` |

## Local development

```sh
nix develop
lake build
lake test
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The shell includes `jj`, Git, the pinned compilers, clang/libclang, and
rust-analyzer. It sets `LEAN_SYSROOT` to the same Lean toolchain that Lake
uses, so the Rust bindings are generated from the matching `lean/lean.h`.
`LIBCLANG_PATH` identifies the libclang used by bindgen.

`lake build` compiles the default `Cybernetix` Lean library. Linking a native
target invokes `cargo build --locked --release --package cybernetix-ffi` for
the Rust archive. Lake tracks Rust sources, Cargo manifests/lockfile, and
toolchain files. `lake test` builds and runs the compiled
`cybernetix-ffi-smoke` executable, exercising both scalar and Lean-object
arguments across the ABI.

These initial externs are linked into compiled executables. Bare `#eval`
requires an additional shared-library loading setup; use the compiled smoke
test for this scaffold.

The repository is colocated Git/Jujutsu: both `.git` and `.jj` are present.
Inside the shell, use `jj status`, `jj diff`, and `jj log`. `.envrc` supports
direnv for users who enable it.

## Nix packages and checks

```sh
nix build                    # Lean library, including shared/static facets
nix build .#rust             # Rust static library
nix run .#ffi-smoke           # Execute the compiled Lean/Rust smoke test
nix flake check              # FFI execution, clippy, and Rust formatting
nix fmt                      # Nix formatting
```

The flake declares Linux and macOS on x86-64 and AArch64. Validation on one
system does not establish a build on the other three. Nix locks the build
inputs; a local Cargo/Lake build still uses the development shell for matching
native tools and header paths.

Validated on x86-64 Linux on 2026-09-26: `lake build`, `lake test`,
`nix run .#ffi-smoke`, and all three `nix flake check` checks passed.

Nix builds Rust once through Crane and passes its archive to Lake using
`CYBERNETIX_RUST_STATIC_LIB`. Lake tracks and copies that input instead of
running Cargo inside the Lean derivation. This avoids editing the Lake source
during a Nix build. The smoke executable reuses the library's Lake artifacts.
The Rust and Lean source filters exclude documentation and local build outputs
from their respective build inputs.

`flake.lock` pins Nix inputs, `Cargo.lock` pins Rust dependencies, and
`lake-manifest.json` currently declares no external Lean packages. Update
toolchains together with the relevant hashes and rerun the linked FFI check.
The Fenix manifest hash in `flake.nix` corresponds to `rust-toolchain.toml`.

## The initial FFI boundary

[`Cybernetix.FFI`](../Cybernetix/FFI.lean) declares two opaque externs supplied
by [`cybernetix-ffi`](../crates/ffi/src/lib.rs):

- `copyBytes` borrows a Lean `ByteArray` and returns a fresh Lean-owned copy.
- `u64ToLEBytes` returns the eight little-endian bytes of a `UInt64`.

Rust allocates returned Lean objects through `lean-ffi`. The smoke test checks
empty and binary payloads, continued use of a borrowed input, repeated larger
allocations, scalar boundary values, and byte order. Failure exits nonzero and
fails the Nix check.

These are build/ABI probes. Their execution tests do not certify the Rust
implementation, native compiler, model arithmetic, or training. The
[portable inference specification](portable-inference.md) describes the
separate refinement and concrete-execution obligations for future backends.
