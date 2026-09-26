# Development

The project builds executable Lean code, a separate mathematical proof library,
and a Rust static archive linked through Lean's FFI.
Its Nix setup follows Ix: `lean4-nix` supplies Lean/Lake, Fenix supplies Rust,
and Crane builds and caches Rust dependencies. The retained flake input
revisions come from Ix's lockfile.

| Component | Pin |
|---|---|
| Lean | `leanprover/lean4:v4.33.1` |
| Mathlib | `v4.33.1`, revision `0df444a360eaa60ab8c11dca51a86af692955474` |
| Rust | `1.98.1`, default profile including rustfmt and clippy |
| Rust Lean bindings | `argumentcomputer/lean-ffi` at `93c7e52952ae94546be08313f4ff3922984c84d5` |

## Local development

```sh
nix develop
lake build
lake test
lake exe cybernetix-sgd
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The shell includes `jj`, Git, the pinned compilers, clang/libclang, and
rust-analyzer. It sets `LEAN_SYSROOT` to the same Lean toolchain that Lake
uses, so the Rust bindings are generated from the matching `lean/lean.h`.
`LIBCLANG_PATH` identifies the libclang used by bindgen.

`lake build` compiles the default `Cybernetix` and `CybernetixProofs` libraries,
including the SGD proofs and transitive axiom audit. Mathlib is imported only
by the proof library; the executable definitions use Lean core. Lake resolves
the locked dependencies on first use. `lake update` explicitly initializes or
updates dependencies and runs Mathlib's cache setup; check manifest changes
before committing a dependency update.

Linking a native target invokes
`cargo build --locked --release --package cybernetix-ffi` for
the Rust archive. Lake tracks Rust sources, Cargo manifests/lockfile, and
toolchain files. `lake test` builds and runs `cybernetix-tests`, covering the
FFI, SGD replay and rejection, checkpoint/resume, state encoding, and native
binary32 comparisons. `lake exe cybernetix-ffi-smoke` still runs the isolated
ABI probes. `lake exe cybernetix-sgd` prints the toy training result described
in [certified SGD](certified-sgd.md).

These initial externs are linked into compiled executables. Bare `#eval`
requires an additional shared-library loading setup; use the compiled smoke
test for this scaffold.

The repository is colocated Git/Jujutsu: both `.git` and `.jj` are present.
Inside the shell, use `jj status`, `jj diff`, and `jj log`. `.envrc` supports
direnv for users who enable it.

## MNIST data and inference

```sh
lake exe cybernetix-mnist fetch
lake exe cybernetix-mnist inspect
lake exe cybernetix-mnist init plans/mnist/zero.bin
lake exe cybernetix-mnist predict plans/mnist/zero.bin SHA256 train 0
```

Replace `SHA256` with the digest printed by `init`. `nix run .#mnist -- …`
provides the same commands with their external tools. The default dataset
directory is the ignored `plans/mnist/data`; `fetch`, `inspect`, and `predict`
accept an alternate directory as their last argument. The Nix shell supplies
curl, gzip, and sha256sum. Ordinary builds and tests need no dataset download.

The [manifest](../Cybernetix/Corpus/MNISTManifest.lean) pins both compressed
and decoded SHA-256 identities from the [CVDF mirror](https://github.com/cvdfoundation/mnist).
Admission hashes the actual bytes passed to the pure IDX decoder. The decoder
checks magic values, dimensions, exact lengths, paired counts and label bounds.
Training indices are `[0,55000)`, validation indices `[55000,60000)`, and the
10,000-image test release is held out. `predict` takes an index within its
selected split and prints the corresponding release index.

The preprocessing profile `cybernetix.mnist-u8-div255.v1.lean-4.33.1` fixes
row-major byte input, exact byte conversion, and rounded binary32 division by
255. The inference command prints this identity alongside the dense profile.

Weight payload v1 contains 7,840 row-major weight words followed by ten bias
words: exactly 31,400 little-endian bytes. The executable fixes the architecture
and numerical profile, requires the expected weight digest, and rejects
nonfinite parameters. It prints input identity, logit words, and a digit;
zero initialization predicts digit zero by the first-tie rule. This exercises
image-to-prediction and does not demonstrate a learned model. Payloads alone
do not encode a training recipe or constitute a certificate.

Proofs cover packed word round trips, optimized forward/reference equality,
argmax maximality and tie-breaking over the finite encoding order, pixel
preprocessing correspondence, and training/validation index separation. The
kernel checks all 256 pixel conversions for finiteness. Native tests include
malformed IDX and weight payloads, all pixel conversions, and numerical edge
cases. Hashing, acquisition tools, native compilation and hardware remain
explicit execution boundaries; cross-platform agreement still needs another host.

## Nix packages and checks

```sh
nix build                    # Lean library, including shared/static facets
nix build .#rust             # Rust static library
nix run .#ffi-smoke           # Execute the compiled Lean/Rust smoke test
nix run .#sgd                 # Execute the toy training example
nix build .#proofs            # Check the SGD theorems and axiom policy
nix flake check              # Proofs, SGD tests, FFI, clippy, Rust formatting
nix fmt                      # Nix formatting
```

The flake declares Linux and macOS on x86-64 and AArch64. Validation on one
system does not establish a build on the other three. Nix locks the build
inputs; a local Cargo/Lake build still uses the development shell for matching
native tools and header paths.

Validated on x86-64 Linux on 2026-09-26: `lake build`, `lake test`,
`nix run .#sgd`, all five `nix flake check` checks, and `nix fmt -- --ci` passed.
The proof suite and native comparison tests have distinct roles: only the
former establishes the stated logical claims. No cross-architecture native
conformance claim follows from this host's results.

Nix builds Rust once through Crane and passes its archive to Lake using
`CYBERNETIX_RUST_STATIC_LIB`. Lake tracks and copies that input instead of
running Cargo inside the Lean derivation. This avoids editing the Lake source
during a Nix build. The smoke executable reuses the library's Lake artifacts.
The Rust and Lean source filters exclude documentation and local build outputs
from their respective build inputs.

Nix fetches Lean dependency sources at the revisions in the Lake manifest
and supplies local path overrides, so the isolated build needs no Lake
network access. Sources are copied writable because some dependencies store
build hashes beside their sources. The imported Mathlib closure is compiled
in a separate `CybernetixProofDependencies` derivation and reused by the proof
package. This avoids rebuilding Mathlib on ordinary proof edits or compiling
its entire unrelated shared/static library. The local Lake build can use
Mathlib's downloaded artifact cache instead.
The first isolated dependency build took about eleven minutes on the
validation host with four Lean workers; subsequent builds reuse that result.

`flake.lock` pins Nix inputs, `Cargo.lock` pins Rust dependencies, and
`lake-manifest.json` pins Mathlib and its transitive Lean dependencies. Update
toolchains together with the relevant hashes and rerun the proofs and linked
FFI checks. The experimental binary32 profile includes the Lean version;
changing its underlying arithmetic requires an explicit profile review.
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
