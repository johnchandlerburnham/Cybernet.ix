import Lake
open System Lake DSL

package cybernetix where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.33.1"

/-- Nix supplies a Crane-built archive; ordinary Lake builds invoke Cargo. -/
target cybernetix_rs pkg : FilePath := do
  let output := pkg.buildDir / "lib" / nameToStaticLib "cybernetix_ffi"
  if let some prebuilt ← IO.getEnv "CYBERNETIX_RUST_STATIC_LIB" then
    let archive ← inputBinFile prebuilt
    buildFileAfterDep output archive (fun path => copyFile path output)
      (extraDepTrace := pure <| .ofHash (pureHash prebuilt) "prebuilt Rust archive")
  else
    let sources ← inputDir (pkg.dir / "crates") true fun path =>
      path.extension == some "rs" || path.extension == some "toml"
    let manifests := Job.collectArray #[
      ← inputTextFile (pkg.dir / "Cargo.toml"),
      ← inputTextFile (pkg.dir / "Cargo.lock"),
      ← inputTextFile (pkg.dir / "rust-toolchain.toml"),
      ← inputTextFile (pkg.dir / "lean-toolchain")
    ]
    let deps := sources.zipWith (fun sourceFiles manifestFiles =>
      (sourceFiles, manifestFiles)) manifests
    let args := #["build", "--locked", "--release", "--package", "cybernetix-ffi",
      "--target-dir", (pkg.dir / "target").toString]
    buildFileAfterDep output deps (fun _ => do
      proc { cmd := "cargo", args, cwd := pkg.dir } (quiet := true)
      copyFile (pkg.dir / "target" / "release" / nameToStaticLib "cybernetix_ffi") output
    ) (extraDepTrace := pure <| .ofHash (pureHash args) "Cargo build arguments")

@[default_target]
lean_lib Cybernetix where
  moreLinkObjs := #[cybernetix_rs]

lean_lib CybernetixProofDependencies

@[default_target]
lean_lib CybernetixProofs

lean_lib Tests

lean_exe «cybernetix-ffi-smoke» where
  root := `Tests.FFIMain

@[test_driver]
lean_exe «cybernetix-tests» where
  root := `Tests.Main

lean_exe «cybernetix-sgd» where
  root := `Examples.SGD

lean_exe «cybernetix-mnist» where
  root := `Examples.MNIST
