{
  description = "Cybernet.ix: Lean 4 and Rust FFI";

  nixConfig = {
    extra-substituters = [ "https://argumentcomputer.cachix.org" ];
    extra-trusted-public-keys = [
      "argumentcomputer.cachix.org-1:ovhbTx1V56BYDerOWInQvXKXl68LlhNwEA+n7EWk1m4="
    ];
  };

  # Start from Ix's build inputs; flake.lock retains its exact revisions.
  inputs = {
    nixpkgs.follows = "lean4-nix/nixpkgs";
    lean4-nix.url = "github:argumentcomputer/lean4-nix";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "lean4-nix/nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    inputs@{
      flake-parts,
      lean4-nix,
      fenix,
      crane,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        { system, pkgs, ... }:
        let
          lean = lean4-nix.lib.${system}.fromToolchainFile ./lean-toolchain;
          rustToolchain = fenix.packages.${system}.fromToolchainFile {
            file = ./rust-toolchain.toml;
            sha256 = "sha256-p8h3Sl/YRByZfZTAKXdsvF6xEenXKrXSVvpphmZENH4=";
          };
          craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
          ffiEnvironment = {
            LEAN_SYSROOT = "${lean}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          };
          rustArgs = ffiEnvironment // {
            src = craneLib.cleanCargoSource ./.;
            pname = "cybernetix-ffi";
            version = "0.1.0";
            strictDeps = true;
            cargoExtraArgs = "--locked --package cybernetix-ffi";
            buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];
          };
          cargoArtifacts = craneLib.buildDepsOnly rustArgs;
          rustLib = craneLib.buildPackage (
            rustArgs
            // {
              inherit cargoArtifacts;
              # The executable Lean check below tests the actual linked FFI.
              doCheck = false;
            }
          );

          lake2nix = pkgs.callPackage lean4-nix.lake { inherit lean; };
          # Provide locked dependency sources, then build only the modules that
          # our targets import. Building Mathlib's entire shared/static library
          # would compile thousands of unrelated modules for this small proof.
          lakeDeps = builtins.listToAttrs (
            map (dependency: {
              inherit (dependency) name;
              value =
                let
                  repository = builtins.fetchGit {
                    inherit (dependency) url rev;
                    shallow = true;
                  };
                in
                if dependency.subDir == null then repository else "${repository}/${dependency.subDir}";
            }) (pkgs.lib.importJSON ./lake-manifest.json).packages
          );
          lakeOverrides = pkgs.writers.writeJSON "lake-package-overrides.json" (
            (pkgs.lib.importJSON ./lake-manifest.json)
            // {
              packages = map (dependency: {
                inherit (dependency) name inherited;
                type = "path";
                dir = ".lake/packages/${dependency.name}";
              }) (pkgs.lib.importJSON ./lake-manifest.json).packages;
            }
          );
          configureLake = ''
            runHook preConfigure
            mkdir -p .lake/packages
            ${pkgs.lib.concatStringsSep "\n" (
              pkgs.lib.mapAttrsToList (name: source: ''
                # Dependencies can write hashes/traces beside their sources.
                # Retain any compiled artifacts copied by lakeArtifacts.
                if [ ! -d ".lake/packages/${name}" ]; then
                  mkdir -p ".lake/packages/${name}"
                  cp -r --no-preserve=mode ${source}/. ".lake/packages/${name}/"
                fi
              '') lakeDeps
            )}
            if [ ! -e .lake/package-overrides.json ]; then
              ln -s ${lakeOverrides} .lake/package-overrides.json
            fi
            runHook postConfigure
          '';
          proofDependencySrc = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./.gitignore
              ./lakefile.lean
              ./lake-manifest.json
              ./lean-toolchain
              ./CybernetixProofDependencies.lean
            ];
          };
          proofDependencies = lake2nix.mkPackage {
            name = "CybernetixProofDependencies";
            src = proofDependencySrc;
            inherit lakeDeps;
            configurePhase = configureLake;
            LEAN_NUM_THREADS = "4";
          };
          leanSrc = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./.gitignore
              ./lakefile.lean
              ./lake-manifest.json
              ./lean-toolchain
              ./rust-toolchain.toml
              ./Cargo.toml
              ./Cargo.lock
              ./Cybernetix.lean
              ./CybernetixProofs.lean
              ./CybernetixProofDependencies.lean
              ./Tests.lean
              (pkgs.lib.fileset.fileFilter (f: f.hasExt "lean") ./Cybernetix)
              (pkgs.lib.fileset.fileFilter (f: f.hasExt "lean") ./CybernetixProofs)
              (pkgs.lib.fileset.fileFilter (f: f.hasExt "lean") ./Tests)
              (pkgs.lib.fileset.fileFilter (f: f.hasExt "lean") ./Examples)
              (pkgs.lib.fileset.fileFilter (f: f.hasExt "rs" || f.hasExt "toml") ./crates)
            ];
          };
          lakeArgs = {
            src = leanSrc;
            inherit lakeDeps;
            configurePhase = configureLake;
            LEAN_NUM_THREADS = "4";
            # An explicit Lake input replaces Ix's source-patching workaround.
            CYBERNETIX_RUST_STATIC_LIB = "${rustLib}/lib/libcybernetix_ffi.a";
          };
          cybernetixLib = lake2nix.mkPackage (
            lakeArgs
            // {
              name = "Cybernetix";
              buildLibrary = true;
            }
          );
          ffiSmoke = lake2nix.mkPackage (
            lakeArgs
            // {
              name = "cybernetix-ffi-smoke";
              lakeArtifacts = cybernetixLib;
              installArtifacts = false;
              meta.mainProgram = "cybernetix-ffi-smoke";
            }
          );
          sgd = lake2nix.mkPackage (
            lakeArgs
            // {
              name = "cybernetix-sgd";
              installArtifacts = false;
              meta.mainProgram = "cybernetix-sgd";
            }
          );
          tests = lake2nix.mkPackage (
            lakeArgs
            // {
              name = "cybernetix-tests";
              installArtifacts = false;
            }
          );
          proofs = lake2nix.mkPackage (
            lakeArgs
            // {
              name = "CybernetixProofs";
              lakeArtifacts = proofDependencies;
            }
          );
        in
        {
          packages = {
            default = cybernetixLib;
            lean = cybernetixLib;
            rust = rustLib;
            ffi-smoke = ffiSmoke;
            inherit sgd proofs;
          };

          checks = {
            inherit proofs;
            sgd = pkgs.runCommand "cybernetix-tests-check" { } ''
              ${tests}/bin/cybernetix-tests
              touch "$out"
            '';
            ffi-smoke = pkgs.runCommand "cybernetix-ffi-smoke-check" { } ''
              ${ffiSmoke}/bin/cybernetix-ffi-smoke
              touch "$out"
            '';
            clippy = craneLib.cargoClippy (
              rustArgs
              // {
                inherit cargoArtifacts;
                cargoClippyExtraArgs = "--all-targets -- -D warnings";
              }
            );
            rustfmt = craneLib.cargoFmt { inherit (rustArgs) src pname version; };
          };

          devShells.default = pkgs.mkShell (
            ffiEnvironment
            // {
              packages = [
                lean
                rustToolchain
                pkgs.clang
                pkgs.pkg-config
                pkgs.rust-analyzer
                pkgs.git
                pkgs.jujutsu
              ];
            }
          );

          formatter = pkgs.nixfmt-tree;
        };
    };
}
