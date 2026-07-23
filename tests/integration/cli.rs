//! CLI contract tests: exit codes and argument handling.
//!
//! The binary's exit code is the scripting interface to convergence:
//! 0 consistent, 1 error, 2 paradox, 3 epoch exhaustion. These tests pin
//! that contract by invoking the built binary directly.

use std::process::Command;

fn ouro() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ourochronos"))
}

/// Write a programme fixture under a shared temp directory and return its
/// path. Filenames must be unique per test: the directory is shared, so a
/// reused name would let parallel tests clobber each other's fixture.
fn write_temp_program(name: &str, source: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("ouro-cli-tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(name);
    std::fs::write(&path, source).expect("write programme");
    path
}

#[test]
fn help_exits_zero_and_prints_usage() {
    let out = ouro().arg("--help").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Usage: ourochronos"));
    assert!(text.contains("Exit codes"));
    assert!(text.contains("--global"));
    assert!(text.contains("--all-fixed"));
    assert!(text.contains("--recurrent"));
    assert!(text.contains("--verify"));
    assert!(text.contains("--state-bits"));
    assert!(text.contains("--solver-timeout"));
    assert!(text.contains("--loop-unroll"));
    assert!(text.contains("--check"));
    assert!(text.contains("--emit-object"));
    assert!(text.contains("--emit-bytecode"));
    assert!(text.contains("--build"));
    assert!(text.contains("--build-executable"));
    assert!(text.contains("link <output.ourobc> <input.ouroobj>..."));
    assert!(text.contains("run-package <file.ouropkg>"));
}

#[test]
fn cli_uses_fallible_lexing_instead_of_dropping_unknown_characters() {
    let path = write_temp_program("invalid_character.ouro", "1 $ OUTPUT\n");
    let out = ouro().arg(path).output().expect("binary runs");
    let text = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr was: {text}");
    assert!(text.contains("Compile Error"), "stderr was: {text}");
    assert!(text.contains("unknown character '$'"), "stderr was: {text}");
}

#[test]
fn cli_runs_imported_initializers_once_relative_to_the_importer() {
    let library = write_temp_program("cli_relative_library.ouro", "41 OUTPUT\n");
    let root = write_temp_program(
        "cli_relative_root.ouro",
        "IMPORT \"cli_relative_library.ouro\"\n1 OUTPUT\n",
    );
    assert_eq!(library.parent(), root.parent());

    let out = ouro().arg(root).output().expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {text}");
    assert!(text.contains("[41][1]"), "stdout was: {text}");
}

#[test]
fn imported_initializers_participate_in_mandatory_region_validation() {
    let library = write_temp_program(
        "cli_invalid_region_library.ouro",
        "TEMPORAL 0 1 BITS 1 { CLOCK POP }\n",
    );
    let root = write_temp_program(
        "cli_invalid_region_root.ouro",
        "IMPORT \"cli_invalid_region_library.ouro\"\n0 POP\n",
    );
    assert_eq!(library.parent(), root.parent());

    let out = ouro()
        .args([root.to_str().unwrap(), "--memory-cells", "1"])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("Temporal region contract errors"), "{text}");
    assert!(text.contains("CLOCK"), "{text}");
}

#[test]
fn global_solver_finds_a_fixed_state_the_orbit_search_misses() {
    let path = write_temp_program(
        "global_witness.ouro",
        "0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }\n",
    );
    let out = ouro()
        .args([path.to_str().unwrap(), "--global", "--memory-cells", "4"])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("FOUND: a replay-verified solution"),
        "stdout was: {}",
        text
    );
    assert!(text.contains("[0]=42"), "stdout was: {}", text);
}

#[test]
fn global_solver_proves_complete_point_fixed_state_nonexistence() {
    let path = write_temp_program("global_unsat.ouro", "0 ORACLE NOT 0 PROPHECY\n");
    let out = ouro()
        .args([path.to_str().unwrap(), "--global", "--memory-cells", "2"])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(2), "stdout was: {}", text);
    assert!(
        text.contains("PROVEN NO POINT FIXED STATE"),
        "stdout was: {}",
        text
    );
}

#[test]
fn proof_modes_do_not_ignore_gas_because_of_an_unreachable_loop() {
    let path = write_temp_program(
        "proof_gas_dead_loop.ouro",
        "PROPERTY zero { ALL_FIXED CELL 0 EQ 0; }\n\
         0 ORACLE NOT 0 PROPHECY HALT WHILE { 0 } { NOP }\n",
    );

    for mode in ["--global", "--all-fixed", "--verify"] {
        let out = ouro()
            .args([
                path.to_str().unwrap(),
                mode,
                "--memory-cells",
                "1",
                "--max-inst",
                "1",
            ])
            .output()
            .expect("binary runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.status.code(), Some(3), "{mode}: {text}");
        assert!(text.contains("UNKNOWN"), "{mode}: {text}");
        assert!(!text.contains("PROVEN"), "{mode}: {text}");
        assert!(!text.contains("VACUOUS"), "{mode}: {text}");
    }
}

#[test]
fn temporal_region_contract_covers_quotes_and_unused_procedures() {
    let quoted = write_temp_program(
        "quoted_invalid_region.ouro",
        "[ TEMPORAL 0 1 BITS 1 { CLOCK POP } ] EXEC\n",
    );
    let unused = write_temp_program(
        "unused_procedure_invalid_region.ouro",
        "PROCEDURE bad { TEMPORAL 0 1 BITS 1 { CLOCK POP } } 0 POP\n",
    );

    for (path, extra) in [
        (&quoted, None),
        (&quoted, Some("--typecheck")),
        (&unused, None),
        (&unused, Some("--typecheck")),
    ] {
        let mut command = ouro();
        command.args([path.to_str().unwrap(), "--memory-cells", "1"]);
        if let Some(flag) = extra {
            command.arg(flag);
        }
        let out = command.output().expect("binary runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.status.code(), Some(1), "{text}");
        assert!(text.contains("Temporal region contract errors"), "{text}");
        assert!(text.contains("CLOCK"), "{text}");
    }
}

#[test]
fn typecheck_retains_unused_procedure_effect_contracts() {
    let path = write_temp_program(
        "unused_pure_procedure.ouro",
        "PROCEDURE bad PURE { CLOCK POP } 0 POP\n",
    );
    let out = ouro()
        .args([path.to_str().unwrap(), "--typecheck"])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("Effect"), "{text}");
    assert!(text.contains("bad"), "{text}");
    assert!(text.contains("Type errors found"), "{text}");
}

#[test]
fn semantic_checks_are_mandatory_without_the_typecheck_flag() {
    let underflow = write_temp_program("mandatory_underflow.ouro", "1 SWAP\n");
    let underflow_out = ouro().arg(underflow).output().expect("binary runs");
    let underflow_text = format!(
        "{}{}",
        String::from_utf8_lossy(&underflow_out.stdout),
        String::from_utf8_lossy(&underflow_out.stderr)
    );
    assert_eq!(underflow_out.status.code(), Some(1), "{underflow_text}");
    assert!(
        underflow_text.contains("mandatory structural stack analysis failed"),
        "{underflow_text}"
    );

    let effect = write_temp_program(
        "mandatory_effect_contract.ouro",
        "PROCEDURE bad PURE { CLOCK POP } 0 POP\n",
    );
    let effect_out = ouro().arg(effect).output().expect("binary runs");
    let effect_text = format!(
        "{}{}",
        String::from_utf8_lossy(&effect_out.stdout),
        String::from_utf8_lossy(&effect_out.stderr)
    );
    assert_eq!(effect_out.status.code(), Some(1), "{effect_text}");
    assert!(effect_text.contains("Effect Violations"), "{effect_text}");
    assert!(effect_text.contains("procedure 'bad'"), "{effect_text}");
}

#[test]
fn unavailable_execution_features_fail_before_legacy_runtime_paths() {
    let foreign = write_temp_program("unlinked_ffi.ouro", "0 FFI_CALL\n");
    let foreign_out = ouro().arg(foreign).output().expect("binary runs");
    let foreign_text = format!(
        "{}{}",
        String::from_utf8_lossy(&foreign_out.stdout),
        String::from_utf8_lossy(&foreign_out.stderr)
    );
    assert_eq!(foreign_out.status.code(), Some(1), "{foreign_text}");
    assert!(
        foreign_text.contains("foreign execution is not linked"),
        "{foreign_text}"
    );

    let recursive = write_temp_program(
        "recursive_procedure_runtime_gate.ouro",
        "PROCEDURE recurse { recurse } recurse\n",
    );
    let recursive_out = ouro().arg(recursive).output().expect("binary runs");
    let recursive_text = format!(
        "{}{}",
        String::from_utf8_lossy(&recursive_out.stdout),
        String::from_utf8_lossy(&recursive_out.stderr)
    );
    assert_eq!(recursive_out.status.code(), Some(1), "{recursive_text}");
    assert!(
        recursive_text.contains("call depth limit"),
        "{recursive_text}"
    );

    let recursive_quote =
        write_temp_program("recursive_quote_legacy_gate.ouro", "[ 0 EXEC ] EXEC\n");
    let quote_out = ouro()
        .args([recursive_quote.to_str().unwrap(), "--diagnostic"])
        .output()
        .expect("binary runs");
    let quote_text = format!(
        "{}{}",
        String::from_utf8_lossy(&quote_out.stdout),
        String::from_utf8_lossy(&quote_out.stderr)
    );
    assert_eq!(quote_out.status.code(), Some(1), "{quote_text}");
    assert!(quote_text.contains("call depth limit"), "{quote_text}");
}

#[test]
fn typed_scalar_foreign_declarations_compile_but_require_an_explicit_host() {
    let source = write_temp_program(
        "typed_scalar_foreign.ouro",
        "FOREIGN \"process\" {\n\
         PROC add_signed AS \"host_add\" (a: u64, b: i64 -- i64) PURE;\n\
         }\n\
         50 8 add_signed OUTPUT\n",
    );
    let checked = ouro()
        .arg(&source)
        .arg("--check")
        .output()
        .expect("binary runs");
    assert_eq!(checked.status.code(), Some(0), "{checked:?}");

    let artifact = std::env::temp_dir().join("ouro-cli-typed-scalar-foreign.ourobc");
    let _ = std::fs::remove_file(&artifact);
    let emitted = ouro()
        .arg(&source)
        .args(["--emit-bytecode", artifact.to_str().unwrap()])
        .output()
        .expect("binary runs");
    assert_eq!(emitted.status.code(), Some(0), "{emitted:?}");
    let linked = ourochronos::BytecodeProgram::from_bytes(
        &std::fs::read(&artifact).expect("read typed foreign artifact"),
    )
    .expect("decode typed foreign artifact");
    assert_eq!(linked.foreigns.len(), 1);
    assert_eq!(linked.foreigns[0].library, "process");
    assert_eq!(linked.foreigns[0].symbol, "host_add");
    assert_eq!(
        linked.foreigns[0].parameters,
        [
            ourochronos::ForeignScalarType::U64,
            ourochronos::ForeignScalarType::I64
        ]
    );

    let run = ouro().arg(&source).output().expect("binary runs");
    let run_text = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.status.code(), Some(1), "{run_text}");
    assert!(
        run_text.contains("foreign execution is not linked"),
        "{run_text}"
    );

    let package = std::env::temp_dir().join("ouro-cli-typed-scalar-foreign.ouropkg");
    let _ = std::fs::remove_file(&package);
    let built = ouro()
        .arg(source)
        .args(["--build", package.to_str().unwrap()])
        .output()
        .expect("binary runs");
    let build_text = format!(
        "{}{}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );
    assert_eq!(built.status.code(), Some(1), "{build_text}");
    assert!(
        build_text.contains("does not declare foreign host dependencies"),
        "{build_text}"
    );
    assert!(!package.exists());
}

#[test]
fn forward_procedure_calls_resolve_before_body_parsing() {
    let path = write_temp_program(
        "forward_procedure_call.ouro",
        "PROCEDURE first { second } PROCEDURE second { 42 } first OUTPUT\n",
    );
    let out = ouro().arg(path).output().expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("[42]"), "{text}");
}

#[test]
fn check_compile_and_build_actions_use_validated_portable_artifacts() {
    let source = write_temp_program("portable_build_source.ouro", "40 2 ADD OUTPUT\n");
    let directory = source.parent().unwrap();
    let object_path = directory.join("portable_build_output.ouroobj");
    let objects_directory = directory.join("portable_build_objects");
    let bytecode_path = directory.join("portable_build_output.ourobc");
    let linked_path = directory.join("portable_build_output_linked.ourobc");
    let package_path = directory.join("portable_build_output.ouropkg");
    let rebuilt_package_path = directory.join("portable_build_output_rebuilt.ouropkg");
    let launcher_path = directory.join("portable_build_output");
    let rebuilt_launcher_path = directory.join("portable_build_output_rebuilt");
    let _ = std::fs::remove_file(&object_path);
    let _ = std::fs::remove_dir_all(&objects_directory);
    let _ = std::fs::remove_file(&bytecode_path);
    let _ = std::fs::remove_file(&linked_path);
    let _ = std::fs::remove_file(&package_path);
    let _ = std::fs::remove_file(&rebuilt_package_path);
    let _ = std::fs::remove_file(&launcher_path);
    let _ = std::fs::remove_file(&rebuilt_launcher_path);

    let checked = ouro()
        .args([source.to_str().unwrap(), "--check"])
        .output()
        .expect("binary runs");
    let checked_text = String::from_utf8_lossy(&checked.stdout);
    assert_eq!(checked.status.code(), Some(0), "{checked_text}");
    assert!(checked_text.contains("Check succeeded"), "{checked_text}");

    let object_build = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-object",
            object_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(
        object_build.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&object_build.stderr)
    );
    let object = std::fs::read(&object_path).expect("object artifact exists");
    ourochronos::ObjectModule::from_bytes(&object).expect("object validates");

    let objects_build = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-objects",
            objects_directory.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(
        objects_build.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&objects_build.stderr)
    );
    let mut object_paths = std::fs::read_dir(&objects_directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    object_paths.sort();

    let compiled = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-bytecode",
            bytecode_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(
        compiled.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytecode = std::fs::read(&bytecode_path).expect("bytecode artifact exists");
    ourochronos::BytecodeProgram::from_bytes(&bytecode).expect("artifact validates");

    let mut link_command = ouro();
    link_command.arg("link").arg(&linked_path);
    for path in &object_paths {
        link_command.arg(path);
    }
    let linked = link_command.output().expect("binary runs");
    assert_eq!(
        linked.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    assert_eq!(
        std::fs::read(&linked_path).expect("standalone link output exists"),
        bytecode
    );

    let built = ouro()
        .args([
            source.to_str().unwrap(),
            "--build",
            package_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(
        built.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let package = std::fs::read(&package_path).expect("package artifact exists");
    let decoded = ourochronos::PortablePackage::from_bytes(&package).expect("package validates");
    assert_eq!(decoded.manifest.name, "portable_build_source");
    assert_eq!(decoded.program.to_bytes().unwrap(), bytecode);

    let rebuilt = ouro()
        .args([
            source.to_str().unwrap(),
            "--build",
            rebuilt_package_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(
        rebuilt.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&rebuilt.stderr)
    );
    assert_eq!(
        std::fs::read(&rebuilt_package_path).expect("rebuilt package exists"),
        package,
        "identical source and configuration must rebuild byte-for-byte"
    );

    let executed = ouro()
        .args(["run-package", package_path.to_str().unwrap()])
        .output()
        .expect("packaged program runs");
    let executed_text = format!(
        "{}{}",
        String::from_utf8_lossy(&executed.stdout),
        String::from_utf8_lossy(&executed.stderr)
    );
    assert_eq!(executed.status.code(), Some(0), "{executed_text}");
    assert!(executed_text.contains("[42]"), "{executed_text}");

    for path in [&launcher_path, &rebuilt_launcher_path] {
        let built = ouro()
            .args([
                source.to_str().unwrap(),
                "--build-executable",
                path.to_str().unwrap(),
            ])
            .output()
            .expect("native launcher builds");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&built.stdout),
            String::from_utf8_lossy(&built.stderr)
        );
        assert_eq!(built.status.code(), Some(0), "{text}");
    }
    assert_eq!(
        std::fs::read(&launcher_path).expect("native launcher exists"),
        std::fs::read(&rebuilt_launcher_path).expect("rebuilt native launcher exists"),
        "identical source and runtime must rebuild a byte-identical native launcher"
    );

    let launched = Command::new(&launcher_path)
        .output()
        .expect("native launcher runs without a source or sidecar package");
    let launched_text = format!(
        "{}{}",
        String::from_utf8_lossy(&launched.stdout),
        String::from_utf8_lossy(&launched.stderr)
    );
    assert_eq!(launched.status.code(), Some(0), "{launched_text}");
    assert!(launched_text.contains("[42]"), "{launched_text}");
}

#[test]
fn imported_graph_emits_real_per_source_objects_without_a_fat_fallback() {
    let library = write_temp_program("object_graph_library.ouro", "PROC answer { 42 }\n");
    let source = write_temp_program(
        "object_graph_main.ouro",
        "IMPORT \"object_graph_library.ouro\"\nanswer OUTPUT\n",
    );
    assert_eq!(library.parent(), source.parent());
    let directory = source.parent().unwrap().join("per-source-objects");
    let linked_path = source.parent().unwrap().join("per-source-linked.ourobc");
    let direct_path = source.parent().unwrap().join("per-source-direct.ourobc");
    let fat_path = source.parent().unwrap().join("forbidden-fat.ouroobj");
    let _ = std::fs::remove_dir_all(&directory);
    for path in [&linked_path, &direct_path, &fat_path] {
        let _ = std::fs::remove_file(path);
    }

    let entry_emitted = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-object",
            fat_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let entry_text = format!(
        "{}{}",
        String::from_utf8_lossy(&entry_emitted.stdout),
        String::from_utf8_lossy(&entry_emitted.stderr)
    );
    assert_eq!(entry_emitted.status.code(), Some(0), "{entry_text}");
    let entry_object = ourochronos::ObjectModule::from_bytes(&std::fs::read(&fat_path).unwrap())
        .expect("entry source object validates");
    assert!(!entry_object.imports.is_empty());
    let unresolved_path = source
        .parent()
        .unwrap()
        .join("entry-only-unresolved.ourobc");
    let entry_only = ouro()
        .args([
            "link",
            unresolved_path.to_str().unwrap(),
            fat_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(entry_only.status.code(), Some(1));

    let emitted = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-objects",
            directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        emitted.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&emitted.stdout),
        String::from_utf8_lossy(&emitted.stderr)
    );
    let mut paths = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths.len(), 3, "source modules plus the linked prelude");
    let objects = paths
        .iter()
        .map(|path| ourochronos::ObjectModule::from_bytes(&std::fs::read(path).unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(objects.iter().any(|object| !object.exports.is_empty()));
    assert!(objects.iter().any(|object| !object.imports.is_empty()));
    let stale_safe = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-objects",
            directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let stale_safe_text = format!(
        "{}{}",
        String::from_utf8_lossy(&stale_safe.stdout),
        String::from_utf8_lossy(&stale_safe.stderr)
    );
    assert_eq!(stale_safe.status.code(), Some(1), "{stale_safe_text}");
    assert!(
        stale_safe_text.contains("must be empty"),
        "{stale_safe_text}"
    );

    let mut link_command = ouro();
    link_command.arg("link").arg(&linked_path);
    for path in &paths {
        link_command.arg(path);
    }
    let linked = link_command.output().unwrap();
    assert_eq!(
        linked.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&linked.stdout),
        String::from_utf8_lossy(&linked.stderr)
    );
    let direct = ouro()
        .args([
            source.to_str().unwrap(),
            "--emit-bytecode",
            direct_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(direct.status.code(), Some(0), "{direct:?}");
    assert_eq!(
        std::fs::read(&linked_path).unwrap(),
        std::fs::read(&direct_path).unwrap()
    );
}

#[test]
fn runtime_global_package_declares_solver_and_never_falls_back_to_orbit() {
    let directory = std::env::temp_dir().join("ouro-cli-tests/runtime-global-package");
    std::fs::create_dir_all(&directory).unwrap();
    let package_path = directory.join("circular.ouropkg");
    let _ = std::fs::remove_file(&package_path);

    let built = ouro()
        .args([
            "examples/case_studies/circular_dataflow.ouro",
            "--build",
            package_path.to_str().unwrap(),
            "--runtime-global-package",
            "--memory-cells",
            "2",
        ])
        .output()
        .unwrap();
    assert_eq!(
        built.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );
    let package =
        ourochronos::PortablePackage::from_bytes(&std::fs::read(&package_path).unwrap()).unwrap();
    assert_eq!(
        package.manifest.resolution_policy,
        ourochronos::PackageResolutionPolicy::RuntimeGlobalPoint
    );
    assert_eq!(
        package.manifest.solver_dependency,
        Some(ourochronos::CURRENT_Z3_SOLVER_DEPENDENCY)
    );
    assert!(package.witness.is_none());

    let run = ouro()
        .args(["run-package", package_path.to_str().unwrap()])
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.status.code(), Some(0), "{text}");
    assert!(text.contains("declared Z3 contract"), "{text}");
    assert!(text.contains("[3][2]"), "{text}");
}

#[test]
fn exact_file_capabilities_freeze_reads_and_commit_selected_writes() {
    let directory = std::env::temp_dir().join("ouro-cli-tests/file-capabilities");
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("input.bin");
    let output = directory.join("output.bin");
    std::fs::write(&input, b"abc").unwrap();
    std::fs::write(&output, b"unchanged").unwrap();
    let source_path = |path: &std::path::Path| {
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    };

    let read_source = write_temp_program(
        "capability_read.ouro",
        &format!("\"{}\" FILE_SIZE OUTPUT\n", source_path(&input)),
    );
    let denied_read = ouro().arg(&read_source).output().unwrap();
    assert_eq!(denied_read.status.code(), Some(1));
    let write_only_read = ouro()
        .args([
            read_source.to_str().unwrap(),
            "--allow-file-write",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(write_only_read.status.code(), Some(1));
    let allowed_read = ouro()
        .args([
            read_source.to_str().unwrap(),
            "--allow-file-read",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        allowed_read.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&allowed_read.stdout),
        String::from_utf8_lossy(&allowed_read.stderr)
    );
    assert!(String::from_utf8_lossy(&allowed_read.stdout).contains("[3]"));

    let write_source = write_temp_program(
        "capability_write.ouro",
        &format!(
            "\"{}\" 26 FILE_OPEN 65 1 BUFFER_FROM_STACK FILE_WRITE POP FILE_CLOSE 0 ORACLE POP 1 0 PROPHECY\n",
            source_path(&output)
        ),
    );
    let denied_write = ouro().arg(&write_source).output().unwrap();
    assert_eq!(denied_write.status.code(), Some(1));
    assert_eq!(std::fs::read(&output).unwrap(), b"unchanged");
    let allowed_write = ouro()
        .args([
            write_source.to_str().unwrap(),
            "--allow-file-write",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        allowed_write.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&allowed_write.stdout),
        String::from_utf8_lossy(&allowed_write.stderr)
    );
    assert_eq!(std::fs::read(&output).unwrap(), b"A");
}

#[test]
fn cli_rejects_oversized_external_inputs_before_reading_them() {
    let directory = std::env::temp_dir().join("ouro-cli-tests/bounded-external-inputs");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let program = write_temp_program("bounded_external_inputs.ouro", "0 POP\n");
    let defaults = ourochronos::BytecodeVmConfig::default();

    let sparse = |name: &str, length: u64| {
        let path = directory.join(name);
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(length).unwrap();
        path
    };
    let file_input = sparse(
        "oversized-file.bin",
        u64::try_from(defaults.max_collection_items).unwrap() + 1,
    );
    let result = ouro()
        .args([
            program.to_str().unwrap(),
            "--allow-file-read",
            file_input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("bounded read limit"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let endpoint_input = sparse(
        "oversized-endpoint.bin",
        u64::try_from(defaults.max_collection_items).unwrap() + 1,
    );
    let endpoint_spec = format!("frozen.test:42={}", endpoint_input.display());
    let result = ouro()
        .args([program.to_str().unwrap(), "--network-input", &endpoint_spec])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("bounded read limit"));

    let descriptor_limit = defaults
        .max_collection_items
        .saturating_mul(2)
        .saturating_add(4_096);
    let process_input = sparse(
        "oversized-process.ouroprocess",
        u64::try_from(descriptor_limit).unwrap() + 1,
    );
    let result = ouro()
        .args([
            program.to_str().unwrap(),
            "--allow-process",
            process_input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("bounded read limit"));

    let package = sparse(
        "oversized.ouropkg",
        u64::try_from(ourochronos::MAX_PACKAGE_BYTES).unwrap() + 1,
    );
    let result = ouro()
        .args(["run-package", package.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("bounded read limit"));

    let object = sparse(
        "oversized.ouroobj",
        u64::try_from(ourochronos::MAX_OBJECT_BYTES).unwrap() + 1,
    );
    let linked = directory.join("linked.ourobc");
    let result = ouro()
        .args(["link", linked.to_str().unwrap(), object.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("bounded read limit"));

    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn system_capabilities_freeze_network_and_commit_send_process_and_sleep_once() {
    use std::io::Read;

    let directory = std::env::temp_dir().join("ouro-cli-tests/system-capabilities");
    std::fs::create_dir_all(&directory).unwrap();

    let receive_tape = directory.join("receive.bin");
    std::fs::write(&receive_tape, b"xy").unwrap();
    let receive_source = write_temp_program(
        "capability_network_receive.ouro",
        "\"frozen.test\" 42 TCP_CONNECT 2 SOCKET_RECV OUTPUT\n",
    );
    let denied_receive = ouro().arg(&receive_source).output().unwrap();
    assert_eq!(denied_receive.status.code(), Some(1));
    let receive_spec = format!("frozen.test:42={}", receive_tape.display());
    let allowed_receive = ouro()
        .args([
            receive_source.to_str().unwrap(),
            "--network-input",
            &receive_spec,
        ])
        .output()
        .unwrap();
    assert_eq!(
        allowed_receive.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&allowed_receive.stdout),
        String::from_utf8_lossy(&allowed_receive.stderr)
    );
    assert!(String::from_utf8_lossy(&allowed_receive.stdout).contains("[2]"));

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let receiver = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).unwrap();
        bytes
    });
    let send_source = write_temp_program(
        "capability_network_send.ouro",
        &format!(
            "\"{}\" {} TCP_CONNECT 65 1 BUFFER_FROM_STACK SOCKET_SEND POP SOCKET_CLOSE 0 ORACLE POP 1 0 PROPHECY\n",
            endpoint.ip(),
            endpoint.port()
        ),
    );
    let sent = ouro()
        .args([
            send_source.to_str().unwrap(),
            "--allow-network-send",
            &endpoint.to_string(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        sent.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&sent.stdout),
        String::from_utf8_lossy(&sent.stderr)
    );
    assert_eq!(receiver.join().unwrap(), b"A");

    let process_marker = directory.join("process-marker.txt");
    let _ = std::fs::remove_file(&process_marker);
    let command = format!("printf X >> {}", process_marker.display());
    let descriptor = directory.join("process.ouroprocess");
    let mut descriptor_bytes =
        format!("OUROPROCESS/1\n0\n{}\n{}", command.len(), command).into_bytes();
    descriptor_bytes.extend_from_slice(b"frozen-output");
    std::fs::write(&descriptor, descriptor_bytes).unwrap();
    let process_source = write_temp_program(
        "capability_process.ouro",
        &format!(
            "\"{}\" PROC_EXEC OUTPUT 0 ORACLE POP 1 0 PROPHECY\n",
            command.replace('\\', "\\\\").replace('"', "\\\"")
        ),
    );
    let denied_process = ouro().arg(&process_source).output().unwrap();
    assert_eq!(denied_process.status.code(), Some(1));
    assert!(!process_marker.exists());
    let allowed_process = ouro()
        .args([
            process_source.to_str().unwrap(),
            "--allow-process",
            descriptor.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        allowed_process.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&allowed_process.stdout),
        String::from_utf8_lossy(&allowed_process.stderr)
    );
    assert_eq!(std::fs::read(&process_marker).unwrap(), b"X");

    let deutsch_process = ouro()
        .args([
            process_source.to_str().unwrap(),
            "--deutsch",
            "--allow-process",
            descriptor.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        deutsch_process.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&deutsch_process.stdout),
        String::from_utf8_lossy(&deutsch_process.stderr)
    );
    assert_eq!(std::fs::read(&process_marker).unwrap(), b"XX");

    let sleep_source = write_temp_program(
        "capability_sleep.ouro",
        "0 SLEEP 0 ORACLE POP 1 0 PROPHECY\n",
    );
    assert_eq!(
        ouro().arg(&sleep_source).output().unwrap().status.code(),
        Some(1)
    );
    let allowed_sleep = ouro()
        .args([sleep_source.to_str().unwrap(), "--allow-sleep-ms", "0"])
        .output()
        .unwrap();
    assert_eq!(
        allowed_sleep.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&allowed_sleep.stdout),
        String::from_utf8_lossy(&allowed_sleep.stderr)
    );
}

#[test]
fn cli_action_mode_executes_recursive_calls_and_quotations_as_bytecode() {
    let source = write_temp_program(
        "bytecode_action_cli.ouro",
        "PROCEDURE choose { 0 ORACLE DUP [ DUP OUTPUT ] EXEC 0 PROPHECY }\nchoose\n",
    );
    let run = ouro()
        .args([
            source.to_str().unwrap(),
            "--action",
            "--seeds",
            "4",
            "--memory-cells",
            "2",
        ])
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.status.code(), Some(0), "{text}");
    assert!(text.contains("ACTION-GUIDED"), "{text}");
    assert!(text.contains('['), "{text}");
}

#[test]
fn build_rejects_programs_the_package_runtime_cannot_execute() {
    let source = write_temp_program("unpackaged_input.ouro", "INPUT OUTPUT\n");
    let package = source.parent().unwrap().join("unpackaged_input.ouropkg");
    let _ = std::fs::remove_file(&package);
    let out = ouro()
        .args([
            source.to_str().unwrap(),
            "--build",
            package.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("does not support primitive INPUT"), "{text}");
    assert!(!package.exists(), "failed build must not leave a package");
}

#[test]
fn strongest_three_cross_the_compiler_solver_replay_and_deployment_boundary() {
    let directory = std::env::temp_dir().join("ouro-cli-tests/strongest-three");
    std::fs::create_dir_all(&directory).unwrap();

    let mutual = "examples/case_studies/mutual_exclusion.ouro";
    let mutual_objects = directory.join("mutual-objects");
    let mutual_bytecode = directory.join("mutual.ourobc");
    let mutual_evidence = directory.join("mutual.json");
    let mutual_package = directory.join("mutual.ouropkg");
    let _ = std::fs::remove_dir_all(&mutual_objects);
    for path in [&mutual_bytecode, &mutual_evidence, &mutual_package] {
        let _ = std::fs::remove_file(path);
    }

    let compiled = ouro()
        .args([
            mutual,
            "--emit-objects",
            mutual_objects.to_str().unwrap(),
            "--memory-cells",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(compiled.status.code(), Some(0), "{:?}", compiled);
    let mut object_paths = std::fs::read_dir(&mutual_objects)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    object_paths.sort();
    let mut link_command = ouro();
    link_command.arg("link").arg(&mutual_bytecode);
    for path in &object_paths {
        link_command.arg(path);
    }
    let linked = link_command.output().unwrap();
    assert_eq!(linked.status.code(), Some(0), "{:?}", linked);
    ourochronos::BytecodeProgram::from_bytes(&std::fs::read(&mutual_bytecode).unwrap()).unwrap();

    let solved = ouro()
        .args([
            mutual,
            "--global",
            "--memory-cells",
            "1",
            "--artifact",
            mutual_evidence.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let solved_text = String::from_utf8_lossy(&solved.stdout);
    assert_eq!(solved.status.code(), Some(0), "{solved_text}");
    assert!(solved_text.contains("replay-verified"), "{solved_text}");
    assert!(std::fs::read_to_string(&mutual_evidence)
        .unwrap()
        .contains("\"replay_verified\":true"));

    let verified = ouro()
        .args([mutual, "--verify", "--memory-cells", "1"])
        .output()
        .unwrap();
    let verified_text = String::from_utf8_lossy(&verified.stdout);
    assert_eq!(verified.status.code(), Some(0), "{verified_text}");
    assert!(verified_text.contains("PROVEN mutual_exclusion"));

    let built = ouro()
        .args([
            mutual,
            "--build",
            mutual_package.to_str().unwrap(),
            "--embed-global-witness",
            "--memory-cells",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(built.status.code(), Some(0), "{:?}", built);
    let decoded =
        ourochronos::PortablePackage::from_bytes(&std::fs::read(&mutual_package).unwrap()).unwrap();
    assert!(decoded.witness.is_some());
    let executed = ouro()
        .args(["run-package", mutual_package.to_str().unwrap()])
        .output()
        .unwrap();
    let executed_text = String::from_utf8_lossy(&executed.stdout);
    assert_eq!(executed.status.code(), Some(0), "{executed_text}");
    assert!(
        executed_text.contains("[1]") || executed_text.contains("[2]"),
        "{executed_text}"
    );

    let circular = "examples/case_studies/circular_dataflow.ouro";
    let circular_launcher = directory.join("circular-native");
    let _ = std::fs::remove_file(&circular_launcher);
    let built = ouro()
        .args([
            circular,
            "--build-executable",
            circular_launcher.to_str().unwrap(),
            "--memory-cells",
            "2",
        ])
        .output()
        .unwrap();
    assert_eq!(built.status.code(), Some(0), "{:?}", built);
    let executed = Command::new(&circular_launcher).output().unwrap();
    let executed_text = String::from_utf8_lossy(&executed.stdout);
    assert_eq!(executed.status.code(), Some(0), "{executed_text}");
    assert!(executed_text.contains("[3][2]"), "{executed_text}");

    let game = "examples/case_studies/retrocausal_game.ouro";
    let game_dot = directory.join("game.dot");
    let game_launcher = directory.join("game-native");
    let _ = std::fs::remove_file(&game_dot);
    let _ = std::fs::remove_file(&game_launcher);
    let recurrent = ouro()
        .args([
            game,
            "--recurrent",
            "--memory-cells",
            "1",
            "--state-bits",
            "2",
            "--dot",
            game_dot.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let recurrent_text = String::from_utf8_lossy(&recurrent.stdout);
    assert_eq!(recurrent.status.code(), Some(0), "{recurrent_text}");
    assert!(
        recurrent_text.contains("Class 0: period 2"),
        "{recurrent_text}"
    );
    assert!(std::fs::read_to_string(&game_dot)
        .unwrap()
        .contains("digraph"));
    let built = ouro()
        .args([
            game,
            "--build-executable",
            game_launcher.to_str().unwrap(),
            "--memory-cells",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(built.status.code(), Some(0), "{:?}", built);
    let executed = Command::new(&game_launcher).output().unwrap();
    let executed_text = String::from_utf8_lossy(&executed.stdout);
    assert_eq!(executed.status.code(), Some(2), "{executed_text}");
    assert!(executed_text.contains("OSCILLATION"), "{executed_text}");

    let _ = std::fs::remove_dir_all(mutual_objects);
    for path in [
        mutual_bytecode,
        mutual_evidence,
        mutual_package,
        circular_launcher,
        game_dot,
        game_launcher,
    ] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn all_fixed_mode_proves_uniqueness_and_exhibits_ambiguity() {
    let unique_path = write_temp_program(
        "all_fixed_unique.ouro",
        "0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }\n",
    );
    let unique = ouro()
        .args([
            unique_path.to_str().unwrap(),
            "--all-fixed",
            "--memory-cells",
            "2",
        ])
        .output()
        .expect("binary runs");
    let unique_text = String::from_utf8_lossy(&unique.stdout);
    assert_eq!(unique.status.code(), Some(0), "stdout was: {}", unique_text);
    assert!(
        unique_text.contains("PROVEN UNIQUE"),
        "stdout was: {}",
        unique_text
    );

    let ambiguous_path = write_temp_program("all_fixed_ambiguous.ouro", "0 ORACLE 0 PROPHECY\n");
    let ambiguous = ouro()
        .args([
            ambiguous_path.to_str().unwrap(),
            "--all-fixed",
            "--memory-cells",
            "2",
        ])
        .output()
        .expect("binary runs");
    let ambiguous_text = String::from_utf8_lossy(&ambiguous.stdout);
    assert_eq!(
        ambiguous.status.code(),
        Some(2),
        "stdout was: {}",
        ambiguous_text
    );
    assert!(
        ambiguous_text.contains("AMBIGUOUS"),
        "stdout was: {}",
        ambiguous_text
    );
    assert!(
        ambiguous_text.contains("Both witnesses passed"),
        "stdout was: {}",
        ambiguous_text
    );
}

#[test]
fn recurrent_mode_completely_classifies_a_closed_small_domain() {
    let path = write_temp_program(
        "recurrent_four_cycle.ouro",
        "0 ORACLE 1 ADD 3 AND 0 PROPHECY\n",
    );
    let out = ouro()
        .args([
            path.to_str().unwrap(),
            "--recurrent",
            "--memory-cells",
            "1",
            "--state-bits",
            "2",
        ])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("Classified all 4 states"),
        "stdout was: {}",
        text
    );
    assert!(text.contains("period 4"), "stdout was: {}", text);
    assert!(
        text.contains("point fixed states: 0"),
        "stdout was: {}",
        text
    );
}

#[test]
fn recurrent_mode_executes_linked_bytecode_quotation_calls() {
    let path = write_temp_program(
        "recurrent_bytecode_quote.ouro",
        "[ 0 ORACLE NOT 0 PROPHECY ] EXEC\n",
    );
    let out = ouro()
        .args([
            path.to_str().unwrap(),
            "--recurrent",
            "--memory-cells",
            "1",
            "--state-bits",
            "1",
        ])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("Classified all 2 states"), "{text}");
    assert!(text.contains("period 2"), "{text}");
}

#[test]
fn recurrent_mode_refuses_a_non_closed_domain() {
    let path = write_temp_program("recurrent_not_closed.ouro", "0 ORACLE 1 ADD 0 PROPHECY\n");
    let out = ouro()
        .args([
            path.to_str().unwrap(),
            "--recurrent",
            "--memory-cells",
            "1",
            "--state-bits",
            "2",
        ])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "stdout was: {}", text);
    assert!(
        text.contains("RECURRENT ANALYSIS REFUSED"),
        "stdout was: {}",
        text
    );
    assert!(
        text.contains("outside the 2-bit domain"),
        "stdout was: {}",
        text
    );
}

#[test]
fn verify_mode_proves_and_refutes_all_fixed_properties() {
    let proven_path = write_temp_program(
        "property_proven.ouro",
        "PROPERTY answer { ALL_FIXED CELL 0 EQ 42; }\n\
         0 ORACLE DUP 42 EQ IF { 0 PROPHECY } ELSE { POP PARADOX }\n",
    );
    let proven = ouro()
        .args([
            proven_path.to_str().unwrap(),
            "--verify",
            "--memory-cells",
            "2",
        ])
        .output()
        .expect("binary runs");
    let proven_text = String::from_utf8_lossy(&proven.stdout);
    assert_eq!(proven.status.code(), Some(0), "stdout was: {}", proven_text);
    assert!(
        proven_text.contains("PROVEN answer"),
        "stdout was: {}",
        proven_text
    );

    let refuted_path = write_temp_program(
        "property_refuted.ouro",
        "PROPERTY zero { ALL_FIXED CELL 0 EQ 0; }\n0 ORACLE 0 PROPHECY\n",
    );
    let refuted = ouro()
        .args([
            refuted_path.to_str().unwrap(),
            "--verify",
            "--memory-cells",
            "2",
        ])
        .output()
        .expect("binary runs");
    let refuted_text = String::from_utf8_lossy(&refuted.stdout);
    assert_eq!(
        refuted.status.code(),
        Some(2),
        "stdout was: {}",
        refuted_text
    );
    assert!(
        refuted_text.contains("REFUTED zero"),
        "stdout was: {}",
        refuted_text
    );
    assert!(
        refuted_text.contains("counterexample"),
        "stdout was: {}",
        refuted_text
    );
}

#[test]
fn verification_json_and_recurrent_dot_artifacts_are_machine_readable() {
    let property_path = write_temp_program(
        "artifact_property.ouro",
        "TEMPORAL a @ 0 DEFAULT 9;\n\
         TEMPORAL b @ 1 DEFAULT 8;\n\
         PROPERTY zero { ALL_FIXED CELL a EQ 0 AND NOT (CELL b NE 0); }\n\
         0 ORACLE 0 PROPHECY\n",
    );
    let artifact_path = std::env::temp_dir()
        .join("ouro-cli-tests")
        .join("property-artifact.json");
    let verified = ouro()
        .args([
            property_path.to_str().unwrap(),
            "--verify",
            "--memory-cells",
            "2",
            "--artifact",
            artifact_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(verified.status.code(), Some(2));
    let artifact = std::fs::read_to_string(&artifact_path).expect("JSON artifact");
    assert!(artifact.starts_with("{\"schema\":\"ourochronos.verification-batch/v1\""));
    assert!(artifact.contains("\"status\":\"refuted\""));
    assert!(artifact.contains("\"counterexample\""));
    assert!(artifact.contains("\"replay_verified\":true"));
    assert!(artifact.contains("\"addresses\":[0,1]"), "{artifact}");
    assert!(artifact.contains("\"name\":\"a\""), "{artifact}");
    assert!(
        artifact.contains("\"incremental_reuse\":false"),
        "{artifact}"
    );

    let recurrent_path = write_temp_program("artifact_recurrent.ouro", "0 ORACLE NOT 0 PROPHECY\n");
    let dot_path = std::env::temp_dir()
        .join("ouro-cli-tests")
        .join("recurrent-artifact.dot");
    let graphed = ouro()
        .args([
            recurrent_path.to_str().unwrap(),
            "--recurrent",
            "--memory-cells",
            "1",
            "--state-bits",
            "1",
            "--dot",
            dot_path.to_str().unwrap(),
        ])
        .output()
        .expect("binary runs");
    assert_eq!(graphed.status.code(), Some(0));
    let dot = std::fs::read_to_string(dot_path).expect("DOT artifact");
    assert!(dot.starts_with("digraph ourochronos_recurrence"));
    assert!(dot.contains("n0 -> n1"));
    assert!(dot.contains("n1 -> n0"));
}

#[test]
fn bounded_global_solver_reports_unknown_instead_of_a_false_proof() {
    let path = write_temp_program(
        "global_bounded_unknown.ouro",
        "0 ORACLE WHILE { DUP 100 GT } { 1 SUB } 1 ADD 0 PROPHECY\n",
    );
    let out = ouro()
        .args([
            path.to_str().unwrap(),
            "--global",
            "--memory-cells",
            "2",
            "--loop-unroll",
            "0",
        ])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "stdout was: {}", text);
    assert!(text.contains("UNKNOWN"), "stdout was: {}", text);
    assert!(text.contains("not a global proof"), "stdout was: {}", text);
}

#[test]
fn global_solver_rejects_unsupported_linked_bytecode_boundaries() {
    let path = write_temp_program("global_bytecode_unsupported.ouro", "VEC_NEW POP\n");
    let out = ouro()
        .args([path.to_str().unwrap(), "--global", "--memory-cells", "1"])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(
        text.contains("unsupported temporal bytecode boundary"),
        "{text}"
    );
    assert!(text.contains("VecNew"), "{text}");
}

#[test]
fn smt_export_uses_the_typed_logical_not_semantics() {
    let path = write_temp_program("typed_smt_not.ouro", "0 ORACLE NOT 0 PROPHECY\n");
    let out = ouro()
        .args([path.to_str().unwrap(), "--smt", "--memory-cells", "2"])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("Typed temporal IR"), "stdout was: {}", text);
    assert!(
        !text.contains("bvnot"),
        "logical NOT must not be emitted as bitwise NOT: {}",
        text
    );
}

#[test]
fn version_exits_zero() {
    let out = ouro().arg("--version").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn no_arguments_exits_one_with_usage() {
    let out = ouro().output().expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage"));
}

#[test]
fn missing_file_exits_one_without_panic() {
    let out = ouro()
        .arg("does-not-exist.ouro")
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("Compile Error"));
    assert!(err.contains("failed to canonicalize"));
    assert!(!err.contains("panicked"));
}

#[test]
fn unknown_flag_exits_one() {
    let out = ouro()
        .args(["examples/hello.ouro", "--bogus"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown option"));
}

#[test]
fn malformed_flag_value_exits_one() {
    let out = ouro()
        .args(["examples/hello.ouro", "--seed", "abc"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("requires a number"));
}

#[test]
fn configurable_memory_crosses_the_legacy_address_boundary() {
    let path = write_temp_program(
        "wide_memory.ouro",
        "42 65536 PROPHECY 65536 PRESENT OUTPUT 0 ORACLE POP\n",
    );
    let out = ouro()
        .args([
            path.to_str().unwrap(),
            "--memory-cells",
            "70000",
            "--strict",
        ])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("[42]"), "stdout was: {}", text);
}

#[test]
fn zero_memory_width_is_rejected() {
    let out = ouro()
        .args(["examples/hello.ouro", "--memory-cells", "0"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("greater than 0"));
}

#[test]
fn temporal_region_contracts_are_enforced_without_typecheck_flag() {
    let path = write_temp_program(
        "effectful_temporal_region.ouro",
        "TEMPORAL 0 1 { CLOCK POP }\n",
    );
    let out = ouro().arg(path).output().expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Finite temporal region"), "{}", stderr);
    assert!(stderr.contains("CLOCK"), "{}", stderr);
    assert!(stderr.contains("contract errors"), "{}", stderr);
}

#[test]
fn resources_report_is_explicitly_instance_level() {
    let out = ouro()
        .args([
            "examples/paradox.ouro",
            "--deutsch",
            "--memory-cells",
            "70000",
            "--resources",
        ])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("70000 cells x 64 bits"),
        "stdout was: {}",
        text
    );
    assert!(
        text.contains("Address bits required: 17"),
        "stdout was: {}",
        text
    );
    assert!(
        text.contains("not by itself a PSPACE-family proof"),
        "stdout was: {}",
        text
    );
}

#[test]
fn resources_report_surfaces_source_family_contract() {
    let out = ouro()
        .args(["examples/pspace_contract.ouro", "--deutsch", "--resources"])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("Declared PSPACE Family Contract: identity_family"));
    assert!(text.contains("semantic proof remains external"));
}

#[test]
fn bounded_halting_distinguishes_halt_from_unknown() {
    let halted = ouro()
        .args(["examples/turing_complete.ouro", "--halting-bound", "1000"])
        .output()
        .expect("binary runs");
    let halted_text = String::from_utf8_lossy(&halted.stdout);
    assert_eq!(halted.status.code(), Some(0), "stdout was: {}", halted_text);
    assert!(
        halted_text.contains("HALTED"),
        "stdout was: {}",
        halted_text
    );

    let unknown = ouro()
        .args(["examples/halting_limit.ouro", "--halting-bound", "1000"])
        .output()
        .expect("binary runs");
    let unknown_text = String::from_utf8_lossy(&unknown.stdout);
    assert_eq!(
        unknown.status.code(),
        Some(3),
        "stdout was: {}",
        unknown_text
    );
    assert!(
        unknown_text.contains("UNKNOWN"),
        "stdout was: {}",
        unknown_text
    );
    assert!(
        unknown_text.contains("not proof of nonhalting"),
        "stdout was: {}",
        unknown_text
    );
}

#[test]
fn bounded_halting_rejects_temporal_semantics() {
    let out = ouro()
        .args(["examples/paradox.ouro", "--halting-bound", "1000"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("UNSUPPORTED HALTING ANALYSIS"));
}

#[test]
fn source_markov_model_is_solved_across_all_fixed_points() {
    let out = ouro()
        .arg("examples/stochastic_accept.ouro")
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("stationary {0:1/3, 1:2/3}"),
        "stdout was: {}",
        text
    );
    assert!(text.contains("DECISION: ACCEPT on every stationary fixed point"));
}

#[test]
fn source_markov_model_exposes_ambiguous_readouts() {
    let out = ouro()
        .arg("examples/stochastic_ambiguous.ouro")
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(2), "stdout was: {}", text);
    assert!(text.contains("Class 0 [0]"), "stdout was: {}", text);
    assert!(text.contains("Class 1 [1]"), "stdout was: {}", text);
    assert!(text.contains("DECISION: AMBIGUOUS"), "stdout was: {}", text);
}

#[test]
fn source_quantum_channel_verifies_every_fixed_density() {
    let out = ouro()
        .arg("examples/quantum_reset.ouro")
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("Fixed affine dimension: 0"),
        "stdout was: {}",
        text
    );
    assert!(text.contains("[1.000000000000, 1.000000000000]"));
    assert!(text.contains("ACCEPT on every fixed density operator"));
}

#[test]
fn source_quantum_channel_exposes_fixed_density_ambiguity() {
    let out = ouro()
        .arg("examples/quantum_identity.ouro")
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(2), "stdout was: {}", text);
    assert!(
        text.contains("Fixed affine dimension: 3"),
        "stdout was: {}",
        text
    );
    assert!(text.contains("[0.000000000000, 1.000000000000]"));
    assert!(text.contains("DECISION: AMBIGUOUS"));
}

#[test]
fn consistent_program_exits_zero() {
    let out = ouro()
        .arg("examples/hello.ouro")
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Hello World!"));
}

#[test]
fn oscillating_paradox_exits_two() {
    let out = ouro()
        .arg("examples/paradox.ouro")
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("OSCILLATION"));
}

#[test]
fn deutsch_mode_accepts_a_cycle_as_stationary() {
    let out = ouro()
        .args(["examples/paradox.ouro", "--deutsch"])
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("DEUTSCH-CONSISTENT"), "stdout was: {}", text);
    assert!(text.contains("probability 1/2"), "stdout was: {}", text);
}

#[test]
fn diagnostic_quotation_uses_bytecode_and_commits_only_fixed_output() {
    let path = write_temp_program(
        "diagnostic_bytecode_quote.ouro",
        "[ 0 ORACLE DUP OUTPUT POP 42 0 PROPHECY ] EXEC\n",
    );
    let out = ouro()
        .args([path.to_str().unwrap(), "--diagnostic"])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("Running in DIAGNOSTIC mode"), "{text}");
    assert!(text.contains("CONSISTENT after 2 epochs"), "{text}");
    assert!(text.contains("[42]"), "{text}");
    assert!(!text.contains("[0]"), "candidate output leaked: {text}");
}

#[test]
fn deutsch_procedure_uses_bytecode_with_unanimous_output() {
    let path = write_temp_program(
        "deutsch_bytecode_procedure.ouro",
        "PROCEDURE flip { 0 ORACLE NOT 0 PROPHECY 7 OUTPUT } flip\n",
    );
    let out = ouro()
        .args([path.to_str().unwrap(), "--deutsch"])
        .output()
        .expect("binary runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("DEUTSCH-CONSISTENT"), "{text}");
    assert!(text.contains("probability 1/2"), "{text}");
    assert!(text.contains("[7]"), "{text}");
    assert!(!text.contains("AMBIGUOUS READOUT"), "{text}");
}

#[test]
fn execution_modes_are_mutually_exclusive() {
    let out = ouro()
        .args(["examples/paradox.ouro", "--deutsch", "--diagnostic"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("mutually exclusive"));
}

#[test]
fn non_converging_program_exits_three() {
    // A strictly increasing counter never revisits a state and never
    // converges, so the search exhausts max_epochs.
    let path = write_temp_program("diverge.ouro", "0 ORACLE 1 ADD 0 PROPHECY\n");

    let out = ouro()
        .arg(path.to_str().unwrap())
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "stdout was: {}", text);
    assert!(text.contains("TIMEOUT"));
}

#[test]
fn diagnostic_mode_freezes_interactive_input() {
    // The Temporal Input Invariant: the first epoch's inputs are replayed in
    // every later epoch. Before the fix only standard mode froze inputs;
    // diagnostic mode re-read stdin each epoch, so with input lines 5, 9, 7
    // the search would drift to the EOF fallback (0) instead of converging
    // on the first value read.
    use std::io::Write;
    use std::process::Stdio;

    let path = write_temp_program(
        "frozen_input.ouro",
        "INPUT DUP 0 PROPHECY OUTPUT 0 ORACLE POP\n",
    );

    let mut child = ouro()
        .args([path.to_str().unwrap(), "--diagnostic"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary runs");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"5\n9\n7\n")
        .expect("pipe input");
    let out = child.wait_with_output().expect("binary finishes");

    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(
        text.contains("[5]"),
        "expected frozen input 5, stdout was: {}",
        text
    );
}

#[test]
fn standard_bytecode_mode_captures_interactive_input_once() {
    use std::io::Write;
    use std::process::Stdio;

    let path = write_temp_program(
        "bytecode_frozen_input.ouro",
        "INPUT DUP 0 PROPHECY OUTPUT 0 ORACLE POP\n",
    );
    let mut child = ouro()
        .arg(path.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary runs");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"17\n23\n")
        .expect("pipe input");
    let out = child.wait_with_output().expect("binary finishes");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "{text}");
    assert!(text.contains("[17]"), "{text}");
    assert!(!text.contains("[23]"), "{text}");
}

#[test]
fn effects_flag_gates_nondeterminism_in_search() {
    let path = write_temp_program(
        "clock_in_loop.ouro",
        "0 ORACLE POP CLOCK POP 0 0 PROPHECY\n",
    );

    // Default policy declines with a message naming the opcode.
    let out = ouro()
        .arg(path.to_str().unwrap())
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("CLOCK"));

    // Unrestricted policy lets the search run to a consistent timeline.
    let out = ouro()
        .args([path.to_str().unwrap(), "--effects", "unrestricted"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(0));

    // Garbage policy value is a usage error.
    let out = ouro()
        .args([path.to_str().unwrap(), "--effects", "sometimes"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--effects"));
}

#[test]
fn repeated_flag_exits_one() {
    let out = ouro()
        .args(["examples/hello.ouro", "--seed", "5", "--seed", "9"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("more than once"));
}

#[test]
fn temporal_sort_example_converges_on_the_sorted_witness() {
    // Regression: the shipped example was written against 1-based PICK
    // indexing and failed on its first verification step for as long as it
    // existed. It must reach the sorted fixed point and print it.
    let out = ouro()
        .arg("examples/temporal_sort.ouro")
        .output()
        .expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("[1][2][3][4]"), "stdout was: {}", text);
}

#[test]
fn examples_meet_their_contracts() {
    // The examples directory is the thesis evidence: each programme pins its
    // exit code and, where deterministic, an output fragment. The fibonacci
    // benchmarks are excluded (they need a raised --max-inst by design, as
    // their headers document), as are examples/modules (IMPORT resolves
    // relative to the working directory).
    let contracts: &[(&str, i32, Option<&str>)] = &[
        ("hello", 0, Some("Hello World!")),
        ("simple", 0, None),
        ("simple_string", 0, None),
        ("strings", 0, None),
        ("effects", 0, None),
        ("expression_syntax", 0, None),
        ("fibonacci", 0, None),
        ("let_bindings", 0, None),
        ("procedures", 0, None),
        ("program", 0, None),
        ("recursion", 0, None),
        ("sqrt", 0, None),
        ("temporal", 0, Some("[42]")),
        ("sat", 0, None),
        ("bootstrap", 0, Some("HELLO")),
        ("temporal_variables", 0, Some("[3]")),
        ("primality", 0, Some("[3]")),
        ("combinators", 0, Some("[10][20][11][10][11][20][11][57]")),
        ("data_structures", 0, Some("[10][30][100][42][3][2]")),
        ("turing_complete", 0, Some("[3]")),
        ("pspace_contract", 0, Some("[0]")),
        ("stochastic_accept", 0, Some("DECISION: ACCEPT")),
        ("stochastic_ambiguous", 2, Some("DECISION: AMBIGUOUS")),
        ("quantum_reset", 0, Some("ACCEPT on every fixed density")),
        ("quantum_identity", 2, Some("DECISION: AMBIGUOUS")),
        ("paradox", 2, Some("OSCILLATION")),
        ("quantum_suicide", 2, Some("OSCILLATION")),
        ("assert_test", 1, Some("Expect Failure")),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (name, expected_exit, fragment) in contracts {
        let path = format!("examples/{}.ouro", name);
        let out = ouro().arg(&path).output().expect("binary runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if out.status.code() != Some(*expected_exit) {
            failures.push(format!(
                "{}: expected exit {}, got {:?}; output: {}",
                name,
                expected_exit,
                out.status.code(),
                text.trim()
            ));
            continue;
        }
        if let Some(needle) = fragment {
            if !text.contains(needle) {
                failures.push(format!(
                    "{}: output missing '{}'; output: {}",
                    name,
                    needle,
                    text.trim()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "example contracts broken:\n{}",
        failures.join("\n")
    );
}
