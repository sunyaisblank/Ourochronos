use ourochronos::audit::{self, ActionCategory, AuditConfig, AuditEntry, AuditFormat, Severity};
use ourochronos::{
    build_native_artifact_launcher, bytecode_vm_supports, check_semantics, embedded_artifact,
    link_with_metadata, type_check, types, verify_bytecode, ActionConfig, BytecodeProgram,
    BytecodeTimeLoop, BytecodeTimeLoopConfig, BytecodeVmConfig, ConvergenceStatus, ErrorConfig,
    Instruction, ModuleGraph, ObjectModule, ObligationKind, OpCode, PackageManifest,
    PackageWitness, PortableArtifact, PortableArtifactPayload, PortablePackage, TemporalIrConfig,
};
use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

/// Process exit codes. A consistent timeline (or successful pure run) exits 0;
/// usage, parse, and runtime errors exit 1; a detected paradox (oscillation,
/// divergence, or explicit PARADOX) exits 2; resource-bounded unknown exits 3, so
/// scripts can distinguish a detected inconsistency on the explored orbit
/// from resource exhaustion, which remains an unknown result.
const EXIT_OK: i32 = 0;
const EXIT_ERROR: i32 = 1;
const EXIT_PARADOX: i32 = 2;
const EXIT_TIMEOUT: i32 = 3;

/// How a flag consumes the argument that follows it.
#[derive(Clone, Copy, PartialEq)]
enum Arity {
    /// Stands alone.
    None,
    /// Requires a value.
    Value,
    /// Takes a value when the next argument is not a flag.
    OptionalValue,
}

/// One command-line flag: name, alias, arity, and usage text.
///
/// This table is the single authority for the flag surface: validation,
/// duplicate rejection, and the usage screen all derive from it, so a flag
/// cannot be recognised in one place and unknown in another.
struct FlagSpec {
    name: &'static str,
    alias: Option<&'static str>,
    arity: Arity,
    metavar: &'static str,
    help: &'static [&'static str],
}

const FLAGS: &[FlagSpec] = &[
    FlagSpec {
        name: "--diagnostic",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Enable diagnostic mode (full trajectory recording)"],
    },
    FlagSpec {
        name: "--deutsch",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Accept cycles as Deutsch stationary ensembles"],
    },
    FlagSpec {
        name: "--action",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Enable action-guided mode (solves the Genie Effect)"],
    },
    FlagSpec {
        name: "--typecheck",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Run static type analysis (temporal tainting)"],
    },
    FlagSpec {
        name: "--resources",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Print concrete complexity-resource bounds before execution"],
    },
    FlagSpec {
        name: "--check",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Run every mandatory compiler/verifier gate without executing"],
    },
    FlagSpec {
        name: "--emit-object",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Compile the entry source module to a relocatable object (dependencies separate)"],
    },
    FlagSpec {
        name: "--emit-objects",
        alias: None,
        arity: Arity::Value,
        metavar: "<directory>",
        help: &["Compile every source module to a deterministic relocatable object"],
    },
    FlagSpec {
        name: "--emit-bytecode",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Compile and link a deterministic portable bytecode artifact"],
    },
    FlagSpec {
        name: "--build",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Build a deterministic portable package and do not execute"],
    },
    FlagSpec {
        name: "--build-executable",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Build a directly runnable native launcher with embedded bytecode"],
    },
    FlagSpec {
        name: "--embed-global-witness",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Solve and embed a replay-verified initial point state in a package"],
    },
    FlagSpec {
        name: "--runtime-global-package",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Package an explicit versioned Z3 global-point runtime dependency"],
    },
    FlagSpec {
        name: "--smt",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Export the typed temporal IR as SMT-LIB2 instead of running"],
    },
    FlagSpec {
        name: "--global",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Globally solve F(s)=s for the finite temporal region"],
    },
    FlagSpec {
        name: "--all-fixed",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Prove zero/unique/multiple point fixed states globally"],
    },
    FlagSpec {
        name: "--recurrent",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Classify every cycle/basin in a closed small-state domain"],
    },
    FlagSpec {
        name: "--verify",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Verify every source PROPERTY over all point fixed states"],
    },
    FlagSpec {
        name: "--verify-family",
        alias: None,
        arity: Arity::Value,
        metavar: "<bitstring>",
        help: &["Verify one finite FAMILY specialization for an exact binary input"],
    },
    FlagSpec {
        name: "--artifact",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Write a verification result/counterexample JSON artifact"],
    },
    FlagSpec {
        name: "--dot",
        alias: None,
        arity: Arity::Value,
        metavar: "<file>",
        help: &["Write the complete --recurrent graph as Graphviz DOT"],
    },
    FlagSpec {
        name: "--state-bits",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Bits per cell in recurrent/family domain (default: 1)"],
    },
    FlagSpec {
        name: "--state-limit",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Maximum states enumerated by recurrent/family proof (default: 65536)"],
    },
    FlagSpec {
        name: "--solver-timeout",
        alias: None,
        arity: Arity::Value,
        metavar: "<ms>",
        help: &["Global solver timeout in milliseconds (default: 30000)"],
    },
    FlagSpec {
        name: "--loop-unroll",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Loop iterations represented by the global solver (default: 10)"],
    },
    FlagSpec {
        name: "--stationary",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Solve a source MARKOV declaration across every stationary class"],
    },
    FlagSpec {
        name: "--quantum-fixed",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Analyze every fixed density of a source qubit QCHANNEL"],
    },
    FlagSpec {
        name: "--seed",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Set initial seed value"],
    },
    FlagSpec {
        name: "--seeds",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Number of seeds to try in action mode (default: 4)"],
    },
    FlagSpec {
        name: "--max-inst",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Maximum instructions per epoch (default: 10000000)"],
    },
    FlagSpec {
        name: "--halting-bound",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Observe deterministic classical halting for at most n instructions"],
    },
    FlagSpec {
        name: "--memory-cells",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Temporal memory width (default: 65536 cells)"],
    },
    FlagSpec {
        name: "--fast",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Use fast VM for pure (non-temporal) programmes"],
    },
    FlagSpec {
        name: "--lsp",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Start Language Server Protocol server"],
    },
    FlagSpec {
        name: "--strict",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Strict error handling (bounds violations produce errors)"],
    },
    FlagSpec {
        name: "--permissive",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Permissive error handling (wrap addresses, zero on underflow)"],
    },
    FlagSpec {
        name: "--audit",
        alias: None,
        arity: Arity::OptionalValue,
        metavar: "[file]",
        help: &["Enable audit logging (default: ourochronos-audit.log)"],
    },
    FlagSpec {
        name: "--audit-json",
        alias: None,
        arity: Arity::None,
        metavar: "",
        help: &["Use JSON Lines format for audit output"],
    },
    FlagSpec {
        name: "--provenance-limit",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Provenance saturation limit (default: 256)"],
    },
    FlagSpec {
        name: "--effects",
        alias: None,
        arity: Arity::Value,
        metavar: "<policy>",
        help: &[
            "Effects inside the fixed-point search:",
            "'decline' (default) requires frozen/modelled inputs;",
            "'unrestricted' permits one first-candidate CLOCK/RANDOM",
            "capture which is then frozen. Host writes still require",
            "an exact capability and commit only after selection",
        ],
    },
    FlagSpec {
        name: "--allow-file-read",
        alias: None,
        arity: Arity::Value,
        metavar: "<path>",
        help: &["Freeze one exact host file path before temporal evaluation (repeatable)"],
    },
    FlagSpec {
        name: "--allow-file-write",
        alias: None,
        arity: Arity::Value,
        metavar: "<path>",
        help: &["Authorize one exact selected-timeline file commit path (repeatable)"],
    },
    FlagSpec {
        name: "--network-input",
        alias: None,
        arity: Arity::Value,
        metavar: "<endpoint>=<file>",
        help: &[
            "Freeze an exact endpoint receive stream before evaluation (repeatable)",
            "The endpoint is host:port; this does not authorize a host send",
        ],
    },
    FlagSpec {
        name: "--allow-network-send",
        alias: None,
        arity: Arity::Value,
        metavar: "<endpoint>",
        help: &["Authorize one exact selected-timeline TCP send (repeatable)"],
    },
    FlagSpec {
        name: "--allow-process",
        alias: None,
        arity: Arity::Value,
        metavar: "<descriptor>",
        help: &[
            "Supply a frozen result and authorize its exact shell command (repeatable)",
            "Descriptor: OUROPROCESS/1, exit, command length, command, output",
        ],
    },
    FlagSpec {
        name: "--allow-sleep-ms",
        alias: None,
        arity: Arity::Value,
        metavar: "<n>",
        help: &["Authorize selected-timeline sleeps of at most n milliseconds"],
    },
    FlagSpec {
        name: "--help",
        alias: Some("-h"),
        arity: Arity::None,
        metavar: "",
        help: &["Show this help"],
    },
    FlagSpec {
        name: "--version",
        alias: Some("-V"),
        arity: Arity::None,
        metavar: "",
        help: &["Show version, platform and compiled features"],
    },
];

fn find_flag(arg: &str) -> Option<&'static FlagSpec> {
    FLAGS.iter().find(|f| f.name == arg || f.alias == Some(arg))
}

fn print_usage() {
    println!("Usage: ourochronos <file.ouro> [options]");
    println!("       ourochronos repl");
    println!("       ourochronos link <output.ourobc> <input.ouroobj>...");
    println!("       ourochronos run-package <file.ouropkg>");
    println!("       ourochronos prove-finite <file.ourobc> <proof.ourofp> <memory-cells> <gas>");
    println!("       ourochronos check-finite <file.ourobc> <proof.ourofp> <memory-cells> <gas>");
    println!("       ourochronos halt-slice <file.ourobc> <saved.ourocp> <slice> <memory-cells> <ceiling>");
    println!("       ourochronos analyse-affine <file.ourobc> <parity-mask> <memory-cells> <gas>");
    println!("       ourochronos isolate <wall-ms> <memory-MiB> <source.ouro> [analysis flags]");
    println!("       ourochronos halt-resume <file.ourobc> <saved.ourocp> <slice> <memory-cells> <ceiling>");
    println!();
    println!("Options:");
    for flag in FLAGS {
        let mut left = flag.name.to_string();
        if let Some(alias) = flag.alias {
            left.push_str(", ");
            left.push_str(alias);
        }
        if !flag.metavar.is_empty() {
            left.push(' ');
            left.push_str(flag.metavar);
        }
        let mut lines = flag.help.iter();
        if let Some(first) = lines.next() {
            println!("  {:<24}{}", left, first);
        }
        for line in lines {
            println!("  {:<24}{}", "", line);
        }
    }
    println!();
    println!("Exit codes: 0 consistent/decided/halted, 1 error, 2 paradox/ambiguous, 3 resource-bounded unknown");
}

/// Print a usage error with the help pointer and return the error exit code.
fn fail_usage(msg: &str) -> i32 {
    eprintln!("Error: {}", msg);
    eprintln!("Run 'ourochronos --help' for usage.");
    EXIT_ERROR
}

fn write_analysis_file(path: &str, contents: &str) -> Result<(), String> {
    fs::write(path, contents)
        .map_err(|error| format!("cannot write analysis artifact '{}': {}", path, error))
}

fn write_binary_file(path: &str, contents: &[u8]) -> Result<(), String> {
    fs::write(path, contents).map_err(|error| format!("cannot write '{}': {}", path, error))
}

fn object_output_name(index: usize, module_name: &str) -> String {
    let stem = Path::new(module_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("module");
    let safe = stem
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!(
        "{index:04}-{}.ouroobj",
        if safe.is_empty() { "module" } else { &safe }
    )
}

fn write_native_launcher_file(
    path: &str,
    contents: &[u8],
    runtime_path: &Path,
) -> Result<(), String> {
    write_binary_file(path, contents)?;
    #[cfg(not(unix))]
    let _ = runtime_path;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let mode = fs::metadata(runtime_path)
            .map_err(|error| {
                format!(
                    "cannot inspect runtime executable '{}': {}",
                    runtime_path.display(),
                    error
                )
            })?
            .mode()
            | 0o111;
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|error| {
            format!(
                "cannot make native launcher '{}' executable: {}",
                path, error
            )
        })?;
    }
    Ok(())
}

fn print_output_items(output: &[ourochronos::OutputItem]) {
    for item in output {
        match item {
            ourochronos::OutputItem::Val(value) => print!("[{}]", value.val),
            ourochronos::OutputItem::Char(character) => print!("{}", *character as char),
        }
    }
    if !output.is_empty() {
        println!();
    }
}

fn print_unsat_evidence(certificate: &ourochronos::UnsatCertificate) {
    println!(
        "  UNSAT evidence: format v{}, {} query bytes + {} proof bytes; fresh Z3 replay verified.",
        certificate.evidence_format,
        certificate.solver_query.len(),
        certificate.backend_proof.len()
    );
}

fn finish_bytecode_standard(outcome: ConvergenceStatus) -> i32 {
    finish_bytecode_orbit(outcome, false)
}

fn finish_bytecode_orbit(outcome: ConvergenceStatus, diagnostic: bool) -> i32 {
    match outcome {
        ConvergenceStatus::Consistent { epochs, output, .. } => {
            if diagnostic || output.is_empty() {
                println!("CONSISTENT after {epochs} epochs.");
            }
            if !output.is_empty() {
                print_output_items(&output);
            }
            EXIT_OK
        }
        ConvergenceStatus::Paradox { message, .. } => {
            println!("PARADOX: {message}");
            EXIT_PARADOX
        }
        ConvergenceStatus::Oscillation {
            period,
            oscillating_cells,
            diagnosis,
            ..
        } => {
            println!("OSCILLATION detected (period {period})");
            if diagnostic && !oscillating_cells.is_empty() {
                println!("Oscillating Addresses: {oscillating_cells:?}");
            }
            if diagnostic {
                match diagnosis {
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::NegativeLoop {
                        explanation,
                        ..
                    } => {
                        println!("\nDIAGNOSIS (Grandfather Paradox):");
                        println!("{explanation}");
                    }
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::Oscillation { cycle } => {
                        println!("\nCycle states:");
                        for (index, state) in cycle.iter().enumerate() {
                            let nonzero = state
                                .iter()
                                .filter(|(_, value)| *value != 0)
                                .collect::<Vec<_>>();
                            if !nonzero.is_empty() {
                                println!("  State {index}: {nonzero:?}");
                            }
                        }
                    }
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::Unknown => {
                        println!("\nDIAGNOSIS: Unknown cause");
                    }
                }
            }
            EXIT_PARADOX
        }
        ConvergenceStatus::Timeout { max_epochs } => {
            println!("TIMEOUT after {max_epochs} epochs.");
            EXIT_TIMEOUT
        }
        ConvergenceStatus::Error { message, .. } => {
            println!("ERROR: {message}");
            EXIT_ERROR
        }
        ConvergenceStatus::DeutschConsistent {
            period,
            transient_epochs,
            unanimous_output,
            ..
        } => {
            println!(
                "DEUTSCH-CONSISTENT: stationary cycle of period {period} after {transient_epochs} transient epoch(s); each state has probability 1/{period}."
            );
            if let Some(output) = unanimous_output {
                print_output_items(&output);
            } else {
                println!("AMBIGUOUS READOUT: cycle states do not all produce the same output.");
            }
            EXIT_OK
        }
        ConvergenceStatus::Divergence { .. } => {
            println!("DIVERGENCE detected.");
            EXIT_PARADOX
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn construct_linked_package_artifact(
    name: String,
    memory_cells: usize,
    max_instructions: u64,
    memory_bounds: ourochronos::BoundsPolicy,
    linked: ourochronos::LinkedProgram,
    embed_global_witness: bool,
    runtime_global_package: bool,
    solver_timeout_ms: u64,
    loop_unroll_limit: usize,
) -> Result<PortableArtifact, String> {
    let artifact = PortableArtifact::from_linked(linked).map_err(|error| error.to_string())?;
    let PortableArtifact {
        payload: PortableArtifactPayload::Bytecode(program),
        provenance,
    } = artifact
    else {
        return Err("linked source did not produce a bytecode payload".into());
    };
    let package = construct_portable_package(
        name,
        memory_cells,
        max_instructions,
        memory_bounds,
        program,
        embed_global_witness,
        runtime_global_package,
        solver_timeout_ms,
        loop_unroll_limit,
    )?;
    Ok(PortableArtifact {
        payload: PortableArtifactPayload::Package(package),
        provenance,
    })
}

#[allow(clippy::too_many_arguments)]
fn construct_portable_package(
    name: String,
    memory_cells: usize,
    max_instructions: u64,
    memory_bounds: ourochronos::BoundsPolicy,
    program: BytecodeProgram,
    embed_global_witness: bool,
    runtime_global_package: bool,
    solver_timeout_ms: u64,
    loop_unroll_limit: usize,
) -> Result<PortablePackage, String> {
    let manifest =
        PackageManifest::with_runtime(name, memory_cells, max_instructions, memory_bounds);
    if runtime_global_package {
        return PortablePackage::with_runtime_global_point(
            manifest,
            program,
            ourochronos::PackageSolverDependency::Z3_GLOBAL_POINT,
        )
        .map_err(|error| error.to_string());
    }
    if !embed_global_witness {
        return PortablePackage::new(manifest, program).map_err(|error| error.to_string());
    }

    let result = ourochronos::GlobalFixedPointSolver::solve_bytecode(
        &program,
        ourochronos::GlobalSolveConfig {
            memory_cells,
            loop_unroll_limit,
            solver_timeout_ms,
            max_instructions,
            bounds_policy: memory_bounds,
        },
    );
    let witness = match result {
        ourochronos::GlobalSolveResult::Found(witness) if witness.is_replay_verified() => witness,
        ourochronos::GlobalSolveResult::Found(_) => {
            return Err("global solver returned a witness without independent replay".to_string());
        }
        ourochronos::GlobalSolveResult::ProvenNoFixedPoint(_) => {
            return Err("cannot embed a witness: the program has no point fixed state".to_string());
        }
        ourochronos::GlobalSolveResult::Unknown {
            reason,
            completeness,
            ..
        } => {
            return Err(format!(
                "cannot embed an incomplete/unknown witness: {reason}; IR completeness: {completeness:?}"
            ));
        }
        ourochronos::GlobalSolveResult::Unsupported { reason } => {
            return Err(format!("cannot embed an unsupported witness: {reason}"));
        }
        ourochronos::GlobalSolveResult::InternalError { reason } => {
            return Err(format!("cannot embed witness after solver error: {reason}"));
        }
    };
    let state = witness
        .memory
        .iter_nonzero()
        .map(|(address, value)| (address, value.val))
        .collect();
    let package_witness =
        PackageWitness::replay_bound(&manifest, &program, state, witness.instructions_executed)
            .map_err(|error| error.to_string())?;
    PortablePackage::with_replay_witness(manifest, program, package_witness)
        .map_err(|error| error.to_string())
}

fn execute_portable_package(
    package: PortablePackage,
    diagnostic_sources: Vec<ourochronos::ObjectSourceFile>,
) -> i32 {
    if let Err(error) = verify_bytecode(&package.program) {
        eprintln!("Package Error: bytecode verification failed: {error}");
        return EXIT_ERROR;
    }
    if let Some(opcode) = package.program.instructions.iter().find_map(|instruction| {
        let Instruction::Primitive(opcode) = instruction else {
            return None;
        };
        (!bytecode_vm_supports(*opcode) || *opcode == OpCode::Input).then_some(*opcode)
    }) {
        eprintln!(
            "Package Error: iterative runtime does not support packaged primitive {}",
            opcode.name()
        );
        return EXIT_ERROR;
    }
    if package
        .program
        .instructions
        .iter()
        .any(|instruction| matches!(instruction, Instruction::CallForeign(_)))
    {
        eprintln!("Package Error: packaged foreign calls are not linked");
        return EXIT_ERROR;
    }
    let memory_cells = match usize::try_from(package.manifest.memory_cells) {
        Ok(memory_cells) => memory_cells,
        Err(_) => {
            eprintln!("Package Error: memory width does not fit this host");
            return EXIT_ERROR;
        }
    };
    let initial_state = match package.manifest.resolution_policy {
        ourochronos::PackageResolutionPolicy::Orbit => Vec::new(),
        ourochronos::PackageResolutionPolicy::EmbeddedPointWitness => package
            .witness
            .as_ref()
            .expect("decoded embedded-point package has a validated witness")
            .state
            .clone(),
        ourochronos::PackageResolutionPolicy::RuntimeGlobalPoint => {
            if package.manifest.solver_dependency != Some(ourochronos::CURRENT_Z3_SOLVER_DEPENDENCY)
            {
                eprintln!("Package Error: runtime global-point solver dependency is incompatible");
                return EXIT_ERROR;
            }
            println!(
                "Resolving packaged global point with declared Z3 contract v{}.",
                ourochronos::CURRENT_Z3_SOLVER_CONTRACT_VERSION
            );
            match ourochronos::GlobalFixedPointSolver::solve_bytecode(
                &package.program,
                ourochronos::GlobalSolveConfig {
                    memory_cells,
                    loop_unroll_limit: 10,
                    solver_timeout_ms: 30_000,
                    max_instructions: package.manifest.max_instructions,
                    bounds_policy: package.manifest.memory_bounds,
                },
            ) {
                ourochronos::GlobalSolveResult::Found(witness) if witness.is_replay_verified() => {
                    witness
                        .memory
                        .iter_nonzero()
                        .map(|(address, value)| (address, value.val))
                        .collect()
                }
                ourochronos::GlobalSolveResult::Found(_) => {
                    eprintln!("Package Error: runtime solver witness failed independent replay");
                    return EXIT_ERROR;
                }
                ourochronos::GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                    println!("PROVEN NO POINT FIXED STATE for packaged runtime-global policy.");
                    print_unsat_evidence(&certificate);
                    return EXIT_PARADOX;
                }
                ourochronos::GlobalSolveResult::Unknown {
                    reason,
                    completeness,
                    ..
                } => {
                    println!("UNKNOWN packaged global resolution: {reason}");
                    println!("IR completeness: {completeness:?}");
                    return EXIT_TIMEOUT;
                }
                ourochronos::GlobalSolveResult::Unsupported { reason } => {
                    eprintln!("Package Error: unsupported runtime global resolution: {reason}");
                    return EXIT_ERROR;
                }
                ourochronos::GlobalSolveResult::InternalError { reason } => {
                    eprintln!("Package Error: runtime global solver failed: {reason}");
                    return EXIT_ERROR;
                }
            }
        }
    };
    let driver = match BytecodeTimeLoop::new(BytecodeTimeLoopConfig {
        memory_cells,
        initial_state,
        diagnostic_sources,
        vm: BytecodeVmConfig {
            max_instructions: package.manifest.max_instructions,
            memory_bounds: package.manifest.memory_bounds,
            ..BytecodeVmConfig::default()
        },
        ..BytecodeTimeLoopConfig::default()
    }) {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("Package Error: {error}");
            return EXIT_ERROR;
        }
    };
    finish_bytecode_standard(driver.run(&package.program))
}

fn execute_package_artifact(artifact: PortableArtifact) -> i32 {
    let diagnostic_sources = artifact
        .provenance
        .map(|provenance| provenance.source_files)
        .unwrap_or_default();
    match artifact.payload {
        PortableArtifactPayload::Package(package) => {
            execute_portable_package(package, diagnostic_sources)
        }
        PortableArtifactPayload::Bytecode(_) => {
            eprintln!("Package Error: artifact has no package runtime/resolution manifest");
            EXIT_ERROR
        }
    }
}

fn run_finite_certificate(args: &[String]) -> i32 {
    use ourochronos::finite_proof::{
        check_certificate, generate_certificate, FiniteNoFixedPointCertificate, FiniteProofConfig,
        FiniteProofQuery, MAX_FINITE_PROOF_BYTES,
    };

    if args.len() != 6 {
        return fail_usage(
            "prove-finite/check-finite require bytecode, certificate, memory-cells, and gas",
        );
    }
    let memory_cells = match args[4].parse::<usize>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("finite proof memory-cells must be a positive integer"),
    };
    let max_instructions = match args[5].parse::<u64>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("finite proof gas must be a positive integer"),
    };
    let operation = || -> Result<_, String> {
        let bytes = read_bounded_regular_file(
            &args[2],
            ourochronos::MAX_PORTABLE_ARTIFACT_BYTES,
            "finite proof bytecode",
        )?;
        let artifact = PortableArtifact::from_bytes(&bytes).map_err(|error| error.to_string())?;
        let PortableArtifactPayload::Bytecode(program) = artifact.payload else {
            return Err(
                "finite proof requires bytecode; package runtime policies are a separate contract"
                    .into(),
            );
        };
        let config = FiniteProofConfig {
            memory_cells,
            max_instructions,
            ..FiniteProofConfig::default()
        };
        let query = FiniteProofQuery::NoFixedPoint;
        let certificate = if args[1] == "prove-finite" {
            generate_certificate(&program, &config, query).map_err(|error| error.to_string())?
        } else {
            let bytes = read_bounded_regular_file(
                &args[3],
                MAX_FINITE_PROOF_BYTES,
                "finite proof certificate",
            )?;
            FiniteNoFixedPointCertificate::from_bytes(&bytes).map_err(|error| error.to_string())?
        };
        let verified = check_certificate(&certificate, &program, &config, query)
            .map_err(|error| error.to_string())?;
        if args[1] == "prove-finite" {
            let bytes = certificate.to_bytes().map_err(|error| error.to_string())?;
            write_binary_file(&args[3], &bytes)?;
        }
        Ok(verified)
    };
    match operation() {
        Ok(verified) => {
            println!(
                "CHECKED NO POINT FIXED STATE: {} states, independent finite checker v{}; memory {}, gas {}.",
                verified.enumerated_states,
                ourochronos::finite_proof::FINITE_PROOF_SEMANTICS_VERSION,
                memory_cells,
                max_instructions
            );
            EXIT_OK
        }
        Err(error) => {
            eprintln!("Finite Proof Error: {error}");
            EXIT_ERROR
        }
    }
}

fn run_affine_analysis(args: &[String]) -> i32 {
    use ourochronos::temporal::affine_recurrence::{
        certify_affine_recurrence, check_affine_bytecode_binding,
        check_affine_recurrence_certificate, extract_affine_bytecode, AffineExtractionConfig,
        AffineReadout, AffineReadoutOutcome,
    };
    if args.len() != 6 {
        return fail_usage("analyse-affine requires bytecode, parity-mask, memory-cells, and gas");
    }
    let mask = match args[3].parse::<u64>() {
        Ok(value) => value,
        _ => return fail_usage("affine parity-mask must be an unsigned 64-bit integer"),
    };
    let memory_cells = match args[4].parse::<usize>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("affine memory-cells must be a positive integer"),
    };
    let max_instructions = match args[5].parse::<u64>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("affine gas must be a positive integer"),
    };
    let operation = || -> Result<_, String> {
        let bytes = read_bounded_regular_file(
            &args[2],
            ourochronos::MAX_PORTABLE_ARTIFACT_BYTES,
            "affine bytecode",
        )?;
        let artifact = PortableArtifact::from_bytes(&bytes).map_err(|error| error.to_string())?;
        let PortableArtifactPayload::Bytecode(program) = artifact.payload else {
            return Err("affine analysis requires bytecode; package runtime policies are a separate contract".into());
        };
        let config = AffineExtractionConfig {
            memory_cells,
            max_instructions,
            ..AffineExtractionConfig::default()
        };
        let extracted =
            extract_affine_bytecode(&program, &config).map_err(|error| error.to_string())?;
        check_affine_bytecode_binding(&extracted, &program, &config)
            .map_err(|error| error.to_string())?;
        let query = AffineReadout {
            mask,
            constant: false,
        };
        let certificate = certify_affine_recurrence(&extracted.model, query)
            .map_err(|error| error.to_string())?;
        check_affine_recurrence_certificate(&certificate, &extracted.model, query)
            .map_err(|error| error.to_string())
    };
    match operation() {
        Ok(verified) => {
            println!("CHECKED AFFINE RECURRENCE: {} Boolean cells; all {} recurrent classes are point fixed states; transient bound {}.",
                verified.bits, verified.class_count.as_u128().expect("affine class count is bounded by 2^64"), verified.transient_bound);
            println!("Trust: bytecode extraction plus independent affine-model certificate checker; parity mask {mask}.");
            match verified.outcome {
                AffineReadoutOutcome::Uniform { value } => {
                    println!("UNIFORM PARITY READOUT: {}.", u8::from(value));
                    EXIT_OK
                }
                AffineReadoutOutcome::Disagreement {
                    zero_state,
                    one_state,
                } => {
                    println!("AMBIGUOUS PARITY READOUT: fixed states {zero_state} and {one_state} give 0 and 1.");
                    EXIT_PARADOX
                }
            }
        }
        Err(error) => {
            eprintln!("Affine Analysis Refused: {error}");
            EXIT_ERROR
        }
    }
}

fn run_isolated(args: &[String]) -> i32 {
    use ourochronos::runtime::isolation::{run_bounded, ProcessLimits};
    use std::io::Write;
    if args.len() < 5 {
        return fail_usage(
            "isolate requires wall-ms, memory-MiB, source, and optional analysis flags",
        );
    }
    let milliseconds = match args[2].parse::<u64>() {
        Ok(value) if (1..=3_600_000).contains(&value) => value,
        _ => return fail_usage("isolation wall-ms must be in 1..=3600000"),
    };
    let memory_mib = match args[3].parse::<u64>() {
        Ok(value) if (16..=16384).contains(&value) => value,
        _ => return fail_usage("isolation memory-MiB must be in 16..=16384"),
    };
    let mut child_args = vec![args[0].clone(), args[4].clone()];
    child_args.extend_from_slice(&args[5..]);
    if let Err(error) = validate_args(&child_args) {
        return fail_usage(&error);
    }
    let mut index = 2;
    while index < child_args.len() {
        let flag = find_flag(&child_args[index]).expect("analysis arguments were validated");
        if !matches!(
            flag.name,
            "--diagnostic"
                | "--deutsch"
                | "--action"
                | "--typecheck"
                | "--resources"
                | "--check"
                | "--smt"
                | "--global"
                | "--all-fixed"
                | "--recurrent"
                | "--verify"
                | "--verify-family"
                | "--state-bits"
                | "--state-limit"
                | "--solver-timeout"
                | "--loop-unroll"
                | "--stationary"
                | "--quantum-fixed"
                | "--seed"
                | "--seeds"
                | "--max-inst"
                | "--halting-bound"
                | "--memory-cells"
                | "--fast"
                | "--strict"
                | "--permissive"
                | "--provenance-limit"
        ) {
            return fail_usage(&format!(
                "{} is outside the isolated analysis profile",
                flag.name
            ));
        }
        index += if flag.arity == Arity::None { 1 } else { 2 };
    }
    let executable = match env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("Isolation Error: {error}");
            return EXIT_ERROR;
        }
    };
    let mut command = std::process::Command::new(executable);
    command.args(&child_args[1..]);
    match run_bounded(
        command,
        ProcessLimits {
            wall_time: std::time::Duration::from_millis(milliseconds),
            address_space_bytes: memory_mib * 1024 * 1024,
            output_bytes: 1024 * 1024,
        },
        None,
    ) {
        Ok(output) if output.stopped.is_some() || output.status.code().is_none() => {
            eprintln!("UNKNOWN isolated execution: stop {:?}, status {}; elapsed {} ms. Partial result output withheld.", output.stopped, output.status, output.elapsed.as_millis());
            EXIT_TIMEOUT
        }
        Ok(output) => {
            if std::io::stdout()
                .write_all(&output.stdout)
                .and_then(|_| std::io::stderr().write_all(&output.stderr))
                .is_err()
            {
                return EXIT_ERROR;
            }
            match output.status.code() {
                Some(code @ 0..=3) => code,
                _ => {
                    eprintln!("Isolation Error: child exited with {}", output.status);
                    EXIT_ERROR
                }
            }
        }
        Err(error) => {
            eprintln!("Isolation Error: {error}");
            EXIT_ERROR
        }
    }
}

fn run_portable_package(args: &[String]) -> i32 {
    let Some(path) = args.get(2) else {
        return fail_usage("run-package requires a package file");
    };
    if args.len() != 3 {
        return fail_usage("run-package accepts exactly one package file");
    }
    let bytes = match read_bounded_regular_file(
        path,
        ourochronos::MAX_PORTABLE_ARTIFACT_BYTES,
        "portable package",
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("Package Error: cannot read '{path}': {error}");
            return EXIT_ERROR;
        }
    };
    match PortableArtifact::from_bytes(&bytes) {
        Ok(artifact) => execute_package_artifact(artifact),
        Err(error) => {
            eprintln!("Package Error: {error}");
            EXIT_ERROR
        }
    }
}

fn run_halting_checkpoint(args: &[String]) -> i32 {
    use ourochronos::checkpoint::{
        CheckpointOutcome, ClassicalCheckpoint, ClassicalCheckpointConfig, UnverifiedCheckpoint,
        MAX_CHECKPOINT_BYTES,
    };
    if args.len() != 7 {
        return fail_usage(
            "halt-slice/halt-resume require bytecode, checkpoint, slice, memory-cells, and ceiling",
        );
    }
    let allowance = match args[4].parse::<u64>() {
        Ok(value) => value,
        _ => return fail_usage("checkpoint slice must be a nonnegative integer"),
    };
    let memory_cells = match args[5].parse::<usize>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("checkpoint memory-cells must be a positive integer"),
    };
    let max_instructions = match args[6].parse::<u64>() {
        Ok(value) if value > 0 => value,
        _ => return fail_usage("checkpoint ceiling must be a positive integer"),
    };
    let operation = || -> Result<i32, String> {
        let bytes = read_bounded_regular_file(
            &args[2],
            ourochronos::MAX_PORTABLE_ARTIFACT_BYTES,
            "checkpoint bytecode",
        )?;
        let artifact = PortableArtifact::from_bytes(&bytes).map_err(|error| error.to_string())?;
        let PortableArtifactPayload::Bytecode(program) = artifact.payload else {
            return Err("halting checkpoint requires bytecode; package runtime policies are a separate contract".into());
        };
        let config = ClassicalCheckpointConfig {
            memory_cells,
            max_instructions,
            ..ClassicalCheckpointConfig::default()
        };
        let checkpoint = if args[1] == "halt-slice" {
            ClassicalCheckpoint::start(&program, config).map_err(|error| error.to_string())?
        } else {
            let bytes =
                read_bounded_regular_file(&args[3], MAX_CHECKPOINT_BYTES, "halting checkpoint")?;
            UnverifiedCheckpoint::from_bytes(&bytes)
                .and_then(|image| image.validate(&program, &config, None))
                .map_err(|error| error.to_string())?
        };
        match checkpoint
            .run_slice(allowance, None)
            .map_err(|error| error.to_string())?
        {
            CheckpointOutcome::Paused { checkpoint, reason } => {
                let bytes = checkpoint.to_bytes().map_err(|error| error.to_string())?;
                write_binary_file(&args[3], &bytes)?;
                println!(
                    "UNKNOWN after {} fetched instructions: {reason:?}; saved {}.",
                    checkpoint.instructions_executed(),
                    args[3]
                );
                Ok(EXIT_TIMEOUT)
            }
            CheckpointOutcome::Complete(execution) => {
                println!(
                    "HALTED after {} fetched instructions.",
                    execution.instructions_executed
                );
                print_output_items(&execution.output);
                Ok(EXIT_OK)
            }
            CheckpointOutcome::Fault {
                error,
                instructions,
            } => Err(format!(
                "runtime fault after {instructions} fetched instructions: {error}"
            )),
        }
    };
    match operation() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("Checkpoint Error: {error}");
            EXIT_ERROR
        }
    }
}

fn run_embedded_package() -> Option<i32> {
    let executable = match env::current_exe() {
        Ok(executable) => executable,
        Err(error) => {
            eprintln!("Launcher Error: cannot locate the current executable: {error}");
            return Some(EXIT_ERROR);
        }
    };
    let bytes = match read_bounded_regular_file(
        &executable,
        ourochronos::MAX_LAUNCHER_BYTES,
        "launcher executable",
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!(
                "Launcher Error: cannot read '{}': {error}",
                executable.display()
            );
            return Some(EXIT_ERROR);
        }
    };
    match embedded_artifact(&bytes) {
        Ok(Some(artifact)) => Some(execute_package_artifact(artifact)),
        Ok(None) => None,
        Err(error) => {
            eprintln!("Launcher Error: {error}");
            Some(EXIT_ERROR)
        }
    }
}

fn link_portable_objects(args: &[String]) -> i32 {
    const MAX_LINK_INPUT_BYTES: usize = ourochronos::MAX_OBJECT_BYTES;
    if args.len() < 4 {
        return fail_usage("link requires an output bytecode file and at least one object");
    }
    if args.len() - 3 > ourochronos::MAX_LINK_OBJECTS {
        eprintln!(
            "Link Error: object count exceeds {}",
            ourochronos::MAX_LINK_OBJECTS
        );
        return EXIT_ERROR;
    }
    let output = &args[2];
    let mut objects = Vec::with_capacity(args.len() - 3);
    let mut aggregate_bytes = 0usize;
    for path in &args[3..] {
        let remaining = MAX_LINK_INPUT_BYTES.saturating_sub(aggregate_bytes);
        let bytes = match read_bounded_regular_file(
            path,
            ourochronos::MAX_OBJECT_BYTES.min(remaining),
            "portable object",
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("Link Error: cannot read object '{path}': {error}");
                return EXIT_ERROR;
            }
        };
        aggregate_bytes = match aggregate_bytes.checked_add(bytes.len()) {
            Some(total) if total <= MAX_LINK_INPUT_BYTES => total,
            _ => {
                eprintln!("Link Error: aggregate object bytes exceed {MAX_LINK_INPUT_BYTES}");
                return EXIT_ERROR;
            }
        };
        match ObjectModule::from_bytes(&bytes) {
            Ok(object) => objects.push(object),
            Err(error) => {
                eprintln!("Link Error: invalid object '{path}': {error}");
                return EXIT_ERROR;
            }
        }
    }
    let linked = match link_with_metadata(&objects) {
        Ok(linked) => linked,
        Err(error) => {
            eprintln!("Link Error: {error}");
            return EXIT_ERROR;
        }
    };
    if let Err(error) = verify_bytecode(&linked.code) {
        eprintln!("Link Error: linked bytecode verification failed: {error}");
        return EXIT_ERROR;
    }
    let artifact =
        match PortableArtifact::from_linked(linked).and_then(|artifact| artifact.to_bytes()) {
            Ok(artifact) => artifact,
            Err(error) => {
                eprintln!("Link Error: cannot serialize linked bytecode: {error}");
                return EXIT_ERROR;
            }
        };
    match write_binary_file(output, &artifact) {
        Ok(()) => {
            println!("Wrote linked bytecode: {output}");
            EXIT_OK
        }
        Err(error) => {
            eprintln!("Link Error: {error}");
            EXIT_ERROR
        }
    }
}

/// Parse the value following a flag, failing loudly on garbage. A silently
/// substituted default would mask a typo in a resource limit.
fn parse_flag_value<T: std::str::FromStr>(
    args: &[String],
    flag: &str,
    default: T,
) -> Result<T, String> {
    match args.iter().position(|a| a == flag) {
        None => Ok(default),
        Some(idx) => match args.get(idx + 1) {
            None => Err(format!("{} requires a value", flag)),
            Some(raw) => raw
                .parse()
                .map_err(|_| format!("{} requires a number, got '{}'", flag, raw)),
        },
    }
}

fn repeated_flag_values(args: &[String], flag: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter_map(|(index, argument)| {
            (argument == flag)
                .then(|| args.get(index + 1).cloned())
                .flatten()
        })
        .collect()
}

fn read_bounded_file_handle(
    file: fs::File,
    path: &Path,
    limit: usize,
    description: &str,
) -> Result<Vec<u8>, String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect {description} '{}': {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{description} '{}' is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > limit as u64 {
        return Err(format!(
            "{description} '{}' exceeds bounded read limit {limit} bytes",
            path.display()
        ));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        format!(
            "{description} '{}' length does not fit this platform",
            path.display()
        )
    })?;
    let take_limit = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    let mut bytes = Vec::with_capacity(capacity);
    file.take(take_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {description} '{}': {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!(
            "{description} '{}' grew beyond bounded read limit {limit} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

fn read_bounded_regular_file<P: AsRef<Path>>(
    path: P,
    limit: usize,
    description: &str,
) -> Result<Vec<u8>, String> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .map_err(|error| format!("cannot open {description} '{}': {error}", path.display()))?;
    read_bounded_file_handle(file, path, limit, description)
}

fn freeze_file_capabilities(
    read_paths: &[String],
    write_paths: &[String],
) -> Result<Vec<ourochronos::FrozenFileSnapshot>, String> {
    let defaults = BytecodeVmConfig::default();
    let mut paths = std::collections::BTreeSet::new();
    paths.extend(read_paths.iter().cloned());
    paths.extend(write_paths.iter().cloned());
    let mut aggregate = 0usize;
    let mut snapshots = Vec::with_capacity(paths.len());
    for path in paths {
        if path.is_empty() {
            return Err("file capability paths must not be empty".to_string());
        }
        let contents = match fs::File::open(&path) {
            Ok(file) => {
                let remaining = defaults.max_file_snapshot_bytes.saturating_sub(aggregate);
                let limit = defaults.max_collection_items.min(remaining);
                let bytes =
                    read_bounded_file_handle(file, Path::new(&path), limit, "file capability")?;
                aggregate = aggregate
                    .checked_add(bytes.len())
                    .ok_or_else(|| "frozen file byte count overflowed".to_string())?;
                if aggregate > defaults.max_file_snapshot_bytes {
                    return Err(format!(
                        "frozen file bytes exceed {}",
                        defaults.max_file_snapshot_bytes
                    ));
                }
                Some(bytes)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(format!("cannot open file capability '{path}': {error}")),
        };
        snapshots.push(ourochronos::FrozenFileSnapshot { path, contents });
    }
    Ok(snapshots)
}

fn freeze_endpoint_tapes(
    input_specs: &[String],
    send_endpoints: &[String],
) -> Result<Vec<ourochronos::FrozenEndpointTape>, String> {
    let defaults = BytecodeVmConfig::default();
    let mut tapes = std::collections::BTreeMap::<String, Vec<u8>>::new();
    let mut aggregate = 0usize;
    for spec in input_specs {
        let (endpoint, path) = spec
            .split_once('=')
            .ok_or_else(|| format!("--network-input requires <endpoint>=<file>, got '{spec}'"))?;
        if endpoint.is_empty() || path.is_empty() {
            return Err(format!(
                "--network-input requires non-empty endpoint and file, got '{spec}'"
            ));
        }
        if tapes.contains_key(endpoint) {
            return Err(format!("duplicate frozen endpoint '{endpoint}'"));
        }
        let remaining = defaults.max_endpoint_tape_bytes.saturating_sub(aggregate);
        let limit = defaults.max_collection_items.min(remaining);
        let bytes = read_bounded_regular_file(path, limit, "frozen endpoint input")?;
        if bytes.len() > defaults.max_collection_items {
            return Err(format!(
                "frozen endpoint '{endpoint}' exceeds per-stream limit {}",
                defaults.max_collection_items
            ));
        }
        aggregate = aggregate
            .checked_add(bytes.len())
            .ok_or_else(|| "frozen endpoint byte count overflowed".to_string())?;
        if aggregate > defaults.max_endpoint_tape_bytes {
            return Err(format!(
                "frozen endpoint bytes exceed {}",
                defaults.max_endpoint_tape_bytes
            ));
        }
        tapes.insert(endpoint.to_string(), bytes);
    }
    for endpoint in send_endpoints {
        if endpoint.is_empty() {
            return Err("network capability endpoints must not be empty".to_string());
        }
        // A virtual connection needs a deterministic receive stream even when
        // the source only sends. An explicitly frozen input wins over this
        // empty EOF stream.
        tapes.entry(endpoint.clone()).or_default();
    }
    Ok(tapes
        .into_iter()
        .map(|(endpoint, recv_bytes)| ourochronos::FrozenEndpointTape {
            endpoint,
            recv_bytes,
        })
        .collect())
}

fn freeze_process_results(
    descriptor_paths: &[String],
) -> Result<Vec<ourochronos::FrozenProcessResult>, String> {
    const MAGIC: &[u8] = b"OUROPROCESS/1\n";
    let defaults = BytecodeVmConfig::default();
    let mut results = Vec::with_capacity(descriptor_paths.len());
    let mut commands = std::collections::BTreeSet::new();
    let mut aggregate = 0usize;
    for path in descriptor_paths {
        const MAX_DESCRIPTOR_OVERHEAD: usize = 4_096;
        let remaining = defaults.max_process_result_bytes.saturating_sub(aggregate);
        let descriptor_limit = remaining
            .min(defaults.max_collection_items.saturating_mul(2))
            .saturating_add(MAX_DESCRIPTOR_OVERHEAD);
        let bytes = read_bounded_regular_file(path, descriptor_limit, "process descriptor")?;
        let rest = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| format!("process descriptor '{path}' must begin with OUROPROCESS/1"))?;
        let (exit_line, rest) = split_descriptor_line(rest, path, "exit code")?;
        let exit_text = std::str::from_utf8(exit_line)
            .map_err(|_| format!("process descriptor '{path}' has a non-UTF-8 exit code"))?;
        let exit_code = exit_text.parse::<i32>().map_err(|_| {
            format!("process descriptor '{path}' has invalid i32 exit code '{exit_text}'")
        })?;
        let (length_line, payload) = split_descriptor_line(rest, path, "command length")?;
        let length_text = std::str::from_utf8(length_line)
            .map_err(|_| format!("process descriptor '{path}' has a non-UTF-8 command length"))?;
        let command_length = length_text.parse::<usize>().map_err(|_| {
            format!("process descriptor '{path}' has invalid command length '{length_text}'")
        })?;
        if command_length > defaults.max_collection_items || command_length > payload.len() {
            return Err(format!(
                "process descriptor '{path}' command length {command_length} is out of bounds"
            ));
        }
        let (command_bytes, output) = payload.split_at(command_length);
        let command = std::str::from_utf8(command_bytes)
            .map_err(|_| format!("process descriptor '{path}' command is not UTF-8"))?
            .to_string();
        if !commands.insert(command.clone()) {
            return Err(format!("duplicate frozen process command {command:?}"));
        }
        if output.len() > defaults.max_collection_items {
            return Err(format!(
                "process descriptor '{path}' output exceeds per-result limit {}",
                defaults.max_collection_items
            ));
        }
        let retained_bytes = command
            .len()
            .checked_add(output.len())
            .ok_or_else(|| "frozen process retained byte count overflowed".to_string())?;
        aggregate = aggregate
            .checked_add(retained_bytes)
            .ok_or_else(|| "frozen process byte count overflowed".to_string())?;
        if aggregate > defaults.max_process_result_bytes {
            return Err(format!(
                "frozen process bytes exceed {}",
                defaults.max_process_result_bytes
            ));
        }
        results.push(ourochronos::FrozenProcessResult {
            command,
            output: output.to_vec(),
            exit_code,
        });
    }
    Ok(results)
}

fn split_descriptor_line<'a>(
    bytes: &'a [u8],
    path: &str,
    field: &str,
) -> Result<(&'a [u8], &'a [u8]), String> {
    let Some(index) = bytes.iter().position(|byte| *byte == b'\n') else {
        return Err(format!(
            "process descriptor '{path}' is missing its {field} line"
        ));
    };
    Ok((&bytes[..index], &bytes[index + 1..]))
}

fn native_effect_adapter(
    file_paths: &[String],
    endpoints: &[String],
    processes: &[ourochronos::FrozenProcessResult],
    max_sleep_milliseconds: Option<u64>,
) -> ourochronos::NativeEffectAdapter {
    let mut adapter = ourochronos::NativeEffectAdapter::new();
    for path in file_paths {
        adapter.allow_file(path);
    }
    for endpoint in endpoints {
        adapter.allow_endpoint(endpoint);
    }
    for process in processes {
        allow_shell_command(&mut adapter, &process.command);
    }
    if let Some(maximum) = max_sleep_milliseconds {
        adapter.allow_sleep(maximum);
    }
    adapter
}

#[cfg(not(windows))]
fn allow_shell_command(adapter: &mut ourochronos::NativeEffectAdapter, command: &str) {
    adapter.allow_process_with_arguments("sh", ["-c", command]);
}

#[cfg(windows)]
fn allow_shell_command(adapter: &mut ourochronos::NativeEffectAdapter, command: &str) {
    adapter.allow_process_with_arguments("cmd", ["/C", command]);
}

/// The four numeric limits, each defaulted when absent.
fn parse_numeric_flags(args: &[String]) -> Result<(u64, usize, u64, usize, usize), String> {
    Ok((
        parse_flag_value(args, "--seed", 0)?,
        parse_flag_value(args, "--seeds", 4)?,
        parse_flag_value(args, "--max-inst", 10_000_000)?,
        parse_flag_value(
            args,
            "--provenance-limit",
            ourochronos::DEFAULT_PROVENANCE_SATURATION_LIMIT,
        )?,
        parse_flag_value(args, "--memory-cells", ourochronos::MEMORY_SIZE)?,
    ))
}

/// Walk the arguments after the filename against the flag table: unknown
/// options, stray positionals, missing values, and repeated flags are all
/// errors. parse_flag_value's first-occurrence lookup is sound because a
/// second occurrence never gets this far.
fn validate_args(args: &[String]) -> Result<(), String> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut i = 2; // args[0] = binary, args[1] = filename
    while i < args.len() {
        let arg = &args[i];
        let Some(flag) = find_flag(arg) else {
            return Err(if arg.starts_with('-') {
                format!("unknown option '{}'", arg)
            } else {
                format!("unexpected argument '{}'", arg)
            });
        };
        let repeatable = matches!(
            flag.name,
            "--allow-file-read"
                | "--allow-file-write"
                | "--network-input"
                | "--allow-network-send"
                | "--allow-process"
        );
        if seen.contains(&flag.name) && !repeatable {
            return Err(format!("{} specified more than once", flag.name));
        }
        if !repeatable {
            seen.push(flag.name);
        }
        i += match flag.arity {
            Arity::None => 1,
            Arity::Value => {
                if args.get(i + 1).is_none() {
                    return Err(format!("{} requires a value", flag.name));
                }
                2
            }
            Arity::OptionalValue => {
                if args.get(i + 1).is_some_and(|a| !a.starts_with('-')) {
                    2
                } else {
                    1
                }
            }
        };
    }
    Ok(())
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = env::args().collect();

    if args.iter().skip(1).any(|a| a == "--help" || a == "-h") {
        print_usage();
        return EXIT_OK;
    }
    if args.iter().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("ourochronos {}", env!("CARGO_PKG_VERSION"));
        println!(
            "platform: {}-{}; runtime ABI {}; native effect commits: Linux only",
            std::env::consts::ARCH,
            std::env::consts::OS,
            ourochronos::package::CURRENT_RUNTIME_ABI
        );
        println!(
            "compiled features: lsp={} dynamic-ffi={}; Z3 native library required",
            cfg!(feature = "lsp"),
            cfg!(feature = "dynamic-ffi")
        );
        return EXIT_OK;
    }

    // LSP mode requires no file argument.
    #[cfg(feature = "lsp")]
    if args.contains(&"--lsp".to_string()) {
        eprintln!("Starting OUROCHRONOS Language Server...");
        if let Err(e) = ourochronos::tooling::lsp::server::run_server() {
            eprintln!("LSP Server error: {}", e);
            return EXIT_ERROR;
        }
        return EXIT_OK;
    }

    #[cfg(not(feature = "lsp"))]
    if args.contains(&"--lsp".to_string()) {
        eprintln!("LSP support not enabled. Rebuild with: cargo build --features lsp");
        return EXIT_ERROR;
    }

    if args.len() == 1 {
        if let Some(status) = run_embedded_package() {
            return status;
        }
        print_usage();
        return EXIT_ERROR;
    }

    if args[1] == "repl" {
        let mut repl = ourochronos::Repl::new();
        return match repl.run() {
            Ok(()) => EXIT_OK,
            Err(e) => {
                eprintln!("REPL error: {}", e);
                EXIT_ERROR
            }
        };
    }

    if args[1] == "run-package" {
        return run_portable_package(&args);
    }
    if matches!(args[1].as_str(), "prove-finite" | "check-finite") {
        return run_finite_certificate(&args);
    }
    if matches!(args[1].as_str(), "halt-slice" | "halt-resume") {
        return run_halting_checkpoint(&args);
    }
    if args[1] == "analyse-affine" {
        return run_affine_analysis(&args);
    }
    if args[1] == "isolate" {
        return run_isolated(&args);
    }
    if args[1] == "link" {
        return link_portable_objects(&args);
    }

    let filename = &args[1];
    if filename.starts_with('-') {
        return fail_usage(&format!(
            "expected a programme file, got option '{}'",
            filename
        ));
    }
    if let Err(msg) = validate_args(&args) {
        return fail_usage(&msg);
    }

    let diagnostic = args.contains(&"--diagnostic".to_string());
    let deutsch_mode = args.contains(&"--deutsch".to_string());
    let action_mode = args.contains(&"--action".to_string());
    let smt = args.contains(&"--smt".to_string());
    let global_mode = args.contains(&"--global".to_string());
    let all_fixed_mode = args.contains(&"--all-fixed".to_string());
    let recurrent_mode = args.contains(&"--recurrent".to_string());
    let verify_mode = args.contains(&"--verify".to_string());
    let family_input = match args.iter().position(|arg| arg == "--verify-family") {
        None => None,
        Some(index) => {
            let raw = args
                .get(index + 1)
                .expect("argument validation requires a --verify-family value");
            if raw.is_empty() {
                return fail_usage("--verify-family requires a nonempty binary input string");
            }
            if raw.len() > ourochronos::MAX_RECURRENT_FROZEN_INPUTS {
                return fail_usage(&format!(
                    "--verify-family input exceeds {} bits",
                    ourochronos::MAX_RECURRENT_FROZEN_INPUTS
                ));
            }
            if !raw.bytes().all(|bit| matches!(bit, b'0' | b'1')) {
                return fail_usage("--verify-family requires a binary input string");
            }
            Some(raw.bytes().map(|bit| bit - b'0').collect::<Vec<_>>())
        }
    };
    let family_input_bits = family_input.as_ref().map(|input| input.len() as u64);
    let family_mode = family_input_bits.is_some();
    let artifact_path = args
        .iter()
        .position(|arg| arg == "--artifact")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let dot_path = args
        .iter()
        .position(|arg| arg == "--dot")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let stationary_mode = args.contains(&"--stationary".to_string());
    let quantum_mode = args.contains(&"--quantum-fixed".to_string());
    let typecheck_mode = args.contains(&"--typecheck".to_string());
    let resources_mode = args.contains(&"--resources".to_string());
    let check_only = args.contains(&"--check".to_string());
    let object_path = args
        .iter()
        .position(|arg| arg == "--emit-object")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let objects_directory = args
        .iter()
        .position(|arg| arg == "--emit-objects")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let bytecode_path = args
        .iter()
        .position(|arg| arg == "--emit-bytecode")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let package_path = args
        .iter()
        .position(|arg| arg == "--build")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let launcher_path = args
        .iter()
        .position(|arg| arg == "--build-executable")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let embed_global_witness = args.contains(&"--embed-global-witness".to_string());
    let runtime_global_package = args.contains(&"--runtime-global-package".to_string());
    let halting_bound = match args.iter().position(|arg| arg == "--halting-bound") {
        None => None,
        Some(index) => match args.get(index + 1).and_then(|raw| raw.parse::<u64>().ok()) {
            Some(0) | None => return fail_usage("--halting-bound requires a positive integer"),
            Some(bound) => Some(bound),
        },
    };
    let fast_mode = args.contains(&"--fast".to_string());
    let strict_mode = args.contains(&"--strict".to_string());
    let permissive_mode = args.contains(&"--permissive".to_string());

    let build_action_count = usize::from(check_only)
        + usize::from(object_path.is_some())
        + usize::from(objects_directory.is_some())
        + usize::from(bytecode_path.is_some())
        + usize::from(package_path.is_some())
        + usize::from(launcher_path.is_some());
    if build_action_count > 1 {
        return fail_usage(
            "--check, --emit-object, --emit-objects, --emit-bytecode, --build, and --build-executable are mutually exclusive",
        );
    }
    if embed_global_witness && package_path.is_none() && launcher_path.is_none() {
        return fail_usage("--embed-global-witness requires --build or --build-executable");
    }
    if runtime_global_package && package_path.is_none() && launcher_path.is_none() {
        return fail_usage("--runtime-global-package requires --build or --build-executable");
    }
    if embed_global_witness && runtime_global_package {
        return fail_usage(
            "--embed-global-witness and --runtime-global-package are mutually exclusive",
        );
    }
    if build_action_count != 0
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || family_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage(
            "compiler/build actions cannot be combined with an execution/search mode",
        );
    }

    if [diagnostic, deutsch_mode, action_mode]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
        > 1
    {
        return fail_usage("--diagnostic, --deutsch, and --action are mutually exclusive");
    }
    if stationary_mode && quantum_mode {
        return fail_usage("--stationary and --quantum-fixed are mutually exclusive");
    }
    if stationary_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--stationary cannot be combined with an execution/search mode");
    }
    if quantum_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--quantum-fixed cannot be combined with an execution/search mode");
    }
    if strict_mode && permissive_mode {
        return fail_usage("--strict and --permissive are mutually exclusive");
    }
    if global_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--global cannot be combined with another execution/search mode");
    }
    if all_fixed_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || recurrent_mode
            || verify_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--all-fixed cannot be combined with another execution/search mode");
    }
    if recurrent_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || verify_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--recurrent cannot be combined with another execution/search mode");
    }
    if verify_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--verify cannot be combined with another execution/search mode");
    }
    if family_mode
        && (diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || stationary_mode
            || quantum_mode
            || fast_mode
            || halting_bound.is_some())
    {
        return fail_usage("--verify-family cannot be combined with another execution/search mode");
    }
    if artifact_path.is_some() && !(global_mode || all_fixed_mode || verify_mode || family_mode) {
        return fail_usage(
            "--artifact requires --global, --all-fixed, --verify, or --verify-family",
        );
    }
    if dot_path.is_some() && !recurrent_mode {
        return fail_usage("--dot requires --recurrent");
    }
    if (args.iter().any(|arg| arg == "--solver-timeout")
        || args.iter().any(|arg| arg == "--loop-unroll"))
        && !(global_mode || all_fixed_mode || verify_mode || smt)
    {
        return fail_usage(
            "--solver-timeout and --loop-unroll require --global, --all-fixed, --verify, or --smt",
        );
    }

    let error_config = if strict_mode {
        ErrorConfig::strict()
    } else if permissive_mode {
        ErrorConfig::permissive()
    } else {
        ErrorConfig::default()
    };

    let (seed, num_seeds, max_instructions, provenance_limit, mut memory_cells) =
        match parse_numeric_flags(&args) {
            Ok(values) => values,
            Err(msg) => return fail_usage(&msg),
        };
    let solver_timeout_ms = match parse_flag_value(&args, "--solver-timeout", 30_000u64) {
        Ok(value) => value,
        Err(msg) => return fail_usage(&msg),
    };
    let loop_unroll_limit = match parse_flag_value(&args, "--loop-unroll", 10usize) {
        Ok(value) => value,
        Err(msg) => return fail_usage(&msg),
    };
    let state_bits = match parse_flag_value(&args, "--state-bits", 1u8) {
        Ok(value) => value,
        Err(msg) => return fail_usage(&msg),
    };
    let state_limit = match parse_flag_value(&args, "--state-limit", 65_536usize) {
        Ok(value) => value,
        Err(msg) => return fail_usage(&msg),
    };

    if max_instructions == 0 {
        return fail_usage("--max-inst must be greater than 0");
    }
    if provenance_limit == 0 {
        return fail_usage("--provenance-limit must be greater than 0");
    }
    if memory_cells == 0 {
        return fail_usage("--memory-cells must be greater than 0");
    }
    if solver_timeout_ms == 0 {
        return fail_usage("--solver-timeout must be greater than 0");
    }
    if !(1..=64).contains(&state_bits) {
        return fail_usage("--state-bits must be in 1..=64");
    }
    if state_limit == 0 {
        return fail_usage("--state-limit must be greater than 0");
    }
    if (args.iter().any(|arg| arg == "--state-bits")
        || args.iter().any(|arg| arg == "--state-limit"))
        && !(recurrent_mode || family_mode)
    {
        return fail_usage("--state-bits and --state-limit require --recurrent or --verify-family");
    }
    if action_mode && num_seeds == 0 {
        return fail_usage("--seeds must be greater than 0 in action mode");
    }

    let effects_policy = match args.iter().position(|a| a == "--effects") {
        None => ourochronos::vm::EffectsPolicy::Decline,
        Some(idx) => match args.get(idx + 1).map(|s| s.as_str()) {
            Some("decline") => ourochronos::vm::EffectsPolicy::Decline,
            Some("unrestricted") => ourochronos::vm::EffectsPolicy::Unrestricted,
            other => {
                return fail_usage(&format!(
                    "--effects requires 'decline' or 'unrestricted', got '{}'",
                    other.unwrap_or("")
                ));
            }
        },
    };
    let file_read_paths = repeated_flag_values(&args, "--allow-file-read");
    let file_write_paths = repeated_flag_values(&args, "--allow-file-write");
    let network_input_specs = repeated_flag_values(&args, "--network-input");
    let network_send_endpoints = repeated_flag_values(&args, "--allow-network-send");
    let process_descriptor_paths = repeated_flag_values(&args, "--allow-process");
    let max_sleep_milliseconds = match args.iter().position(|arg| arg == "--allow-sleep-ms") {
        None => None,
        Some(index) => match args
            .get(index + 1)
            .and_then(|value| value.parse::<u64>().ok())
        {
            Some(value) => Some(value),
            None => return fail_usage("--allow-sleep-ms requires a non-negative integer"),
        },
    };
    let has_runtime_capabilities = !file_read_paths.is_empty()
        || !file_write_paths.is_empty()
        || !network_input_specs.is_empty()
        || !network_send_endpoints.is_empty()
        || !process_descriptor_paths.is_empty()
        || max_sleep_milliseconds.is_some();
    if has_runtime_capabilities
        && (build_action_count != 0
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || family_mode
            || stationary_mode
            || quantum_mode
            || halting_bound.is_some())
    {
        return fail_usage(
            "host capabilities are runtime-only and cannot be combined with build, proof, recurrent, declarative, or halting modes",
        );
    }
    if (global_mode || all_fixed_mode || recurrent_mode || verify_mode || family_mode)
        && args.iter().any(|arg| arg == "--effects")
    {
        return fail_usage(
            "global analysis uses replay-safe deterministic effects and cannot be combined with --effects",
        );
    }

    let audit_json = args.contains(&"--audit-json".to_string());
    if let Some(idx) = args.iter().position(|a| a == "--audit") {
        let audit_path = if idx + 1 < args.len() && !args[idx + 1].starts_with('-') {
            args[idx + 1].clone()
        } else {
            "ourochronos-audit.log".to_string()
        };

        let config = AuditConfig {
            log_path: std::path::PathBuf::from(&audit_path),
            min_severity: Severity::Info,
            echo_stdout: diagnostic,
            format: if audit_json {
                AuditFormat::JsonLines
            } else {
                AuditFormat::Text
            },
        };

        if let Err(e) = audit::init_global_logger(config) {
            eprintln!("Warning: Could not initialize audit logger: {}", e);
        } else if diagnostic {
            println!("Audit logging to: {}", audit_path);
        }
    }

    audit::audit(
        AuditEntry::new("STARTUP", "System", "ourochronos", "CLI session started")
            .with_category(ActionCategory::System)
            .with_meta("file", filename)
            .with_meta(
                "mode",
                if verify_mode {
                    "verify"
                } else if family_mode {
                    "verify-family"
                } else if recurrent_mode {
                    "recurrent"
                } else if all_fixed_mode {
                    "all-fixed"
                } else if global_mode {
                    "global"
                } else if action_mode {
                    "action"
                } else if deutsch_mode {
                    "deutsch"
                } else if fast_mode {
                    "fast"
                } else {
                    "standard"
                },
            ),
    );

    let parse_start = Instant::now();
    let graph = match ModuleGraph::load(filename, ourochronos::StdLib::procedures()) {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("Compile Error: {error}");
            return EXIT_ERROR;
        }
    };

    let parse_duration = parse_start.elapsed();
    audit::audit(
        AuditEntry::new("PARSE", "Program", filename, "Parsed successfully")
            .with_category(ActionCategory::Parse)
            .with_duration_us(parse_duration.as_micros() as u64)
            .with_meta("statements", graph.program().body.len().to_string())
            .with_meta("procedures", graph.program().procedures.len().to_string()),
    );

    let resolve_start = Instant::now();
    let resolved_hir = match graph.resolve_hir() {
        Ok(program) => program,
        Err(errors) => {
            eprintln!("Compile Error: typed name resolution failed");
            for error in errors {
                eprintln!("  - {error}");
            }
            return EXIT_ERROR;
        }
    };
    let parsed_program = graph.program().clone();
    let mut family_width_rule = None;
    if let Some(input_bits) = family_input_bits {
        let Some(declaration) = &parsed_program.family_declaration else {
            return fail_usage("--verify-family requires a source FAMILY declaration");
        };
        let contract = ourochronos::PspaceFamilyContract::from(declaration);
        let width_rule = if args.iter().any(|arg| arg == "--memory-cells") {
            ourochronos::PolynomialBound {
                coefficient: 0,
                degree: 1,
                additive: memory_cells as u64,
            }
        } else {
            contract.ctc_cells
        };
        let Some(width) = width_rule.evaluate(input_bits) else {
            return fail_usage("generated temporal-width polynomial overflows u128");
        };
        if width == 0 {
            return fail_usage("CTC_CELLS must be positive for --verify-family");
        }
        let Ok(generated_width) = usize::try_from(width) else {
            return fail_usage("CTC_CELLS is not representable on this host");
        };
        memory_cells = generated_width;
        family_width_rule = Some(width_rule);
    }
    let semantics_report = check_semantics(&resolved_hir);
    if !semantics_report.is_accepted_for_interpreter() {
        eprintln!("Compile Error: mandatory structural stack analysis failed");
        if diagnostic {
            for summary in &semantics_report.procedures {
                eprintln!("  procedure {}: {:?}", summary.id, summary.stack);
            }
        }
        for error in &semantics_report.errors {
            eprintln!(
                "  - {:?} at {:?} {:?}",
                error.kind, error.site.owner, error.site.context
            );
        }
        return EXIT_ERROR;
    }
    if let Some(obligation) = semantics_report.obligations.iter().find(|obligation| {
        matches!(
            &obligation.kind,
            ObligationKind::DynamicOpcode {
                opcode: OpCode::FFICall | OpCode::FFICallNamed,
                ..
            }
        )
    }) {
        eprintln!(
            "Compile Error: foreign execution is not linked: dynamic dispatch is outside the scalar ABI for {:?}",
            obligation.kind
        );
        return EXIT_ERROR;
    }

    // Run source-wide gates before object construction so CLI diagnostics can
    // retain the complete reports. The object compiler independently repeats
    // these gates and seals the linked set, preventing library callers from
    // bypassing them.
    let region_report = ourochronos::TemporalRegionReport::analyze(&parsed_program, memory_cells);
    let typecheck_result = type_check(&parsed_program);

    if typecheck_mode {
        println!("=== Temporal Type Analysis ===");
        println!("{}", types::display_types(&typecheck_result));
        if !region_report.regions.is_empty() || !region_report.host_effects.is_empty() {
            println!("=== Finite Temporal Region Contracts ===");
            println!("{}", region_report);
        }
        println!(); // blank line before execution
    }

    if !typecheck_result.is_valid {
        if !typecheck_mode {
            eprintln!("=== Mandatory Semantic Analysis ===");
            eprintln!("{}", types::display_types(&typecheck_result));
        }
        eprintln!("Type errors found. Stopping.");
        return EXIT_ERROR;
    }

    if !region_report.is_valid() {
        if !typecheck_mode {
            eprintln!("=== Finite Temporal Region Contracts ===");
            eprintln!("{}", region_report);
        }
        eprintln!("Temporal region contract errors found. Stopping.");
        return EXIT_ERROR;
    }

    let artifact_name = Path::new(filename)
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("ourochronos-program")
        .to_string();
    let objects = match graph.compile_objects_with_memory(memory_cells) {
        Ok(objects) => objects,
        Err(error) => {
            eprintln!("Compile Error: per-source object construction failed: {error}");
            return EXIT_ERROR;
        }
    };
    let linked = match link_with_metadata(&objects) {
        Ok(linked) => linked,
        Err(error) => {
            eprintln!("Compile Error: bytecode link failed: {error}");
            return EXIT_ERROR;
        }
    };
    let linked_metadata = linked.metadata;
    let bytecode = linked.code;
    let bytecode_verification = match verify_bytecode(&bytecode) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("Compile Error: bytecode verification failed: {error}");
            return EXIT_ERROR;
        }
    };
    let native_temporal_ir = if smt {
        match ourochronos::lower_bytecode_temporal_ir(
            &bytecode,
            TemporalIrConfig {
                memory_cells,
                loop_unroll_limit,
                bounds_policy: error_config.memory_bounds,
            },
        ) {
            Ok(ir) => Some(ir),
            Err(error) => {
                eprintln!("Compile Error: native bytecode temporal lowering failed: {error}");
                return EXIT_ERROR;
            }
        }
    } else {
        None
    };
    let bytecode_runtime_mode = build_action_count == 0
        && !smt
        && !global_mode
        && !all_fixed_mode
        && !recurrent_mode
        && !verify_mode
        && !family_mode
        && !stationary_mode
        && !quantum_mode
        && halting_bound.is_none()
        && parsed_program.markov_declaration.is_none()
        && parsed_program.quantum_declaration.is_none();
    audit::audit(
        AuditEntry::new("RESOLVE", "Program", filename, "Resolved typed HIR")
            .with_category(ActionCategory::Parse)
            .with_duration_us(resolve_start.elapsed().as_micros() as u64)
            .with_meta("procedures", resolved_hir.procedures.len().to_string())
            .with_meta("foreigns", resolved_hir.foreigns.len().to_string())
            .with_meta("quotations", resolved_hir.quotes.len().to_string())
            .with_meta(
                "runtime_stack_obligations",
                semantics_report.obligations.len().to_string(),
            )
            .with_meta(
                "bytecode_instructions",
                bytecode.instructions.len().to_string(),
            )
            .with_meta(
                "bytecode_runtime_obligations",
                bytecode_verification.obligations.len().to_string(),
            ),
    );

    // Source syntax is retained for declarations, resource reporting, and
    // source-oriented diagnostics only. Every executable policy consumes the
    // linked bytecode above; no mode preprocesses or interprets this AST.
    let program = parsed_program;

    if let Some(path) = object_path {
        let Some(root_index) = graph
            .modules()
            .iter()
            .position(|module| module.source == graph.root())
        else {
            eprintln!("Compile Error: entry source module was not retained");
            return EXIT_ERROR;
        };
        let Some(object) = objects.get(root_index) else {
            eprintln!("Compile Error: entry source object was not produced");
            return EXIT_ERROR;
        };
        let artifact = match object.to_bytes() {
            Ok(artifact) => artifact,
            Err(error) => {
                eprintln!("Compile Error: cannot serialize object: {error}");
                return EXIT_ERROR;
            }
        };
        return match write_binary_file(&path, &artifact) {
            Ok(()) => {
                println!("Wrote relocatable object: {path}");
                EXIT_OK
            }
            Err(error) => {
                eprintln!("Build Error: {error}");
                EXIT_ERROR
            }
        };
    }

    if let Some(directory) = objects_directory {
        let mut artifacts = Vec::with_capacity(objects.len());
        for (index, object) in objects.iter().enumerate() {
            let artifact = match object.to_bytes() {
                Ok(artifact) => artifact,
                Err(error) => {
                    eprintln!(
                        "Compile Error: cannot serialize source object '{}': {error}",
                        object.name
                    );
                    return EXIT_ERROR;
                }
            };
            artifacts.push((
                object.name.clone(),
                object_output_name(index, &object.name),
                artifact,
            ));
        }
        match fs::read_dir(&directory) {
            Ok(mut entries) => {
                if entries.next().is_some() {
                    eprintln!(
                        "Build Error: object directory '{directory}' must be empty so stale modules cannot survive a rebuild"
                    );
                    return EXIT_ERROR;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!("Build Error: cannot inspect object directory '{directory}': {error}");
                return EXIT_ERROR;
            }
        }
        if let Err(error) = fs::create_dir_all(&directory) {
            eprintln!("Build Error: cannot create object directory '{directory}': {error}");
            return EXIT_ERROR;
        }
        for (name, file_name, artifact) in artifacts {
            let output = Path::new(&directory).join(file_name);
            if let Err(error) = write_binary_file(&output.to_string_lossy(), &artifact) {
                eprintln!("Build Error: {error}");
                return EXIT_ERROR;
            }
            println!("Wrote source object '{name}': {}", output.display());
        }
        return EXIT_OK;
    }

    if let Some(path) = bytecode_path {
        let artifact = match PortableArtifact::from_linked(ourochronos::LinkedProgram {
            code: bytecode,
            metadata: linked_metadata,
        })
        .and_then(|artifact| artifact.to_bytes())
        {
            Ok(artifact) => artifact,
            Err(error) => {
                eprintln!("Compile Error: cannot serialize bytecode: {error}");
                return EXIT_ERROR;
            }
        };
        return match write_binary_file(&path, &artifact) {
            Ok(()) => {
                println!("Wrote linked bytecode: {path}");
                EXIT_OK
            }
            Err(error) => {
                eprintln!("Build Error: {error}");
                EXIT_ERROR
            }
        };
    }

    if let Some(path) = package_path {
        let package = match construct_linked_package_artifact(
            artifact_name,
            memory_cells,
            max_instructions,
            error_config.memory_bounds,
            ourochronos::LinkedProgram {
                code: bytecode,
                metadata: linked_metadata,
            },
            embed_global_witness,
            runtime_global_package,
            solver_timeout_ms,
            loop_unroll_limit,
        ) {
            Ok(package) => package,
            Err(error) => {
                eprintln!("Build Error: cannot construct package: {error}");
                return EXIT_ERROR;
            }
        };
        let artifact = match package.to_bytes() {
            Ok(artifact) => artifact,
            Err(error) => {
                eprintln!("Build Error: cannot serialize package: {error}");
                return EXIT_ERROR;
            }
        };
        return match write_binary_file(&path, &artifact) {
            Ok(()) => {
                println!("Wrote portable package: {path}");
                EXIT_OK
            }
            Err(error) => {
                eprintln!("Build Error: {error}");
                EXIT_ERROR
            }
        };
    }

    if let Some(path) = launcher_path {
        let package = match construct_linked_package_artifact(
            artifact_name,
            memory_cells,
            max_instructions,
            error_config.memory_bounds,
            ourochronos::LinkedProgram {
                code: bytecode,
                metadata: linked_metadata,
            },
            embed_global_witness,
            runtime_global_package,
            solver_timeout_ms,
            loop_unroll_limit,
        ) {
            Ok(package) => package,
            Err(error) => {
                eprintln!("Build Error: cannot construct embedded package: {error}");
                return EXIT_ERROR;
            }
        };
        let executable = match env::current_exe() {
            Ok(executable) => executable,
            Err(error) => {
                eprintln!("Build Error: cannot locate the current runtime: {error}");
                return EXIT_ERROR;
            }
        };
        let runtime = match read_bounded_regular_file(
            &executable,
            ourochronos::MAX_LAUNCHER_BYTES,
            "launcher runtime",
        ) {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!(
                    "Build Error: cannot read runtime '{}': {error}",
                    executable.display()
                );
                return EXIT_ERROR;
            }
        };
        let launcher = match build_native_artifact_launcher(&runtime, &package) {
            Ok(launcher) => launcher,
            Err(error) => {
                eprintln!("Build Error: cannot construct native launcher: {error}");
                return EXIT_ERROR;
            }
        };
        return match write_native_launcher_file(&path, &launcher, &executable) {
            Ok(()) => {
                println!("Wrote native executable: {path}");
                EXIT_OK
            }
            Err(error) => {
                eprintln!("Build Error: {error}");
                EXIT_ERROR
            }
        };
    }

    if check_only {
        println!("Check succeeded: typed HIR, semantics, bytecode, link, effects, and regions");
        return EXIT_OK;
    }

    if resources_mode {
        let semantics = if deutsch_mode {
            "Deutsch stationary-cycle"
        } else if family_mode {
            "exhaustive finite FAMILY-instance verification"
        } else if global_mode {
            "global symbolic point fixed state"
        } else if all_fixed_mode {
            "all-point-fixed-state verification"
        } else if recurrent_mode {
            "complete small-state recurrent analysis"
        } else if verify_mode {
            "all-fixed temporal property verification"
        } else if action_mode {
            "action-guided point fixed state"
        } else if diagnostic {
            "diagnostic point fixed state"
        } else {
            "standard point fixed state"
        };
        let profile = ourochronos::ConcreteResourceProfile::for_program(
            &program,
            memory_cells,
            max_instructions,
            ourochronos::temporal::timeloop::DEFAULT_MAX_EPOCHS,
            semantics,
        );
        println!("{}", profile);
        if let Some(declaration) = &program.family_declaration {
            let contract = ourochronos::PspaceFamilyContract::from(declaration);
            println!("=== Declared PSPACE Family Contract: {} ===", contract.name);
            if contract.declared_eligible() {
                println!("All Aaronson--Watrous obligations are declared; finite-instance evidence and asymptotic assumptions remain distinct.");
            } else {
                println!("Missing declared obligations:");
                for obligation in contract.missing_obligations() {
                    println!("- {}", obligation);
                }
            }
            println!();
        }
    }

    if let Some(input_bits) = family_input_bits {
        let input = family_input.expect("family mode retains its parsed input");
        if program.markov_declaration.is_some() || program.quantum_declaration.is_some() {
            return fail_usage(
                "--verify-family checks an executable deterministic FAMILY body, not MARKOV or QCHANNEL declarations",
            );
        }
        let Some(declaration) = &program.family_declaration else {
            return fail_usage("--verify-family requires a source FAMILY declaration");
        };
        let contract = ourochronos::PspaceFamilyContract::from(declaration);
        let uniform_certificate = match ourochronos::PspaceUniformFamilyGenerator::certify_bytecode(
            &contract,
            &bytecode,
            family_width_rule.expect("family mode establishes a width rule"),
            state_bits,
            error_config.memory_bounds,
        ) {
            Ok(certificate) => certificate,
            Err(error) => {
                println!("UNSUPPORTED UNIFORM FAMILY GENERATOR: {error}");
                return EXIT_ERROR;
            }
        };
        let generated = match uniform_certificate.specialize(&input, state_limit, max_instructions)
        {
            Ok(instance) => instance,
            Err(error) => {
                println!("UNSUPPORTED FAMILY SPECIALIZATION: {error}");
                return EXIT_ERROR;
            }
        };
        debug_assert_eq!(generated.config.input_bits, input_bits);
        debug_assert_eq!(generated.config.graph.memory_cells, memory_cells);
        let family_theorem =
            ourochronos::ProjectionFamilyCertificate::prove(uniform_certificate.clone());
        let family_circuit = match &family_theorem {
            Ok(theorem) => match theorem.generate_circuit(input_bits) {
                Ok(circuit) => Some(circuit),
                Err(error) => {
                    eprintln!("Internal family-circuit generation error: {error}");
                    return EXIT_ERROR;
                }
            },
            Err(_) => None,
        };
        let result = generated.verify();
        if let Ok(theorem) = &family_theorem {
            let theorem_decision = match theorem.decision(&input) {
                Ok(decision) => decision,
                Err(error) => {
                    eprintln!("Internal family-theorem error: {error}");
                    return EXIT_ERROR;
                }
            };
            match &result {
                ourochronos::PspaceInstanceVerificationResult::Verified(certificate) => {
                    if certificate.decision != theorem_decision {
                        eprintln!(
                            "Internal family-theorem mismatch: symbolic decision {theorem_decision}, finite decision {}",
                            certificate.decision
                        );
                        return EXIT_ERROR;
                    }
                    let circuit = family_circuit
                        .as_ref()
                        .expect("projection theorem produces a circuit");
                    if let Err(error) = theorem.cross_check_finite(circuit, certificate) {
                        eprintln!("Internal family-circuit mismatch: {error}");
                        return EXIT_ERROR;
                    }
                }
                ourochronos::PspaceInstanceVerificationResult::Refuted { reason } => {
                    eprintln!(
                        "Internal family-theorem mismatch: all-input theorem conflicts with finite refutation: {reason}"
                    );
                    return EXIT_ERROR;
                }
                _ => {}
            }
        }
        if let Some(path) = &artifact_path {
            let json = match &family_theorem {
                Ok(theorem) => theorem.circuit_verification_json(
                    family_circuit
                        .as_ref()
                        .expect("projection theorem produces a circuit"),
                    &result,
                ),
                Err(_) => uniform_certificate.verification_json(&result),
            };
            if let Err(error) = write_analysis_file(path, &format!("{json}\n")) {
                eprintln!("Error: {error}");
                return EXIT_ERROR;
            }
            println!("Verification artifact: {path}");
        }
        return match result {
            ourochronos::PspaceInstanceVerificationResult::Verified(certificate) => {
                println!("=== VERIFIED FINITE FAMILY INSTANCE ===");
                println!(
                    "Family {} on input {} (length {}): {} states, {} recurrent class(es).",
                    certificate.family_name,
                    certificate
                        .input
                        .iter()
                        .map(|bit| char::from(b'0' + *bit))
                        .collect::<String>(),
                    certificate.input_bits,
                    certificate.state_count,
                    certificate.recurrent_class_count
                );
                println!(
                    "Decision {} is invariant; max transition {} steps; conservative chronology workspace {} bits.",
                    certificate.decision,
                    certificate.maximum_transition_steps,
                    certificate.chronology_respecting_bits
                );
                println!(
                    "Evidence digests: bytecode {:016x}, complete transition table {:016x}.",
                    certificate.bytecode_digest, certificate.transition_digest
                );
                println!(
                    "Structurally uniform generator verified: work <= {}*n^{}+{}, generated CTC width {} cells.",
                    uniform_certificate.generation_work_bound.coefficient,
                    uniform_certificate.generation_work_bound.degree,
                    uniform_certificate.generation_work_bound.additive,
                    generated.generation.temporal_cells,
                );
                match &family_theorem {
                    Ok(theorem) => {
                        let circuit = family_circuit
                            .as_ref()
                            .expect("projection theorem produces a circuit");
                        println!(
                            "Family-wide projection/routing theorem verified for every nonempty Boolean input: {} assignment(s), {:?}; chronology <= {}*n^{}+{}, transition <= {} steps.",
                            theorem.assignments.len(),
                            theorem.decision_rule,
                            theorem.chronology_workspace_bound.coefficient,
                            theorem.chronology_workspace_bound.degree,
                            theorem.chronology_workspace_bound.additive,
                            theorem.transition_steps,
                        );
                        println!(
                            "Explicit Boolean circuit verified: {} temporal-state wires plus one decision wire; every finite successor was differentially checked.",
                            circuit.state_bits()
                        );
                    }
                    Err(error) => println!(
                        "Family-wide semantic theorem unavailable ({error}); this exact input remains completely verified."
                    ),
                }
                println!("Nature's ideal Deutsch selector remains an external model assumption.");
                EXIT_OK
            }
            ourochronos::PspaceInstanceVerificationResult::Refuted { reason } => {
                println!("REFUTED FAMILY INSTANCE: {reason}");
                EXIT_PARADOX
            }
            ourochronos::PspaceInstanceVerificationResult::Unknown { reason } => {
                println!("UNKNOWN FAMILY INSTANCE: {reason}");
                EXIT_TIMEOUT
            }
            ourochronos::PspaceInstanceVerificationResult::Unsupported { reason } => {
                println!("UNSUPPORTED FAMILY INSTANCE: {reason}");
                EXIT_ERROR
            }
        };
    }

    if stationary_mode || program.markov_declaration.is_some() {
        if program.quantum_declaration.is_some() {
            return fail_usage("a program cannot select both MARKOV and QCHANNEL analysis");
        }
        if diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || fast_mode
            || halting_bound.is_some()
            || quantum_mode
        {
            return fail_usage("a MARKOV declaration selects stationary analysis and cannot be combined with an execution/search mode");
        }
        let Some(declaration) = &program.markov_declaration else {
            return fail_usage("--stationary requires a MARKOV declaration");
        };
        return match ourochronos::analyze_declaration(declaration) {
            Ok(analysis) => {
                println!("=== STOCHASTIC DEUTSCH ANALYSIS: {} ===", declaration.name);
                for (index, class) in analysis.family.recurrent_classes.iter().enumerate() {
                    let distribution = &analysis.family.extremal[index];
                    let nonzero: Vec<String> = distribution
                        .iter()
                        .enumerate()
                        .filter(|(_, probability)| !probability.is_zero())
                        .map(|(state, probability)| format!("{}:{}", state, probability))
                        .collect();
                    println!(
                        "Class {} {:?}: stationary {{{}}}; acceptance {}",
                        index,
                        class,
                        nonzero.join(", "),
                        analysis.acceptance_probabilities[index]
                    );
                }
                match analysis.decision {
                    ourochronos::StationaryDecision::Accept => {
                        println!("DECISION: ACCEPT on every stationary fixed point.");
                        EXIT_OK
                    }
                    ourochronos::StationaryDecision::Reject => {
                        println!("DECISION: REJECT on every stationary fixed point.");
                        EXIT_OK
                    }
                    ourochronos::StationaryDecision::Ambiguous => {
                        println!("DECISION: AMBIGUOUS; the all-fixed-points correctness promise is not satisfied.");
                        EXIT_PARADOX
                    }
                }
            }
            Err(message) => {
                println!("STOCHASTIC MODEL ERROR: {}", message);
                EXIT_ERROR
            }
        };
    }

    if quantum_mode || program.quantum_declaration.is_some() {
        if diagnostic
            || deutsch_mode
            || action_mode
            || smt
            || global_mode
            || all_fixed_mode
            || recurrent_mode
            || verify_mode
            || fast_mode
            || halting_bound.is_some()
            || stationary_mode
        {
            return fail_usage("a QCHANNEL declaration selects quantum fixed-space analysis and cannot be combined with an execution/search mode");
        }
        let Some(declaration) = &program.quantum_declaration else {
            return fail_usage("--quantum-fixed requires a QCHANNEL declaration");
        };
        return match ourochronos::analyze_quantum_declaration(declaration) {
            Ok(analysis) => {
                println!(
                    "=== QUBIT DEUTSCH FIXED-SPACE ANALYSIS: {} ===",
                    declaration.name
                );
                println!("Fixed affine dimension: {}", analysis.affine_dimension);
                println!(
                    "Numerically estimated acceptance range: [{:.12}, {:.12}]",
                    analysis.minimum_acceptance, analysis.maximum_acceptance
                );
                println!(
                    "Analysis tolerance: {}; rank and extrema are numerical estimates, not certified error bounds.",
                    analysis.analysis_tolerance
                );
                if analysis.numerical_uncertainty {
                    println!("DECISION: NUMERICALLY UNCERTAIN; the estimated range is too close to a decision threshold.");
                    return EXIT_TIMEOUT;
                }
                match analysis.decision {
                    ourochronos::StationaryDecision::Accept => {
                        println!(
                            "NUMERICAL DECISION: ACCEPT under the estimated fixed-space model."
                        );
                        EXIT_OK
                    }
                    ourochronos::StationaryDecision::Reject => {
                        println!(
                            "NUMERICAL DECISION: REJECT under the estimated fixed-space model."
                        );
                        EXIT_OK
                    }
                    ourochronos::StationaryDecision::Ambiguous => {
                        println!("NUMERICAL DECISION: AMBIGUOUS; the estimated range establishes neither threshold decision.");
                        EXIT_PARADOX
                    }
                }
            }
            Err(message) => {
                println!("QUANTUM CHANNEL ERROR: {}", message);
                EXIT_ERROR
            }
        };
    }

    if let Some(instruction_bound) = halting_bound {
        return match ourochronos::BoundedHaltingAnalyzer::analyze_bytecode(
            &bytecode,
            instruction_bound,
            memory_cells,
        ) {
            ourochronos::BoundedHaltingResult::Halted {
                instructions,
                output,
            } => {
                println!(
                    "HALTED after {} instruction(s) within bound {}.",
                    instructions, instruction_bound
                );
                if !output.is_empty() {
                    for item in output {
                        match item {
                            ourochronos::OutputItem::Val(value) => print!("[{}]", value.val),
                            ourochronos::OutputItem::Char(character) => {
                                print!("{}", character as char)
                            }
                        }
                    }
                    println!();
                }
                EXIT_OK
            }
            ourochronos::BoundedHaltingResult::NotHaltedWithinBound { instruction_bound } => {
                println!(
                    "UNKNOWN: program did not halt within {} instructions; this is not proof of nonhalting.",
                    instruction_bound
                );
                EXIT_TIMEOUT
            }
            ourochronos::BoundedHaltingResult::Unsupported { reason } => {
                println!("UNSUPPORTED HALTING ANALYSIS: {}", reason);
                EXIT_ERROR
            }
            ourochronos::BoundedHaltingResult::RuntimeError { message, .. } => {
                println!("HALTING ANALYSIS ERROR: {}", message);
                EXIT_ERROR
            }
        };
    }

    if recurrent_mode {
        println!("=== COMPLETE SMALL-STATE RECURRENT ANALYSIS ===");
        let config = ourochronos::ProgramGraphConfig {
            memory_cells,
            cell_bits: state_bits,
            max_states: state_limit,
            max_instructions,
            bounds_policy: error_config.memory_bounds,
        };
        return match ourochronos::BytecodeTransitionAnalyzer::analyze(&bytecode, config) {
            Ok(analysis) => {
                if let Some(path) = &dot_path {
                    if let Err(error) = write_analysis_file(path, &analysis.to_dot()) {
                        eprintln!("Error: {}", error);
                        return EXIT_ERROR;
                    }
                    println!("Graphviz artifact: {}", path);
                }
                println!(
                    "Classified all {} states in the closed {}-cell x {}-bit domain.",
                    analysis.graph.state_count(),
                    memory_cells,
                    state_bits
                );
                println!(
                    "Recurrent classes: {}; point fixed states: {}.",
                    analysis.recurrent.recurrent_classes.len(),
                    analysis.recurrent.fixed_states.len()
                );
                for (class_index, states) in analysis.recurrent.recurrent_classes.iter().enumerate()
                {
                    println!(
                        "Class {}: period {}, basin {}, output {}.",
                        class_index,
                        states.len(),
                        analysis.recurrent.basin_sizes[class_index],
                        if analysis.class_has_unanimous_output(class_index) {
                            "unanimous"
                        } else {
                            "ambiguous"
                        }
                    );
                    for state in states.iter().take(16) {
                        println!("  state {} = {:?}", state, analysis.state_memory(*state));
                    }
                    if states.len() > 16 {
                        println!("  ... {} further recurrent states", states.len() - 16);
                    }
                }
                EXIT_OK
            }
            Err(error) => {
                println!("RECURRENT ANALYSIS REFUSED: {}", error);
                EXIT_ERROR
            }
        };
    }

    let global_config = ourochronos::GlobalSolveConfig {
        memory_cells,
        loop_unroll_limit,
        solver_timeout_ms,
        max_instructions,
        bounds_policy: error_config.memory_bounds,
    };

    if verify_mode {
        if program.temporal_properties.is_empty() {
            return fail_usage("--verify requires at least one source PROPERTY declaration");
        }
        println!("=== ALL-FIXED TEMPORAL PROPERTY VERIFICATION ===");
        let mut exit = EXIT_OK;
        let mut artifact_results = Vec::new();
        for property in &program.temporal_properties {
            let result = ourochronos::GlobalFixedPointSolver::verify_property_bytecode(
                &bytecode,
                property,
                global_config,
            );
            artifact_results.push(result.to_json());
            match result {
                ourochronos::PropertyVerificationResult::Proven {
                    property,
                    exemplar,
                    certificate,
                } => {
                    println!(
                        "PROVEN {}: every point fixed state has cell {} {:?} {}.",
                        property.name, property.address, property.comparison, property.value
                    );
                    println!(
                        "  exemplar replay digest {:016x}; proof-query digest {:016x}",
                        exemplar.constraint_digest, certificate.constraint_digest
                    );
                    print_unsat_evidence(&certificate);
                }
                ourochronos::PropertyVerificationResult::Refuted {
                    property,
                    counterexample,
                } => {
                    println!(
                        "REFUTED {}: replayed fixed-state counterexample has cell {} = {}.",
                        property.name,
                        property.address,
                        counterexample.memory.read(property.address).val
                    );
                    println!("  counterexample: {:?}", counterexample.memory);
                    if exit != EXIT_ERROR {
                        exit = EXIT_PARADOX;
                    }
                }
                ourochronos::PropertyVerificationResult::Vacuous {
                    property,
                    no_fixed_point,
                } => {
                    println!(
                        "VACUOUS {}: the program has no point fixed state (digest {:016x}).",
                        property.name, no_fixed_point.constraint_digest
                    );
                    print_unsat_evidence(&no_fixed_point);
                    if exit != EXIT_ERROR {
                        exit = EXIT_PARADOX;
                    }
                }
                ourochronos::PropertyVerificationResult::Unknown {
                    property,
                    reason,
                    completeness,
                    constraint_digest,
                    ..
                } => {
                    println!("UNKNOWN {}: {}", property.name, reason);
                    println!("  IR completeness: {:?}", completeness);
                    if let Some(digest) = constraint_digest {
                        println!("  constraint digest: {:016x}", digest);
                    }
                    if exit == EXIT_OK {
                        exit = EXIT_TIMEOUT;
                    }
                }
                ourochronos::PropertyVerificationResult::Unsupported { property, reason } => {
                    println!("UNSUPPORTED {}: {}", property.name, reason);
                    exit = EXIT_ERROR;
                }
                ourochronos::PropertyVerificationResult::InternalError { property, reason } => {
                    println!(
                        "PROPERTY SOLVER INTERNAL ERROR {}: {}",
                        property.name, reason
                    );
                    exit = EXIT_ERROR;
                }
            }
        }
        if let Some(path) = &artifact_path {
            let artifact = format!(
                "{{\"schema\":\"ourochronos.verification-batch/v1\",\"results\":[{}]}}\n",
                artifact_results.join(",")
            );
            if let Err(error) = write_analysis_file(path, &artifact) {
                eprintln!("Error: {}", error);
                return EXIT_ERROR;
            }
            println!("Verification artifact: {}", path);
        }
        return exit;
    }

    if global_mode {
        println!("=== GLOBAL POINT FIXED-STATE ANALYSIS ===");
        let result = ourochronos::GlobalFixedPointSolver::solve_bytecode(&bytecode, global_config);
        if let Some(path) = &artifact_path {
            if let Err(error) = write_analysis_file(path, &format!("{}\n", result.to_json())) {
                eprintln!("Error: {}", error);
                return EXIT_ERROR;
            }
            println!("Verification artifact: {}", path);
        }
        return match result {
            ourochronos::GlobalSolveResult::Found(witness) => {
                println!("FOUND: a replay-verified solution to F(s)=s.");
                if witness.memory.is_empty() {
                    println!(
                        "Witness: all {} temporal cells are zero.",
                        witness.memory.len()
                    );
                } else {
                    let cells = witness
                        .memory
                        .iter_nonzero()
                        .map(|(address, value)| format!("[{}]={}", address, value.val))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("Witness: {}", cells);
                }
                println!(
                    "Replay: verified in {} instruction(s); IR completeness: {:?}; constraint digest: {:016x}",
                    witness.instructions_executed,
                    witness.completeness,
                    witness.constraint_digest
                );
                if !witness.output.is_empty() {
                    for item in witness.output {
                        match item {
                            ourochronos::OutputItem::Val(value) => print!("[{}]", value.val),
                            ourochronos::OutputItem::Char(character) => {
                                print!("{}", character as char)
                            }
                        }
                    }
                    println!();
                }
                EXIT_OK
            }
            ourochronos::GlobalSolveResult::ProvenNoFixedPoint(certificate) => {
                println!(
                    "PROVEN NO POINT FIXED STATE: {} returned UNSAT for a complete finite IR.",
                    certificate.backend
                );
                println!("Constraint digest: {:016x}", certificate.constraint_digest);
                print_unsat_evidence(&certificate);
                EXIT_PARADOX
            }
            ourochronos::GlobalSolveResult::Unknown {
                reason,
                completeness,
                constraint_digest,
            } => {
                println!("UNKNOWN: {}", reason);
                println!("IR completeness: {:?}", completeness);
                if let Some(digest) = constraint_digest {
                    println!("Constraint digest: {:016x}", digest);
                }
                EXIT_TIMEOUT
            }
            ourochronos::GlobalSolveResult::Unsupported { reason } => {
                println!("UNSUPPORTED GLOBAL ANALYSIS: {}", reason);
                EXIT_ERROR
            }
            ourochronos::GlobalSolveResult::InternalError { reason } => {
                println!("GLOBAL SOLVER INTERNAL ERROR: {}", reason);
                EXIT_ERROR
            }
        };
    }

    if all_fixed_mode {
        println!("=== ALL POINT FIXED-STATES ANALYSIS ===");
        let result = ourochronos::GlobalFixedPointSolver::analyze_uniqueness_bytecode(
            &bytecode,
            global_config,
        );
        if let Some(path) = &artifact_path {
            if let Err(error) = write_analysis_file(path, &format!("{}\n", result.to_json())) {
                eprintln!("Error: {}", error);
                return EXIT_ERROR;
            }
            println!("Verification artifact: {}", path);
        }
        return match result {
            ourochronos::GlobalUniquenessResult::NoFixedPoint(certificate) => {
                println!("PROVEN: the complete finite IR has no point fixed state.");
                println!("Constraint digest: {:016x}", certificate.constraint_digest);
                print_unsat_evidence(&certificate);
                EXIT_PARADOX
            }
            ourochronos::GlobalUniquenessResult::Unique {
                witness,
                certificate,
            } => {
                println!("PROVEN UNIQUE: exactly one point fixed state exists.");
                if witness.memory.is_empty() {
                    println!(
                        "Witness: all {} temporal cells are zero.",
                        witness.memory.len()
                    );
                } else {
                    let cells = witness
                        .memory
                        .iter_nonzero()
                        .map(|(address, value)| format!("[{}]={}", address, value.val))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("Witness: {}", cells);
                }
                println!(
                    "Witness replay verified; uniqueness-query digest: {:016x}",
                    certificate.constraint_digest
                );
                print_unsat_evidence(&certificate);
                EXIT_OK
            }
            ourochronos::GlobalUniquenessResult::Multiple {
                first,
                second,
                differing_cells,
            } => {
                println!("AMBIGUOUS: multiple point fixed states exist.");
                println!("First witness: {:?}", first.memory);
                println!("Second witness: {:?}", second.memory);
                println!("Differing temporal cells: {:?}", differing_cells);
                println!("Both witnesses passed independent reference-VM replay.");
                EXIT_PARADOX
            }
            ourochronos::GlobalUniquenessResult::Unknown {
                reason,
                witness,
                completeness,
                constraint_digest,
            } => {
                println!("UNKNOWN: {}", reason);
                println!("IR completeness: {:?}", completeness);
                if witness.is_some() {
                    println!("At least one replay-verified point fixed state exists.");
                }
                if let Some(digest) = constraint_digest {
                    println!("Constraint digest: {:016x}", digest);
                }
                EXIT_TIMEOUT
            }
            ourochronos::GlobalUniquenessResult::Unsupported { reason } => {
                println!("UNSUPPORTED ALL-FIXED ANALYSIS: {}", reason);
                EXIT_ERROR
            }
            ourochronos::GlobalUniquenessResult::InternalError { reason } => {
                println!("ALL-FIXED SOLVER INTERNAL ERROR: {}", reason);
                EXIT_ERROR
            }
        };
    }

    if smt {
        let Some(ir) = native_temporal_ir else {
            eprintln!("Typed temporal IR error: native bytecode lowering was not produced");
            return EXIT_ERROR;
        };
        println!("; Typed temporal IR for {}", filename);
        println!("{}", ir.to_smt2(true));
        return EXIT_OK;
    }

    if bytecode_runtime_mode {
        ourochronos::core::provenance::set_saturation_limit(provenance_limit);
        ourochronos::core::provenance::set_address_space_size(memory_cells);
        let file_snapshots = match freeze_file_capabilities(&file_read_paths, &file_write_paths) {
            Ok(snapshots) => snapshots,
            Err(error) => {
                eprintln!("File capability error: {error}");
                return EXIT_ERROR;
            }
        };
        let endpoint_tapes =
            match freeze_endpoint_tapes(&network_input_specs, &network_send_endpoints) {
                Ok(tapes) => tapes,
                Err(error) => {
                    eprintln!("Network capability error: {error}");
                    return EXIT_ERROR;
                }
            };
        let process_results = match freeze_process_results(&process_descriptor_paths) {
            Ok(results) => results,
            Err(error) => {
                eprintln!("Process capability error: {error}");
                return EXIT_ERROR;
            }
        };
        let process_capabilities = process_results
            .iter()
            .map(|result| result.command.clone())
            .collect();
        let has_commit_capabilities = !file_write_paths.is_empty()
            || !network_send_endpoints.is_empty()
            || !process_results.is_empty()
            || max_sleep_milliseconds.is_some();
        let vm = BytecodeVmConfig {
            max_instructions,
            memory_bounds: error_config.memory_bounds,
            allow_interactive_input: true,
            allow_live_system_inputs: matches!(
                effects_policy,
                ourochronos::vm::EffectsPolicy::Unrestricted
            ),
            file_snapshots,
            file_read_capabilities: file_read_paths.clone(),
            file_write_capabilities: file_write_paths.clone(),
            endpoint_tapes,
            network_send_capabilities: network_send_endpoints.clone(),
            process_results: process_results.clone(),
            process_capabilities,
            max_sleep_milliseconds,
            ..BytecodeVmConfig::default()
        };
        let runtime = BytecodeTimeLoopConfig {
            max_epochs: ourochronos::temporal::timeloop::DEFAULT_MAX_EPOCHS,
            memory_cells,
            seed,
            initial_state: Vec::new(),
            vm,
            diagnostic_sources: linked_metadata.source_files,
        };
        if action_mode {
            println!(
                "Running in ACTION-GUIDED mode (exploring {} seeds).",
                num_seeds
            );
            let temporal_core = program.temporal_op_count();
            let action = if temporal_core > 0 {
                ActionConfig::derive_from_analysis(temporal_core, memory_cells, 2.0)
            } else {
                ActionConfig::anti_trivial()
            };
            let runner =
                match ourochronos::BytecodeActionRunner::new(ourochronos::BytecodeActionConfig {
                    runtime,
                    action,
                    num_seeds,
                }) {
                    Ok(runner) => runner,
                    Err(error) => {
                        eprintln!("Bytecode action configuration error: {error}");
                        return EXIT_ERROR;
                    }
                };
            let report = runner.run_report(&bytecode);
            audit::audit(
                AuditEntry::new(
                    "BYTECODE_ACTION_OUTCOME",
                    "Program",
                    filename,
                    "Bytecode action-guided fixed-state selection finished",
                )
                .with_category(ActionCategory::Execute)
                .with_meta("attempted_seeds", report.stats.attempted_seeds.to_string())
                .with_meta(
                    "consistent_candidates",
                    report.stats.consistent_candidates.to_string(),
                )
                .with_meta("cache_hits", report.stats.cache_hits.to_string())
                .with_meta("cache_misses", report.stats.cache_misses.to_string()),
            );
            let mut outcome = report.status;
            if let Some(batch) = report.selected_batch {
                if !batch.effects.is_empty() {
                    let mut adapter = native_effect_adapter(
                        &file_write_paths,
                        &network_send_endpoints,
                        &process_results,
                        max_sleep_milliseconds,
                    );
                    if let Err(message) = ourochronos::EffectCommitAdapter::apply_selected(
                        &mut adapter,
                        &batch.receipt,
                        &batch.effects,
                    ) {
                        outcome = ConvergenceStatus::Error { message, epoch: 0 };
                    }
                }
            }
            return finish_bytecode_orbit(outcome, true);
        }

        if fast_mode {
            println!(
                "Using prepared bytecode dispatch (the optimized and reference paths share one VM)."
            );
        }
        let driver = match BytecodeTimeLoop::new(runtime) {
            Ok(driver) => driver,
            Err(error) => {
                eprintln!("Bytecode runtime configuration error: {error}");
                return EXIT_ERROR;
            }
        };
        let outcome = if deutsch_mode {
            println!("Running in DEUTSCH mode (cycles are stationary ensembles).");
            if !has_commit_capabilities {
                driver.run_deutsch(&bytecode)
            } else {
                let mut log = ourochronos::CommitLog::default();
                let mut adapter = native_effect_adapter(
                    &file_write_paths,
                    &network_send_endpoints,
                    &process_results,
                    max_sleep_milliseconds,
                );
                driver.run_deutsch_with_adapter(
                    &bytecode,
                    ourochronos::CommitToken(1),
                    &mut log,
                    &mut adapter,
                )
            }
        } else if diagnostic {
            println!("Running in DIAGNOSTIC mode.");
            if !has_commit_capabilities {
                driver.run_diagnostic(&bytecode)
            } else {
                let mut log = ourochronos::CommitLog::default();
                let mut adapter = native_effect_adapter(
                    &file_write_paths,
                    &network_send_endpoints,
                    &process_results,
                    max_sleep_milliseconds,
                );
                driver.run_diagnostic_with_adapter(
                    &bytecode,
                    ourochronos::CommitToken(1),
                    &mut log,
                    &mut adapter,
                )
            }
        } else if has_commit_capabilities {
            let mut log = ourochronos::CommitLog::default();
            let mut adapter = native_effect_adapter(
                &file_write_paths,
                &network_send_endpoints,
                &process_results,
                max_sleep_milliseconds,
            );
            driver.run_with_adapter(
                &bytecode,
                ourochronos::CommitToken(1),
                &mut log,
                &mut adapter,
            )
        } else {
            driver.run(&bytecode)
        };
        audit::audit(
            AuditEntry::new(
                "BYTECODE_OUTCOME",
                "Program",
                filename,
                "Iterative bytecode fixed-point execution finished",
            )
            .with_category(ActionCategory::Execute),
        );
        return finish_bytecode_orbit(outcome, diagnostic);
    }
    unreachable!("every non-build execution/search mode returns through a linked bytecode policy")
}
