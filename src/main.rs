use ourochronos::{TimeLoop, ConvergenceStatus, Config, ExecutionMode, tokenize, Parser, ActionConfig, type_check, types, ErrorConfig};
use ourochronos::vm::fast_vm::{is_program_pure, FastExecutor};
use ourochronos::vm::EpochStatus;
use ourochronos::audit::{self, AuditEntry, AuditConfig, AuditFormat, ActionCategory, Severity};
use std::env;
use std::fs;
use std::time::Instant;

/// Process exit codes. A consistent timeline (or successful pure run) exits 0;
/// usage, parse, and runtime errors exit 1; a detected paradox (oscillation,
/// divergence, or explicit PARADOX) exits 2; epoch exhaustion exits 3, so
/// scripts can distinguish "no fixed point exists" from "gave up looking".
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
    FlagSpec { name: "--diagnostic", alias: None, arity: Arity::None, metavar: "",
        help: &["Enable diagnostic mode (full trajectory recording)"] },
    FlagSpec { name: "--action", alias: None, arity: Arity::None, metavar: "",
        help: &["Enable action-guided mode (solves the Genie Effect)"] },
    FlagSpec { name: "--typecheck", alias: None, arity: Arity::None, metavar: "",
        help: &["Run static type analysis (temporal tainting)"] },
    FlagSpec { name: "--smt", alias: None, arity: Arity::None, metavar: "",
        help: &["Generate SMT-LIB2 output instead of running"] },
    FlagSpec { name: "--seed", alias: None, arity: Arity::Value, metavar: "<n>",
        help: &["Set initial seed value"] },
    FlagSpec { name: "--seeds", alias: None, arity: Arity::Value, metavar: "<n>",
        help: &["Number of seeds to try in action mode (default: 4)"] },
    FlagSpec { name: "--max-inst", alias: None, arity: Arity::Value, metavar: "<n>",
        help: &["Maximum instructions per epoch (default: 10000000)"] },
    FlagSpec { name: "--fast", alias: None, arity: Arity::None, metavar: "",
        help: &["Use fast VM for pure (non-temporal) programmes"] },
    FlagSpec { name: "--lsp", alias: None, arity: Arity::None, metavar: "",
        help: &["Start Language Server Protocol server"] },
    FlagSpec { name: "--strict", alias: None, arity: Arity::None, metavar: "",
        help: &["Strict error handling (bounds violations produce errors)"] },
    FlagSpec { name: "--permissive", alias: None, arity: Arity::None, metavar: "",
        help: &["Permissive error handling (wrap addresses, zero on underflow)"] },
    FlagSpec { name: "--audit", alias: None, arity: Arity::OptionalValue, metavar: "[file]",
        help: &["Enable audit logging (default: ourochronos-audit.log)"] },
    FlagSpec { name: "--audit-json", alias: None, arity: Arity::None, metavar: "",
        help: &["Use JSON Lines format for audit output"] },
    FlagSpec { name: "--provenance-limit", alias: None, arity: Arity::Value, metavar: "<n>",
        help: &["Provenance saturation limit (default: 256)"] },
    FlagSpec { name: "--effects", alias: None, arity: Arity::Value, metavar: "<policy>",
        help: &[
            "Effects inside the fixed-point search:",
            "'decline' (default) errors on external effects and",
            "non-determinism; 'unrestricted' permits them on",
            "every search epoch",
        ] },
    FlagSpec { name: "--help", alias: Some("-h"), arity: Arity::None, metavar: "",
        help: &["Show this help"] },
    FlagSpec { name: "--version", alias: Some("-V"), arity: Arity::None, metavar: "",
        help: &["Show version"] },
];

fn find_flag(arg: &str) -> Option<&'static FlagSpec> {
    FLAGS.iter().find(|f| f.name == arg || f.alias == Some(arg))
}

fn print_usage() {
    println!("Usage: ourochronos <file.ouro> [options]");
    println!("       ourochronos repl");
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
    println!("Exit codes: 0 consistent, 1 error, 2 paradox, 3 epoch exhaustion");
}

/// Print a usage error with the help pointer and return the error exit code.
fn fail_usage(msg: &str) -> i32 {
    eprintln!("Error: {}", msg);
    eprintln!("Run 'ourochronos --help' for usage.");
    EXIT_ERROR
}

/// Parse the value following a flag, failing loudly on garbage. A silently
/// substituted default would mask a typo in a resource limit.
fn parse_flag_value<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T, String> {
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

/// The four numeric limits, each defaulted when absent.
fn parse_numeric_flags(args: &[String]) -> Result<(u64, usize, u64, usize), String> {
    Ok((
        parse_flag_value(args, "--seed", 0)?,
        parse_flag_value(args, "--seeds", 4)?,
        parse_flag_value(args, "--max-inst", 10_000_000)?,
        parse_flag_value(
            args,
            "--provenance-limit",
            ourochronos::DEFAULT_PROVENANCE_SATURATION_LIMIT,
        )?,
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
        if seen.contains(&flag.name) {
            return Err(format!("{} specified more than once", flag.name));
        }
        seen.push(flag.name);
        i += match flag.arity {
            Arity::None => 1,
            Arity::Value => {
                if args.get(i + 1).is_none() {
                    return Err(format!("{} requires a value", flag.name));
                }
                2
            }
            Arity::OptionalValue => {
                if args.get(i + 1).is_some_and(|a| !a.starts_with('-')) { 2 } else { 1 }
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

    if args.len() < 2 {
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

    let filename = &args[1];
    if filename.starts_with('-') {
        return fail_usage(&format!("expected a programme file, got option '{}'", filename));
    }
    if let Err(msg) = validate_args(&args) {
        return fail_usage(&msg);
    }

    let diagnostic = args.contains(&"--diagnostic".to_string());
    let action_mode = args.contains(&"--action".to_string());
    let smt = args.contains(&"--smt".to_string());
    let typecheck_mode = args.contains(&"--typecheck".to_string());
    let fast_mode = args.contains(&"--fast".to_string());
    let strict_mode = args.contains(&"--strict".to_string());
    let permissive_mode = args.contains(&"--permissive".to_string());

    let error_config = if strict_mode {
        ErrorConfig::strict()
    } else if permissive_mode {
        ErrorConfig::permissive()
    } else {
        ErrorConfig::default()
    };

    let (seed, num_seeds, max_instructions, provenance_limit) =
        match parse_numeric_flags(&args) {
            Ok(values) => values,
            Err(msg) => return fail_usage(&msg),
        };

    if max_instructions == 0 {
        return fail_usage("--max-inst must be greater than 0");
    }
    if provenance_limit == 0 {
        return fail_usage("--provenance-limit must be greater than 0");
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
            format: if audit_json { AuditFormat::JsonLines } else { AuditFormat::Text },
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
            .with_meta("mode", if action_mode { "action" } else if fast_mode { "fast" } else { "standard" })
    );

    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: cannot read '{}': {}", filename, e);
            return EXIT_ERROR;
        }
    };

    let parse_start = Instant::now();
    let tokens = tokenize(&source);
    let mut parser = Parser::new(&tokens);
    parser.register_procedures(ourochronos::StdLib::procedures());

    let parsed_program = match parser.parse_program() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse Error: {}", e);
            return EXIT_ERROR;
        }
    };

    let parse_duration = parse_start.elapsed();
    audit::audit(
        AuditEntry::new("PARSE", "Program", filename, "Parsed successfully")
            .with_category(ActionCategory::Parse)
            .with_duration_us(parse_duration.as_micros() as u64)
            .with_meta("statements", parsed_program.body.len().to_string())
            .with_meta("procedures", parsed_program.procedures.len().to_string())
    );

    // Inline all procedure calls
    let program = if !parsed_program.procedures.is_empty() {
        if diagnostic {
            println!("Inlining {} procedure(s)...", parsed_program.procedures.len());
        }
        parsed_program.inline_procedures()
    } else {
        parsed_program
    };

    if typecheck_mode {
        println!("=== Temporal Type Analysis ===");
        let result = type_check(&program);
        println!("{}", types::display_types(&result));
        if !result.is_valid {
            eprintln!("Type errors found. Stopping.");
            return EXIT_ERROR;
        }
        println!(); // blank line before execution
    }

    if smt {
        let mut encoder = ourochronos::SmtEncoder::new();
        return match encoder.encode(&program) {
            Ok(smt_code) => {
                println!(";; Generated SMT-LIB2 for {}", filename);
                println!("{}", smt_code);
                EXIT_OK
            }
            Err(e) => {
                eprintln!("SMT encoding error: {}", e);
                EXIT_ERROR
            }
        };
    }

    // Fast VM mode for pure programmes
    if fast_mode {
        if is_program_pure(&program) {
            if diagnostic {
                println!("Using FAST VM (pure programme detected)");
            }
            let mut fast_exec = FastExecutor::new(max_instructions);
            return match fast_exec.execute_pure(&program, &program.quotes) {
                Ok(()) => {
                    if fast_exec.status == EpochStatus::Running {
                        fast_exec.status = EpochStatus::Finished;
                    }
                    match &fast_exec.status {
                        EpochStatus::Finished => {
                            if !fast_exec.output.is_empty() {
                                for val in &fast_exec.output {
                                    match val {
                                        ourochronos::OutputItem::Val(v) => print!("[{}]", v.val),
                                        ourochronos::OutputItem::Char(c) => print!("{}", *c as char),
                                    }
                                }
                                println!();
                            }
                            EXIT_OK
                        }
                        EpochStatus::Error(msg) => {
                            eprintln!("ERROR: {}", msg);
                            EXIT_ERROR
                        }
                        _ => EXIT_ERROR,
                    }
                }
                Err(e) => {
                    eprintln!("Fast VM Error: {}", e);
                    EXIT_ERROR
                }
            };
        } else if diagnostic {
            println!("Programme contains temporal operations, falling back to standard VM");
        }
    }

    // Determine execution mode
    let mode = if action_mode {
        println!("Running in ACTION-GUIDED mode (exploring {} seeds).", num_seeds);
        // Weights derived from the programme's own temporal footprint; the
        // hand-tuned anti_trivial constants remain the zero-knowledge
        // fallback for programmes with no measurable temporal core.
        let temporal_core = program.temporal_op_count();
        let action_config = if temporal_core > 0 {
            ActionConfig::derive_from_analysis(temporal_core, ourochronos::MEMORY_SIZE, 2.0)
        } else {
            ActionConfig::anti_trivial()
        };
        ExecutionMode::ActionGuided {
            config: action_config,
            num_seeds,
        }
    } else if diagnostic {
        println!("Running in DIAGNOSTIC mode.");
        ExecutionMode::Diagnostic
    } else {
        ExecutionMode::Standard
    };

    let config = Config {
        mode,
        seed,
        verbose: diagnostic || action_mode,
        max_instructions,
        error_config: error_config.clone(),
        provenance_limit,
        effects: effects_policy,
        ..Config::default()
    };

    let mut driver = match TimeLoop::new(config.clone()) {
        Ok(driver) => driver,
        Err(e) => {
            eprintln!("Error: {}", e);
            return EXIT_ERROR;
        }
    };
    let outcome = driver.run(&program);

    // Record the run's outcome so --audit captures how the timeline resolved,
    // not merely that a session started.
    let (outcome_name, outcome_detail) = match &outcome {
        ConvergenceStatus::Consistent { epochs, .. } => ("consistent", format!("{} epochs", epochs)),
        ConvergenceStatus::Paradox { epoch, .. } => ("paradox", format!("epoch {}", epoch)),
        ConvergenceStatus::Oscillation { period, .. } => ("oscillation", format!("period {}", period)),
        ConvergenceStatus::Timeout { max_epochs } => ("timeout", format!("{} epochs", max_epochs)),
        ConvergenceStatus::Divergence { .. } => ("divergence", String::new()),
        ConvergenceStatus::Error { message, .. } => ("error", message.clone()),
    };
    audit::audit(
        AuditEntry::new("OUTCOME", "Program", filename, "Fixed-point search finished")
            .with_category(ActionCategory::Execute)
            .with_meta("status", outcome_name)
            .with_meta("detail", outcome_detail)
    );

    match outcome {
        ConvergenceStatus::Consistent { epochs, output, .. } => {
            // Only print consistency status if verbose/diagnostic or NO output
            if config.verbose || output.is_empty() {
                println!("CONSISTENT after {} epochs.", epochs);
            }

            if !output.is_empty() {
                for val in output {
                    match val {
                        ourochronos::OutputItem::Val(v) => print!("[{}]", v.val),
                        ourochronos::OutputItem::Char(c) => print!("{}", c as char),
                    }
                }
                println!();
            }
            EXIT_OK
        },
        ConvergenceStatus::Paradox { message, .. } => {
            println!("PARADOX: {}", message);
            EXIT_PARADOX
        },
        ConvergenceStatus::Oscillation { period, oscillating_cells, diagnosis } => {
            println!("OSCILLATION detected (period {})", period);
            if diagnostic && !oscillating_cells.is_empty() {
                println!("Oscillating Addresses: {:?}", oscillating_cells);
            } else if diagnostic {
                println!("No specific single-cell oscillations detected (global state cycle).");
            }

            if diagnostic {
                match diagnosis {
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::NegativeLoop { explanation, .. } => {
                        println!("\nDIAGNOSIS (Grandfather Paradox):");
                        println!("{}", explanation);
                    },
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::Oscillation { cycle } => {
                        println!("\nCycle states:");
                        for (i, state) in cycle.iter().enumerate() {
                            // Only print non-empty states for brevity
                            let non_zeros: Vec<_> = state.iter().filter(|(_,v)| *v != 0).collect();
                            if !non_zeros.is_empty() {
                                println!("  State {}: {:?}", i, non_zeros);
                            }
                        }
                    },
                    ourochronos::temporal::timeloop::ParadoxDiagnosis::Unknown => {
                        println!("\nDIAGNOSIS: Unknown cause");
                    }
                }
            }
            EXIT_PARADOX
        },
        ConvergenceStatus::Timeout { max_epochs } => {
            println!("TIMEOUT after {} epochs.", max_epochs);
            EXIT_TIMEOUT
        },
        ConvergenceStatus::Divergence { .. } => {
            println!("DIVERGENCE detected.");
            EXIT_PARADOX
        },
        ConvergenceStatus::Error { message, .. } => {
            println!("ERROR: {}", message);
            EXIT_ERROR
        }
    }
}
