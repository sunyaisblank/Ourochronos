use ourochronos::{TimeLoop, ConvergenceStatus, Config, ExecutionMode, tokenize, Parser, ActionConfig, type_check, types};
use ourochronos::fast_vm::{is_program_pure, FastExecutor};
use ourochronos::tracing_jit::JitFastExecutor;
use ourochronos::vm::EpochStatus;
use ourochronos::audit::{self, AuditEntry, AuditConfig, AuditFormat, ActionCategory, Severity, Outcome};
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for LSP mode first (no file required)
    #[cfg(feature = "lsp")]
    if args.contains(&"--lsp".to_string()) {
        eprintln!("Starting OUROCHRONOS Language Server...");
        if let Err(e) = ourochronos::lsp::server::run_server() {
            eprintln!("LSP Server error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    #[cfg(not(feature = "lsp"))]
    if args.contains(&"--lsp".to_string()) {
        eprintln!("LSP support not enabled. Rebuild with: cargo build --features lsp");
        std::process::exit(1);
    }

    if args.len() < 2 {
        println!("Usage: ourochronos <file.ouro> [options]");
        println!();
        println!("Options:");
        println!("  --diagnostic    Enable diagnostic mode (full trajectory recording)");
        println!("  --action        Enable action-guided mode (solves the Genie Effect)");
        println!("  --typecheck     Run static type analysis (temporal tainting)");
        println!("  --smt           Generate SMT-LIB2 output instead of running");
        println!("  --seed <n>      Set initial seed value");
        println!("  --seeds <n>     Number of seeds to try in action mode (default: 4)");
        println!("  --max-inst <n>  Maximum instructions per epoch (default: 10000000)");
        println!("  --fast          Use fast VM for pure (non-temporal) programmes");
        println!("  --jit           Use JIT-enabled fast VM with hot loop compilation");
        println!("  --lsp           Start Language Server Protocol server");
        println!("  --audit [file]  Enable audit logging (default: ourochronos-audit.log)");
        println!("  --audit-json    Use JSON Lines format for audit output");
        return;
    }

    let filename = &args[1];
    let diagnostic = args.contains(&"--diagnostic".to_string());
    let action_mode = args.contains(&"--action".to_string());
    let smt = args.contains(&"--smt".to_string());
    let typecheck_mode = args.contains(&"--typecheck".to_string());
    let fast_mode = args.contains(&"--fast".to_string());
    let jit_mode = args.contains(&"--jit".to_string());
    
    // Parse seed: --seed <u64>
    let mut seed = 0;
    if let Some(idx) = args.iter().position(|a| a == "--seed") {
        if idx + 1 < args.len() {
             seed = args[idx+1].parse().unwrap_or(0);
        }
    }
    
    // Parse num_seeds: --seeds <usize>
    let mut num_seeds = 4;
    if let Some(idx) = args.iter().position(|a| a == "--seeds") {
        if idx + 1 < args.len() {
             num_seeds = args[idx+1].parse().unwrap_or(4);
        }
    }

    // Parse max_instructions: --max-inst <u64>
    let mut max_instructions: u64 = 10_000_000;
    if let Some(idx) = args.iter().position(|a| a == "--max-inst") {
        if idx + 1 < args.len() {
             max_instructions = args[idx+1].parse().unwrap_or(10_000_000);
        }
    }

    // Parse audit options
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

    // Log startup
    audit::audit(
        AuditEntry::new("STARTUP", "System", "ourochronos", "CLI session started")
            .with_category(ActionCategory::System)
            .with_meta("file", filename)
            .with_meta("mode", if action_mode { "action" } else if fast_mode { "fast" } else if jit_mode { "jit" } else { "standard" })
    );

    let source = fs::read_to_string(filename).expect("Failed to read file");
    
    let parse_start = Instant::now();
    let tokens = tokenize(&source);
    let mut parser = Parser::new(&tokens);
    parser.register_procedures(ourochronos::StdLib::procedures());

    match parser.parse_program() {
        Ok(parsed_program) => {
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
            
            // Type check if requested
            if typecheck_mode {
                println!("=== Temporal Type Analysis ===");
                let result = type_check(&program);
                println!("{}", types::display_types(&result));
                if !result.is_valid {
                    eprintln!("Type errors found. Stopping.");
                    return;
                }
                println!(); // blank line before execution
            }
            
            if smt {
                let mut encoder = ourochronos::SmtEncoder::new();
                let smt_code = encoder.encode(&program);
                println!(";; Generated SMT-LIB2 for {}", filename);
                println!("{}", smt_code);
                return;
            }

            // Fast VM mode for pure programmes
            if fast_mode {
                if is_program_pure(&program) {
                    if diagnostic {
                        println!("Using FAST VM (pure programme detected)");
                    }
                    let mut fast_exec = FastExecutor::new(max_instructions);
                    match fast_exec.execute_pure(&program, &program.quotes) {
                        Ok(()) => {
                            if fast_exec.status == EpochStatus::Running {
                                fast_exec.status = EpochStatus::Finished;
                            }
                            if fast_exec.status == EpochStatus::Finished {
                                if !fast_exec.output.is_empty() {
                                    for val in &fast_exec.output {
                                        match val {
                                            ourochronos::OutputItem::Val(v) => print!("[{}]", v.val),
                                            ourochronos::OutputItem::Char(c) => print!("{}", *c as char),
                                        }
                                    }
                                    println!();
                                }
                            } else if let EpochStatus::Error(msg) = &fast_exec.status {
                                eprintln!("ERROR: {}", msg);
                            }
                        }
                        Err(e) => {
                            eprintln!("Fast VM Error: {}", e);
                        }
                    }
                    return;
                } else {
                    if diagnostic {
                        println!("Programme contains temporal operations, falling back to standard VM");
                    }
                }
            }

            // JIT-enabled fast VM mode for pure programmes
            if jit_mode {
                if is_program_pure(&program) {
                    if diagnostic {
                        println!("Using JIT-enabled fast VM (pure programme detected)");
                    }
                    let mut jit_exec = JitFastExecutor::new(max_instructions);
                    match jit_exec.execute_pure(&program, &program.quotes) {
                        Ok(()) => {
                            if jit_exec.status == EpochStatus::Running {
                                jit_exec.status = EpochStatus::Finished;
                            }
                            if jit_exec.status == EpochStatus::Finished {
                                if !jit_exec.output.is_empty() {
                                    for val in &jit_exec.output {
                                        match val {
                                            ourochronos::OutputItem::Val(v) => print!("[{}]", v.val),
                                            ourochronos::OutputItem::Char(c) => print!("{}", *c as char),
                                        }
                                    }
                                    println!();
                                }
                                // Print JIT stats if diagnostic
                                if diagnostic {
                                    let stats = jit_exec.jit_stats();
                                    println!("JIT Stats: {} traces recorded, {} compiled, {} compiled executions",
                                        stats.traces_recorded, stats.traces_compiled, stats.compiled_executions);
                                    println!("Instructions saved: {}", stats.instructions_saved);
                                }
                            } else if let EpochStatus::Error(msg) = &jit_exec.status {
                                eprintln!("ERROR: {}", msg);
                            }
                        }
                        Err(e) => {
                            eprintln!("JIT VM Error: {}", e);
                        }
                    }
                    return;
                } else {
                    if diagnostic {
                        println!("Programme contains temporal operations, falling back to standard VM");
                    }
                }
            }

            // Determine execution mode
            let mode = if action_mode {
                println!("Running in ACTION-GUIDED mode (exploring {} seeds).", num_seeds);
                ExecutionMode::ActionGuided {
                    config: ActionConfig::anti_trivial(),
                    num_seeds,
                }
            } else if diagnostic {
                println!("Running in DIAGNOSTIC mode.");
                ExecutionMode::Diagnostic
            } else {
                ExecutionMode::Standard
            };
            
            let config = Config {
                max_epochs: 1000,
                mode,
                seed,
                verbose: diagnostic || action_mode,
                frozen_inputs: Vec::new(),
                max_instructions,
            };
            
            let mut driver = TimeLoop::new(config.clone());
            match driver.run(&program) {
                ConvergenceStatus::Consistent { epochs, output, .. } => {
                    // Only print consistency status if verbose/diagnostic or NO output
                    if config.verbose || output.is_empty() {
                        println!("CONSISTENT after {} epochs.", epochs);
                    }
                    
                    if !output.is_empty() {
                         // Print raw output without label
                         for val in output {
                              match val {
                                  ourochronos::OutputItem::Val(v) => print!("[{}]", v.val),
                                  ourochronos::OutputItem::Char(c) => print!("{}", c as char),
                              }
                         }
                         println!(); // Newline
                    }
                },
                ConvergenceStatus::Paradox { message, .. } => {
                    println!("PARADOX: {}", message);
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
                            ourochronos::timeloop::ParadoxDiagnosis::NegativeLoop { explanation, .. } => {
                                println!("\nDIAGNOSIS (Grandfather Paradox):");
                                println!("{}", explanation);
                            },
                            ourochronos::timeloop::ParadoxDiagnosis::Oscillation { cycle } => {
                                println!("\nCycle states:");
                                for (i, state) in cycle.iter().enumerate() {
                                     // Only print non-empty states for brevity
                                     let non_zeros: Vec<_> = state.iter().filter(|(_,v)| *v != 0).collect();
                                     if !non_zeros.is_empty() {
                                         println!("  State {}: {:?}", i, non_zeros);
                                     }
                                }
                            },
                            ourochronos::timeloop::ParadoxDiagnosis::Unknown => {
                                println!("\nDIAGNOSIS: Unknown cause");
                            }
                        }
                    }
                },
                ConvergenceStatus::Timeout { max_epochs } => {
                    println!("TIMEOUT after {} epochs.", max_epochs);
                },
                ConvergenceStatus::Divergence { .. } => {
                     println!("DIVERGENCE detected.");
                },
                ConvergenceStatus::Error { message, .. } => {
                     println!("ERROR: {}", message);
                }
            }
        },
        Err(e) => {
            eprintln!("Parse Error: {}", e);
        }
    }
}
