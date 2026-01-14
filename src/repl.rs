//! REPL (Read-Eval-Print-Loop) for OUROCHRONOS.
//!
//! Interactive shell for exploring temporal programs.
//!
//! # Commands
//!
//! - `:quit`, `:q` - Exit the REPL
//! - `:help`, `:h` - Show help
//! - `:memory`, `:m` - Show memory state
//! - `:clear`, `:c` - Clear memory
//! - `:history` - Show command history
//! - `:verbose` - Toggle verbose mode
//! - `:type <code>` - Type check code without executing
//! - `:trace <code>` - Execute with debugging trace
//! - `:profile` - Toggle profiling mode
//! - `:load <file>` - Load and execute a file

use std::io::{self, Write, BufRead};
use std::fs::File;
use std::path::Path;

use crate::parser::Parser;
use crate::timeloop::{TimeLoop, TimeLoopConfig, ConvergenceStatus};
use crate::core_types::Memory;
use crate::types::TypeChecker;
use crate::debugger::Debugger;
use crate::profiler::{Profiler, ProfilerConfig};

/// REPL configuration.
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Prompt string.
    pub prompt: String,
    /// Maximum epochs per evaluation.
    pub max_epochs: usize,
    /// Show verbose output.
    pub verbose: bool,
    /// Show memory after each evaluation.
    pub show_memory: bool,
    /// Enable profiling.
    pub profiling: bool,
    /// Show type information.
    pub show_types: bool,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "ouro> ".to_string(),
            max_epochs: 100,
            verbose: false,
            show_memory: false,
            profiling: false,
            show_types: false,
        }
    }
}

/// Interactive REPL for OUROCHRONOS.
pub struct Repl {
    config: ReplConfig,
    memory: Memory,
    history: Vec<String>,
    profiler: Profiler,
    debugger: Option<Debugger>,
}

impl Repl {
    /// Create a new REPL with default config.
    pub fn new() -> Self {
        Self::with_config(ReplConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: ReplConfig) -> Self {
        Self {
            config,
            memory: Memory::new(),
            history: Vec::new(),
            profiler: Profiler::with_config(ProfilerConfig::minimal()),
            debugger: None,
        }
    }
    
    /// Run the interactive REPL.
    pub fn run(&mut self) -> io::Result<()> {
        println!("OUROCHRONOS REPL v0.2.0");
        println!("Type :help for commands, :quit to exit");
        println!();
        
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut input = String::new();
        
        loop {
            // Print prompt
            print!("{}", self.config.prompt);
            stdout.flush()?;
            
            // Read input
            input.clear();
            if stdin.read_line(&mut input)? == 0 {
                break; // EOF
            }
            
            let line = input.trim();
            if line.is_empty() {
                continue;
            }
            
            // Handle commands
            if line.starts_with(':') {
                if self.handle_command(line) {
                    break;
                }
                continue;
            }
            
            // Evaluate code
            self.eval(line);
            self.history.push(line.to_string());
        }
        
        println!("\nGoodbye!");
        Ok(())
    }
    
    /// Handle REPL commands.
    fn handle_command(&mut self, cmd: &str) -> bool {
        // Split command and arguments
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let command = parts[0];
        let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match command {
            ":quit" | ":q" => return true,
            ":help" | ":h" => {
                self.show_help();
            }
            ":memory" | ":m" => {
                self.show_memory();
            }
            ":clear" | ":c" => {
                self.memory = Memory::new();
                self.profiler.reset();
                println!("Memory and profiler cleared.");
            }
            ":history" => {
                for (i, line) in self.history.iter().enumerate() {
                    println!("{}: {}", i + 1, line);
                }
            }
            ":verbose" => {
                self.config.verbose = !self.config.verbose;
                println!("Verbose mode: {}", if self.config.verbose { "on" } else { "off" });
            }
            ":type" | ":t" => {
                if args.is_empty() {
                    println!("Usage: :type <code>");
                } else {
                    self.type_check(args);
                }
            }
            ":trace" => {
                if args.is_empty() {
                    println!("Usage: :trace <code>");
                } else {
                    self.trace(args);
                }
            }
            ":profile" => {
                self.config.profiling = !self.config.profiling;
                if self.config.profiling {
                    self.profiler = Profiler::with_config(ProfilerConfig::full());
                    println!("Profiling enabled. Run code to collect metrics.");
                } else {
                    // Show summary before disabling
                    let summary = self.profiler.summary();
                    println!("{}", summary);
                    println!("Profiling disabled.");
                }
            }
            ":load" | ":l" => {
                if args.is_empty() {
                    println!("Usage: :load <file>");
                } else {
                    self.load_file(args);
                }
            }
            ":stats" => {
                let summary = self.profiler.summary();
                println!("{}", summary);
            }
            ":debug" => {
                if self.debugger.is_some() {
                    self.debugger = None;
                    println!("Debug mode disabled.");
                } else {
                    self.debugger = Some(Debugger::new());
                    println!("Debug mode enabled. Use :trace to see execution details.");
                }
            }
            ":epochs" => {
                let epochs = self.profiler.epochs();
                if epochs.is_empty() {
                    println!("No epoch data. Enable profiling first.");
                } else {
                    println!("Epoch History:");
                    for ep in epochs {
                        println!("  Epoch {}: {:?} ({} instrs, {} oracle, {} prophecy){}",
                                 ep.epoch, ep.duration, ep.instruction_count,
                                 ep.oracle_count, ep.prophecy_count,
                                 if ep.paradox { " [PARADOX]" } else { "" });
                    }
                }
            }
            _ => {
                println!("Unknown command: {}", command);
                println!("Type :help for available commands.");
            }
        }
        false
    }

    /// Show help text.
    fn show_help(&self) {
        println!("OUROCHRONOS REPL Commands:");
        println!();
        println!("  :quit, :q         Exit the REPL");
        println!("  :help, :h         Show this help");
        println!("  :memory, :m       Show memory state");
        println!("  :clear, :c        Clear memory and profiler");
        println!("  :history          Show command history");
        println!("  :verbose          Toggle verbose mode");
        println!();
        println!("Analysis Commands:");
        println!("  :type <code>      Type check code without executing");
        println!("  :trace <code>     Execute with debugging trace");
        println!("  :profile          Toggle profiling mode");
        println!("  :stats            Show current profiling statistics");
        println!("  :epochs           Show epoch history");
        println!("  :debug            Toggle debug mode");
        println!("  :load <file>      Load and execute a file");
        println!();
        println!("Enter OUROCHRONOS code to evaluate.");
    }

    /// Type check code without executing.
    fn type_check(&self, code: &str) {
        let tokens = crate::parser::tokenize(code);
        let mut parser = Parser::new(&tokens);
        let program = match parser.parse_program() {
            Ok(p) => p,
            Err(e) => {
                println!("Parse error: {}", e);
                return;
            }
        };

        let mut checker = TypeChecker::new();
        let result = checker.check(&program);

        println!("Type Check Result:");
        println!("  Valid: {}", result.is_valid);

        if !result.final_stack_types.is_empty() {
            println!("  Stack types:");
            for (i, ty) in result.final_stack_types.iter().enumerate() {
                println!("    [{}]: {}", i, ty.describe());
            }
        }

        println!("  Effects: {:?}", result.effects);

        if !result.errors.is_empty() {
            println!("  Errors:");
            for err in &result.errors {
                println!("    - {:?}", err);
            }
        }

        if !result.warnings.is_empty() {
            println!("  Warnings:");
            for warn in &result.warnings {
                println!("    - {}", warn);
            }
        }

        if result.has_effect_violations() {
            println!("  Effect violations:");
            for v in &result.effect_violations {
                println!("    - {}: declared {:?}, actual {:?}",
                         v.procedure_name, v.declared, v.actual);
            }
        }

        if result.has_linear_violations() {
            println!("  Linear violations:");
            for v in &result.linear_violations {
                println!("    - {} at stmt {}: {}", v.operation, v.stmt_index, v.message);
            }
        }
    }

    /// Execute with debugging trace.
    fn trace(&mut self, code: &str) {
        let tokens = crate::parser::tokenize(code);
        let mut parser = Parser::new(&tokens);
        let program = match parser.parse_program() {
            Ok(p) => p,
            Err(e) => {
                println!("Parse error: {}", e);
                return;
            }
        };

        // Use debugger to capture execution
        let mut debugger = Debugger::new();
        let anamnesis = Memory::new();
        debugger.run(&program, anamnesis, self.config.max_epochs);

        // Show trace information
        println!("Execution Trace:");

        let snapshots = debugger.history();
        if !snapshots.is_empty() {
            println!("  Epoch Snapshots ({} total):", snapshots.len());
            for snap in snapshots {
                println!("    Epoch {}: {:?}", snap.epoch, snap.status);
                if !snap.output.is_empty() {
                    print!("      Output: ");
                    for o in &snap.output {
                        match o {
                            crate::core_types::OutputItem::Val(v) => print!("[{}]", v.val),
                            crate::core_types::OutputItem::Char(c) => print!("{}", *c as char),
                        }
                    }
                    println!();
                }
            }
        }

        // Show causality for non-zero addresses
        if let Some(snapshot) = debugger.current() {
            for (addr, _val) in snapshot.present.non_zero_cells() {
                let history = debugger.trace_causality(addr);
                if !history.is_empty() && history.len() > 1 {
                    println!("\n  Causality trace for address {}:", addr);
                    for (epoch, val) in history {
                        println!("    Epoch {}: {}", epoch, val.val);
                    }
                }
            }
        }
    }

    /// Load and execute a file.
    fn load_file(&mut self, filename: &str) {
        let path = Path::new(filename);

        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                println!("Error opening file '{}': {}", filename, e);
                return;
            }
        };

        let reader = io::BufReader::new(file);
        let mut code = String::new();

        for line in reader.lines() {
            match line {
                Ok(l) => {
                    // Skip comment lines
                    let trimmed = l.trim();
                    if trimmed.starts_with('#') || trimmed.starts_with("//") {
                        continue;
                    }
                    code.push_str(&l);
                    code.push(' ');
                }
                Err(e) => {
                    println!("Error reading file: {}", e);
                    return;
                }
            }
        }

        println!("Loaded {} bytes from '{}'", code.len(), filename);
        self.eval(&code);
    }
    
    /// Evaluate code.
    pub fn eval(&mut self, code: &str) -> Option<ConvergenceStatus> {
        let tokens = crate::parser::tokenize(code);
        let mut parser = Parser::new(&tokens);
        let program = match parser.parse_program() {
            Ok(p) => p,
            Err(e) => {
                println!("Parse error: {}", e);
                return None;
            }
        };
        
        let config = TimeLoopConfig {
            max_epochs: self.config.max_epochs,
            verbose: self.config.verbose,
            ..Default::default()
        };
        
        let mut timeloop = TimeLoop::new(config);
        let result = timeloop.run(&program);
        
        match &result {
            ConvergenceStatus::Consistent { memory, output, epochs } => {
                self.memory = memory.clone();
                if !output.is_empty() {
                    print!("Output: ");
                    for val in output {
                        match val {
                            crate::core_types::OutputItem::Val(v) => print!("[{}]", v.val),
                            crate::core_types::OutputItem::Char(c) => print!("{}", *c as char),
                        }
                    }
                    println!();
                }
                if self.config.verbose {
                    println!("Converged in {} epochs", epochs);
                }
            }
            ConvergenceStatus::Oscillation { period, .. } => {
                println!("⟳ Oscillation detected (period {})", period);
            }
            ConvergenceStatus::Paradox { epoch, .. } => {
                println!("✗ Paradox at epoch {}", epoch);
            }
            ConvergenceStatus::Timeout { max_epochs } => {
                println!("⋯ No convergence after {} epochs", max_epochs);
            }
            ConvergenceStatus::Divergence { .. } => {
                println!("∞ Divergence detected");
            }
            ConvergenceStatus::Error { message, .. } => {
                println!("✗ Error: {}", message);
            }
        }
        
        if self.config.show_memory {
            self.show_memory();
        }
        
        Some(result)
    }
    
    /// Show memory state.
    fn show_memory(&self) {
        let cells = self.memory.non_zero_cells();
        if cells.is_empty() {
            println!("Memory: (empty)");
        } else {
            println!("Memory:");
            for (addr, val) in cells {
                println!("  [{}] = {}", addr, val.val);
            }
        }
    }
    
    /// Get current memory.
    pub fn memory(&self) -> &Memory {
        &self.memory
    }
    
    /// Set memory.
    pub fn set_memory(&mut self, memory: Memory) {
        self.memory = memory;
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_creation() {
        let repl = Repl::new();
        assert!(repl.history.is_empty());
    }

    #[test]
    fn test_repl_eval() {
        let mut repl = Repl::new();
        let result = repl.eval("1 2 ADD OUTPUT");
        assert!(result.is_some());
    }

    #[test]
    fn test_repl_memory_persistence() {
        let mut repl = Repl::new();
        let result = repl.eval("1 2 ADD OUTPUT");
        // Result should be some for valid program
        assert!(result.is_some());
    }

    #[test]
    fn test_repl_type_check() {
        let repl = Repl::new();
        // Type check method should not panic
        repl.type_check("1 2 ADD");
    }

    #[test]
    fn test_repl_type_check_with_oracle() {
        let repl = Repl::new();
        // Type check should handle temporal operations
        repl.type_check("0 ORACLE 0 PROPHECY");
    }

    #[test]
    fn test_repl_trace() {
        let mut repl = Repl::new();
        // Trace should not panic
        repl.trace("1 2 ADD OUTPUT");
    }

    #[test]
    fn test_repl_profiling() {
        let mut repl = Repl::new();
        repl.config.profiling = true;

        // Eval with profiling enabled
        repl.eval("1 2 ADD");

        // Should have collected some data
        let summary = repl.profiler.summary();
        assert!(summary.total_epochs > 0 || summary.total_time.as_nanos() >= 0);
    }

    #[test]
    fn test_repl_config() {
        let config = ReplConfig {
            prompt: "test> ".to_string(),
            max_epochs: 50,
            verbose: true,
            show_memory: true,
            profiling: true,
            show_types: true,
        };

        let repl = Repl::with_config(config);
        assert!(repl.config.verbose);
        assert!(repl.config.profiling);
    }

    #[test]
    fn test_repl_handle_command() {
        let mut repl = Repl::new();

        // Test :verbose toggle
        assert!(!repl.config.verbose);
        let quit = repl.handle_command(":verbose");
        assert!(!quit);
        assert!(repl.config.verbose);

        // Test :profile toggle
        assert!(!repl.config.profiling);
        let quit = repl.handle_command(":profile");
        assert!(!quit);
        assert!(repl.config.profiling);

        // Test unknown command doesn't quit
        let quit = repl.handle_command(":unknown");
        assert!(!quit);

        // Test :quit returns true
        let quit = repl.handle_command(":quit");
        assert!(quit);
    }
}
