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
//! - `:load <file>` - Load and execute a file

use std::io::{self, BufRead, Write};
use std::path::Path;

use super::frontend::{analyze_file_source, analyze_virtual_source, ToolingAnalysis};
use crate::bytecode_timeloop::{BytecodeTimeLoop, BytecodeTimeLoopConfig};
use crate::bytecode_vm::BytecodeVmConfig;
use crate::core::Memory;
use crate::temporal::timeloop::ConvergenceStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplRead {
    EndOfFile,
    Line,
    TooLong,
}

fn discard_through_newline(reader: &mut impl BufRead) -> io::Result<()> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(());
        }
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        let finished = available.get(consumed.saturating_sub(1)) == Some(&b'\n');
        reader.consume(consumed);
        if finished {
            return Ok(());
        }
    }
}

fn read_bounded_repl_line(
    reader: &mut impl BufRead,
    input: &mut String,
    limit: usize,
) -> io::Result<ReplRead> {
    input.clear();
    let read = {
        let take_limit = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
        let mut bounded = std::io::Read::take(&mut *reader, take_limit);
        bounded.read_line(input)?
    };
    if read == 0 {
        return Ok(ReplRead::EndOfFile);
    }
    if input.len() > limit {
        if !input.ends_with('\n') {
            discard_through_newline(reader)?;
        }
        input.clear();
        return Ok(ReplRead::TooLong);
    }
    Ok(ReplRead::Line)
}

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
    /// Show type information.
    pub show_types: bool,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "ouro> ".to_string(),
            max_epochs: crate::temporal::timeloop::DEFAULT_MAX_EPOCHS,
            verbose: false,
            show_memory: false,
            show_types: false,
        }
    }
}

/// Interactive REPL for OUROCHRONOS.
pub struct Repl {
    config: ReplConfig,
    memory: Memory,
    history: Vec<String>,
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
        }
    }

    /// Run the interactive REPL.
    pub fn run(&mut self) -> io::Result<()> {
        println!("OUROCHRONOS REPL v0.2.0");
        println!("Type :help for commands, :quit to exit");
        println!();

        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        let mut stdout = io::stdout();
        let mut input = String::new();

        loop {
            // Print prompt
            print!("{}", self.config.prompt);
            stdout.flush()?;

            // Read input
            match read_bounded_repl_line(
                &mut stdin,
                &mut input,
                crate::source::MAX_SOURCE_FILE_BYTES,
            )? {
                ReplRead::EndOfFile => break,
                ReplRead::TooLong => {
                    eprintln!(
                        "REPL input exceeds {} bytes and was discarded",
                        crate::source::MAX_SOURCE_FILE_BYTES
                    );
                    continue;
                }
                ReplRead::Line => {}
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
                println!("Memory cleared.");
            }
            ":history" => {
                for (i, line) in self.history.iter().enumerate() {
                    println!("{}: {}", i + 1, line);
                }
            }
            ":verbose" => {
                self.config.verbose = !self.config.verbose;
                println!(
                    "Verbose mode: {}",
                    if self.config.verbose { "on" } else { "off" }
                );
            }
            ":type" | ":t" => {
                if args.is_empty() {
                    println!("Usage: :type <code>");
                } else {
                    self.type_check(args);
                }
            }
            ":load" | ":l" => {
                if args.is_empty() {
                    println!("Usage: :load <file>");
                } else {
                    self.load_file(args);
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
        println!("  :clear, :c        Clear memory");
        println!("  :history          Show command history");
        println!("  :verbose          Toggle verbose mode");
        println!();
        println!("Analysis Commands:");
        println!("  :type <code>      Type check code without executing");
        println!("  :load <file>      Load and execute a file");
        println!();
        println!("Enter OUROCHRONOS code to evaluate.");
    }

    /// Type check code without executing.
    fn type_check(&self, code: &str) {
        let analysis = analyze_virtual_source("<repl:type>", code, self.memory.len());
        let Some(result) = analysis.types.as_ref() else {
            Self::show_compile_errors(&analysis);
            return;
        };

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
                println!(
                    "    - {}: declared {:?}, actual {:?}",
                    v.procedure_name, v.declared, v.actual
                );
            }
        }

        if result.has_linear_violations() {
            println!("  Linear violations:");
            for v in &result.linear_violations {
                println!(
                    "    - {} at stmt {}: {}",
                    v.operation, v.stmt_index, v.message
                );
            }
        }

        if let Some(semantics) = &analysis.semantics {
            println!(
                "  Structural stack obligations: {}",
                semantics.obligations.len()
            );
        }
        if let Some(regions) = &analysis.regions {
            if !regions.regions.is_empty() || !regions.host_effects.is_empty() {
                print!("{}", regions);
            }
        }

        // Type checking in the REPL is the complete compiler admission path,
        // not the legacy type pass alone.
        Self::show_compile_errors(&analysis);
    }

    /// Load and execute a file.
    fn load_file(&mut self, filename: &str) -> Option<ConvergenceStatus> {
        let path = Path::new(filename);
        println!("Loading '{}'", filename);
        let analysis = analyze_file_source(path, None, self.memory.len());
        self.eval_analysis(analysis)
    }

    /// Evaluate code.
    pub fn eval(&mut self, code: &str) -> Option<ConvergenceStatus> {
        let analysis = analyze_virtual_source("<repl>", code, self.memory.len());
        self.eval_analysis(analysis)
    }

    fn eval_analysis(&mut self, analysis: ToolingAnalysis) -> Option<ConvergenceStatus> {
        if !analysis.is_executable() {
            Self::show_compile_errors(&analysis);
            return None;
        }
        let bytecode = analysis
            .bytecode
            .as_ref()
            .expect("an executable analysis contains bytecode");

        let config = BytecodeTimeLoopConfig {
            max_epochs: self.config.max_epochs,
            memory_cells: self.memory.len(),
            initial_state: self
                .memory
                .non_zero_cells()
                .into_iter()
                .map(|(address, value)| (address, value.val))
                .collect(),
            vm: BytecodeVmConfig {
                allow_interactive_input: true,
                ..BytecodeVmConfig::default()
            },
            ..BytecodeTimeLoopConfig::default()
        };

        let timeloop = match BytecodeTimeLoop::new(config) {
            Ok(timeloop) => timeloop,
            Err(e) => {
                println!("Configuration error: {}", e);
                return None;
            }
        };
        let result = timeloop.run(bytecode);

        match &result {
            ConvergenceStatus::Consistent {
                memory,
                output,
                epochs,
            } => {
                self.memory = memory.clone();
                if !output.is_empty() {
                    print!("Output: ");
                    for val in output {
                        match val {
                            crate::core::OutputItem::Val(v) => print!("[{}]", v.val),
                            crate::core::OutputItem::Char(c) => print!("{}", *c as char),
                        }
                    }
                    println!();
                }
                if self.config.verbose {
                    println!("Converged in {} epochs", epochs);
                }
            }
            ConvergenceStatus::DeutschConsistent {
                period,
                unanimous_output,
                ..
            } => {
                println!("Deutsch-consistent stationary cycle (period {})", period);
                if unanimous_output.is_none() {
                    println!("Readout is ambiguous across the cycle");
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

    fn show_compile_errors(analysis: &ToolingAnalysis) {
        for diagnostic in &analysis.diagnostics {
            if let Some(span) = diagnostic.span {
                println!(
                    "{} error at bytes {}..{}: {}",
                    diagnostic.phase.code(),
                    span.range.start,
                    span.range.end,
                    diagnostic.message
                );
            } else {
                println!("{} error: {}", diagnostic.phase.code(), diagnostic.message);
            }
        }
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
    use crate::core::OutputItem;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ourochronos-repl-tests-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, contents: &str) -> PathBuf {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn output_values(status: ConvergenceStatus) -> Vec<u64> {
        match status {
            ConvergenceStatus::Consistent { output, .. } => output
                .into_iter()
                .map(|item| match item {
                    OutputItem::Val(value) => value.val,
                    OutputItem::Char(value) => u64::from(value),
                })
                .collect(),
            other => panic!("expected a consistent REPL result, got {other:?}"),
        }
    }

    #[test]
    fn test_repl_creation() {
        let repl = Repl::new();
        assert!(repl.history.is_empty());
    }

    #[test]
    fn oversized_repl_line_is_discarded_without_consuming_the_next_command() {
        let mut bytes = vec![b'x'; 17];
        bytes.extend_from_slice(b"\n:quit\n");
        let mut reader = std::io::Cursor::new(bytes);
        let mut input = String::new();
        assert_eq!(
            read_bounded_repl_line(&mut reader, &mut input, 16).unwrap(),
            ReplRead::TooLong
        );
        assert!(input.is_empty());
        assert_eq!(
            read_bounded_repl_line(&mut reader, &mut input, 16).unwrap(),
            ReplRead::Line
        );
        assert_eq!(input, ":quit\n");
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
        assert!(matches!(
            repl.eval("1 0 PROPHECY"),
            Some(ConvergenceStatus::Consistent { .. })
        ));
        let result = repl
            .eval("0 ORACLE DUP OUTPUT 0 PROPHECY")
            .expect("the second entry must execute");
        assert_eq!(output_values(result), vec![1]);
        assert_eq!(repl.memory().read(0).val, 1);
    }

    #[test]
    fn load_uses_importer_relative_graph_and_runs_dependency_initializer() {
        let temp = TempDir::new();
        temp.write("library/dependency.ouro", "7 OUTPUT");
        let root = temp.write(
            "application/root.ouro",
            "IMPORT \"../library/dependency.ouro\"\n8 OUTPUT",
        );

        let mut repl = Repl::new();
        let result = repl
            .load_file(root.to_str().unwrap())
            .expect("a canonical file graph must execute");
        assert_eq!(output_values(result), vec![7, 8]);
    }

    #[test]
    fn repl_rejects_fallible_lexing_before_execution() {
        let mut repl = Repl::new();
        assert!(repl.eval("1 ? OUTPUT").is_none());
    }

    #[test]
    fn repl_executes_procedures_through_verified_bytecode() {
        let mut repl = Repl::new();
        let result = repl.eval("PROCEDURE answer { 40 2 ADD } answer OUTPUT");
        assert!(matches!(result, Some(ConvergenceStatus::Consistent { .. })));
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
    fn test_repl_config() {
        let config = ReplConfig {
            prompt: "test> ".to_string(),
            max_epochs: 50,
            verbose: true,
            show_memory: true,
            show_types: true,
        };

        let repl = Repl::with_config(config);
        assert!(repl.config.verbose);
        assert!(repl.config.show_memory);
    }

    #[test]
    fn test_repl_handle_command() {
        let mut repl = Repl::new();

        // Test :verbose toggle
        assert!(!repl.config.verbose);
        let quit = repl.handle_command(":verbose");
        assert!(!quit);
        assert!(repl.config.verbose);

        // Test unknown command doesn't quit
        let quit = repl.handle_command(":unknown");
        assert!(!quit);

        // Test :quit returns true
        let quit = repl.handle_command(":quit");
        assert!(quit);
    }
}
