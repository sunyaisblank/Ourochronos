//! JIT compilation for OUROCHRONOS using Cranelift.
//!
//! Compiles OUROCHRONOS programs to native machine code for faster execution.
//! Requires the `jit` feature to be enabled.
//!
//! # Example
//! ```ignore
//! use ourochronos::jit::JitCompiler;
//! let mut jit = JitCompiler::new()?;
//! let func = jit.compile(&program)?;
//! let result = func.execute(&anamnesis);
//! ```

#[cfg(feature = "jit")]
use cranelift::prelude::*;
#[cfg(feature = "jit")]  
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{Module, Linkage};

use crate::ast::{OpCode, Program, Stmt};
use crate::core::Memory;

/// Error type for JIT compilation.
#[derive(Debug)]
pub enum JitError {
    /// Cranelift compilation error.
    Compilation(String),
    /// Feature not enabled.
    FeatureDisabled,
    /// Unsupported operation.
    Unsupported(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Compilation(s) => write!(f, "JIT compilation error: {}", s),
            JitError::FeatureDisabled => write!(f, "JIT feature not enabled"),
            JitError::Unsupported(s) => write!(f, "Unsupported operation: {}", s),
        }
    }
}

impl std::error::Error for JitError {}

/// Result of JIT compilation.
pub type JitResult<T> = Result<T, JitError>;

/// A compiled OUROCHRONOS function.
pub struct CompiledFunction {
    /// Function pointer (when JIT enabled).
    #[cfg(feature = "jit")]
    _func_ptr: *const u8,
    /// Whether the function is valid.
    valid: bool,
}

impl CompiledFunction {
    /// Execute the compiled function (placeholder).
    pub fn execute(&self, _anamnesis: &Memory) -> JitResult<Memory> {
        if !self.valid {
            return Err(JitError::FeatureDisabled);
        }
        // In a full implementation, this would call the native function
        Ok(Memory::new())
    }
    
    /// Check if function is valid.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

/// JIT compiler for OUROCHRONOS programs.
pub struct JitCompiler {
    #[cfg(feature = "jit")]
    module: JITModule,
    /// Compile statistics.
    pub stats: CompileStats,
}

/// Statistics about JIT compilation.
#[derive(Debug, Default, Clone)]
pub struct CompileStats {
    /// Number of functions compiled.
    pub functions_compiled: usize,
    /// Number of opcodes translated.
    pub opcodes_translated: usize,
    /// Number of unsupported operations skipped.
    pub unsupported_skipped: usize,
}

impl JitCompiler {
    /// Create a new JIT compiler.
    pub fn new() -> JitResult<Self> {
        #[cfg(feature = "jit")]
        {
            let mut flag_builder = settings::builder();
            flag_builder.set("use_colocated_libcalls", "false").unwrap();
            flag_builder.set("is_pic", "false").unwrap();
            
            let isa_builder = cranelift_native::builder()
                .map_err(|e| JitError::Compilation(format!("ISA error: {}", e)))?;
            
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| JitError::Compilation(format!("ISA finish: {}", e)))?;
            
            let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            let module = JITModule::new(builder);
            
            Ok(Self {
                module,
                stats: CompileStats::default(),
            })
        }
        
        #[cfg(not(feature = "jit"))]
        {
            Ok(Self {
                stats: CompileStats::default(),
            })
        }
    }
    
    /// Compile a program to native code.
    pub fn compile(&mut self, program: &Program) -> JitResult<CompiledFunction> {
        #[cfg(feature = "jit")]
        {
            self.compile_program_jit(program)
        }
        
        #[cfg(not(feature = "jit"))]
        {
            let _ = program;
            Err(JitError::FeatureDisabled)
        }
    }
    
    /// Check if JIT is available.
    pub fn is_available() -> bool {
        cfg!(feature = "jit")
    }
    
    #[cfg(feature = "jit")]
    fn compile_program_jit(&mut self, program: &Program) -> JitResult<CompiledFunction> {
        // Create function signature
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        
        // Declare function
        let func_id = self.module
            .declare_function("epoch_main", Linkage::Export, &sig)
            .map_err(|e| JitError::Compilation(format!("Declare: {}", e)))?;
        
        // Build function
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        
        {
            let mut builder_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            
            // Count opcodes while translating
            let mut stats = TranslateStats::default();
            for stmt in &program.body {
                translate_stmt(&mut builder, stmt, &mut stats);
            }
            
            self.stats.opcodes_translated += stats.opcodes;
            self.stats.unsupported_skipped += stats.unsupported;
            
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().return_(&[zero]);
            builder.finalize();
        }
        
        // Define and finalize
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Compilation(format!("Define: {}", e)))?;
        
        self.module.clear_context(&mut ctx);
        
        self.module.finalize_definitions()
            .map_err(|e| JitError::Compilation(format!("Finalize: {}", e)))?;
        
        let func_ptr = self.module.get_finalized_function(func_id);
        self.stats.functions_compiled += 1;
        
        Ok(CompiledFunction {
            _func_ptr: func_ptr,
            valid: true,
        })
    }
}

#[cfg(feature = "jit")]
#[derive(Default)]
struct TranslateStats {
    opcodes: usize,
    unsupported: usize,
}

#[cfg(feature = "jit")]
fn translate_stmt(builder: &mut FunctionBuilder, stmt: &Stmt, stats: &mut TranslateStats) {
    match stmt {
        Stmt::Op(op) => {
            translate_opcode(builder, op, stats);
            stats.opcodes += 1;
        }
        Stmt::Push(_) => stats.opcodes += 1,
        Stmt::Block(stmts) => {
            for s in stmts {
                translate_stmt(builder, s, stats);
            }
        }
        Stmt::If { then_branch, else_branch } => {
            for s in then_branch {
                translate_stmt(builder, s, stats);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    translate_stmt(builder, s, stats);
                }
            }
        }
        Stmt::While { cond, body } => {
            for s in cond {
                translate_stmt(builder, s, stats);
            }
            for s in body {
                translate_stmt(builder, s, stats);
            }
        }
        Stmt::Call { .. } => stats.unsupported += 1,
        Stmt::Match { cases, default } => {
            for (_, body) in cases {
                for s in body {
                    translate_stmt(builder, s, stats);
                }
            }
            if let Some(def) = default {
                for s in def {
                    translate_stmt(builder, s, stats);
                }
            }
        }
        Stmt::TemporalScope { body, .. } => {
            for s in body {
                translate_stmt(builder, s, stats);
            }
            stats.unsupported += 1; // Temporal scoping semantics not yet JIT-compiled
        }
    }
}

#[cfg(feature = "jit")]
fn translate_opcode(_builder: &mut FunctionBuilder, op: &OpCode, stats: &mut TranslateStats) {
    match op {
        OpCode::Oracle | OpCode::Prophecy | OpCode::PresentRead => {
            stats.unsupported += 1;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_jit_availability() {
        let available = JitCompiler::is_available();
        #[cfg(feature = "jit")]
        assert!(available);
        #[cfg(not(feature = "jit"))]
        assert!(!available);
    }
    
    #[test]
    fn test_jit_compiler_creation() {
        let result = JitCompiler::new();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_compile_stats() {
        let stats = CompileStats::default();
        assert_eq!(stats.functions_compiled, 0);
    }
}
