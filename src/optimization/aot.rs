//! Ahead-of-Time (AOT) compilation for OUROCHRONOS.
//!
//! Compiles programs to object files that can be linked into executables.
//! Requires the `aot` feature.
//!
//! Based on the Brainfuck compiler blog Part 4.

#[cfg(feature = "aot")]
use cranelift::prelude::*;
#[cfg(feature = "aot")]
use cranelift_module::{Module, Linkage};
#[cfg(feature = "aot")]
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::Program;
use super::jit::JitError;

/// AOT compiler that produces object files.
pub struct AotCompiler {
    #[cfg(feature = "aot")]
    module: ObjectModule,
    /// Compilation statistics.
    pub stats: AotStats,
}

/// Statistics about AOT compilation.
#[derive(Debug, Default, Clone)]
pub struct AotStats {
    /// Number of functions compiled.
    pub functions_compiled: usize,
    /// Size of generated code in bytes.
    pub code_size: usize,
    /// Number of relocations.
    pub relocations: usize,
}

/// Result of AOT compilation.
pub struct ObjectFile {
    /// Raw object file bytes.
    pub bytes: Vec<u8>,
    /// Target triple.
    pub target: String,
}

impl AotCompiler {
    /// Create a new AOT compiler for the host platform.
    pub fn new() -> Result<Self, JitError> {
        #[cfg(feature = "aot")]
        {
            let mut flag_builder = settings::builder();
            flag_builder.set("opt_level", "speed").unwrap();
            flag_builder.set("is_pic", "true").unwrap();
            
            let isa_builder = cranelift_native::builder()
                .map_err(|e| JitError::Compilation(format!("ISA error: {}", e)))?;
            
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| JitError::Compilation(format!("ISA finish: {}", e)))?;
            
            let builder = ObjectBuilder::new(
                isa,
                "ourochronos_program",
                cranelift_module::default_libcall_names(),
            ).map_err(|e| JitError::Compilation(format!("ObjectBuilder: {}", e)))?;
            
            let module = ObjectModule::new(builder);
            
            Ok(Self {
                module,
                stats: AotStats::default(),
            })
        }
        
        #[cfg(not(feature = "aot"))]
        {
            Ok(Self {
                stats: AotStats::default(),
            })
        }
    }
    
    /// Compile a program to an object file.
    pub fn compile(&mut self, program: &Program) -> Result<ObjectFile, JitError> {
        #[cfg(feature = "aot")]
        {
            self.compile_to_object(program)
        }
        
        #[cfg(not(feature = "aot"))]
        {
            let _ = program;
            Err(JitError::FeatureDisabled)
        }
    }
    
    /// Check if AOT is available.
    pub fn is_available() -> bool {
        cfg!(feature = "aot")
    }
    
    #[cfg(feature = "aot")]
    fn compile_to_object(&mut self, program: &Program) -> Result<ObjectFile, JitError> {
        // Create main function signature
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // memory pointer
        sig.returns.push(AbiParam::new(types::I64)); // result
        
        // Declare the main epoch function
        let func_id = self.module
            .declare_function("ouro_epoch_main", Linkage::Export, &sig)
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
            
            // Get memory pointer parameter
            let _memory_ptr = builder.block_params(entry)[0];
            
            // Translate program body (simplified)
            for _stmt in &program.body {
                // Full translation would go here
            }
            
            // Return 0 (success)
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().return_(&[zero]);
            builder.finalize();
        }
        
        // Define function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Compilation(format!("Define: {}", e)))?;
        
        self.module.clear_context(&mut ctx);
        self.stats.functions_compiled += 1;
        
        // Emit object file - need to recreate module since finish() consumes it
        // Create a fresh module to replace the old one
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder.set("is_pic", "true").unwrap();
        
        let isa_builder = cranelift_native::builder()
            .map_err(|e| JitError::Compilation(format!("ISA error: {}", e)))?;
        
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Compilation(format!("ISA finish: {}", e)))?;
        
        let builder = ObjectBuilder::new(
            isa,
            "ourochronos_program",
            cranelift_module::default_libcall_names(),
        ).map_err(|e| JitError::Compilation(format!("ObjectBuilder: {}", e)))?;
        
        let new_module = ObjectModule::new(builder);
        let old_module = std::mem::replace(&mut self.module, new_module);
        
        let product = old_module.finish();
        let bytes = product.emit()
            .map_err(|e| JitError::Compilation(format!("Emit: {}", e)))?;
        
        self.stats.code_size = bytes.len();
        
        Ok(ObjectFile {
            bytes,
            target: "native".to_string(),
        })
    }
}

impl ObjectFile {
    /// Write the object file to disk.
    pub fn write_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, &self.bytes)
    }
    
    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aot_availability() {
        let available = AotCompiler::is_available();
        #[cfg(feature = "aot")]
        assert!(available);
        #[cfg(not(feature = "aot"))]
        assert!(!available);
    }
    
    #[test]
    fn test_aot_stats() {
        let stats = AotStats::default();
        assert_eq!(stats.functions_compiled, 0);
        assert_eq!(stats.code_size, 0);
    }
}
