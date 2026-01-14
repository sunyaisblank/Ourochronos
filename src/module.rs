//! Module system for OUROCHRONOS.
//!
//! Provides support for organizing code into reusable modules with imports/exports.
//!
//! # Syntax
//!
//! ```ourochronos
//! # primality.ouro
//! MODULE primality;
//!
//! EXPORT PROCEDURE is_factor { ... }
//!
//! # main.ouro
//! IMPORT primality;
//!
//! 15 3 primality::is_factor IF { ... }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;

use crate::ast::Procedure;

/// A module definition.
#[derive(Debug, Clone)]
pub struct Module {
    /// Module name.
    pub name: String,
    /// Exported procedures.
    pub exports: Vec<String>,
    /// All procedures defined in the module.
    pub procedures: Vec<Procedure>,
    /// Source file path.
    pub source_path: PathBuf,
}

impl Module {
    /// Create a new module with the given name.
    pub fn new(name: String, source_path: PathBuf) -> Self {
        Self {
            name,
            exports: Vec::new(),
            procedures: Vec::new(),
            source_path,
        }
    }
    
    /// Check if a procedure is exported.
    pub fn is_exported(&self, proc_name: &str) -> bool {
        self.exports.contains(&proc_name.to_string()) || self.exports.is_empty()
    }
    
    /// Get an exported procedure by name.
    pub fn get_procedure(&self, name: &str) -> Option<&Procedure> {
        self.procedures.iter().find(|p| p.name == name)
    }
}

/// Module registry for tracking loaded modules.
#[derive(Debug, Default)]
pub struct ModuleRegistry {
    /// Map from module name to module.
    modules: HashMap<String, Module>,
    /// Search paths for module resolution.
    search_paths: Vec<PathBuf>,
}

impl ModuleRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a search path for module resolution.
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }
    
    /// Register a module.
    pub fn register(&mut self, module: Module) {
        self.modules.insert(module.name.clone(), module);
    }
    
    /// Get a module by name.
    pub fn get(&self, name: &str) -> Option<&Module> {
        self.modules.get(name)
    }
    
    /// Resolve a qualified name (module::procedure) to a procedure.
    pub fn resolve_qualified(&self, qualified_name: &str) -> Option<&Procedure> {
        let parts: Vec<&str> = qualified_name.split("::").collect();
        if parts.len() != 2 {
            return None;
        }
        
        let module_name = parts[0];
        let proc_name = parts[1];
        
        self.get(module_name)
            .filter(|m| m.is_exported(proc_name))
            .and_then(|m| m.get_procedure(proc_name))
    }
    
    /// Load a module from file.
    pub fn load_module(&mut self, name: &str) -> Result<(), String> {
        // Search for module file
        let filename = format!("{}.ouro", name);
        
        let mut found_path: Option<PathBuf> = None;
        for search_path in &self.search_paths {
            let candidate = search_path.join(&filename);
            if candidate.exists() {
                found_path = Some(candidate);
                break;
            }
        }
        
        let path = found_path.ok_or_else(|| format!("Module not found: {}", name))?;
        
        // Read and parse
        let source = fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read module {}: {}", name, e))?;
        
        // Parse the module (simplified - would need to extract MODULE/EXPORT declarations)
        let program = crate::parser::parse(&source)?;
        
        let mut module = Module::new(name.to_string(), path);
        module.procedures = program.procedures;
        
        // Mark all procedures as exported (simplified)
        module.exports = module.procedures.iter().map(|p| p.name.clone()).collect();
        
        self.register(module);
        Ok(())
    }
    
    /// List all registered modules.
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }
}

/// Parse MODULE declaration from source.
pub fn parse_module_declaration(source: &str) -> Option<String> {
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with("MODULE") {
            // Extract module name
            let rest = line.strip_prefix("MODULE")?.trim();
            let name = rest.trim_end_matches(';').trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        // Stop at first non-comment, non-declaration line
        if !line.is_empty() && !line.starts_with('#') && !line.starts_with("MODULE") {
            break;
        }
    }
    None
}

/// Parse EXPORT declarations from source.
pub fn parse_exports(source: &str) -> Vec<String> {
    let mut exports = Vec::new();
    
    for line in source.lines() {
        let line = line.trim();
        if line.starts_with("EXPORT") {
            // EXPORT PROCEDURE name or EXPORT name
            let rest = line.strip_prefix("EXPORT").unwrap().trim();
            let rest = rest.strip_prefix("PROCEDURE").unwrap_or(rest).trim();
            let name = rest.trim_end_matches(';').trim();
            if let Some(name) = name.split_whitespace().next() {
                exports.push(name.to_lowercase());
            }
        }
    }
    
    exports
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_module_declaration() {
        assert_eq!(
            parse_module_declaration("MODULE math;"),
            Some("math".to_string())
        );
        assert_eq!(
            parse_module_declaration("# comment\nMODULE crypto;"),
            Some("crypto".to_string())
        );
        assert_eq!(
            parse_module_declaration("10 20 ADD"),
            None
        );
    }
    
    #[test]
    fn test_parse_exports() {
        let source = "
            EXPORT PROCEDURE gcd;
            EXPORT is_prime;
            PROCEDURE helper { }
        ";
        let exports = parse_exports(source);
        assert!(exports.contains(&"gcd".to_string()));
        assert!(exports.contains(&"is_prime".to_string()));
        assert!(!exports.contains(&"helper".to_string()));
    }
    
    #[test]
    fn test_module_registry() {
        let mut registry = ModuleRegistry::new();
        
        let module = Module::new("test".to_string(), PathBuf::from("test.ouro"));
        registry.register(module);
        
        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
