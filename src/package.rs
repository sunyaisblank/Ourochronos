//! Package Manager for OUROCHRONOS.
//!
//! Provides package discovery, installation, and dependency management
//! for temporal code libraries.

use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;

/// Package metadata.
#[derive(Debug, Clone)]
pub struct PackageManifest {
    /// Package name.
    pub name: String,
    /// Version string.
    pub version: String,
    /// Description.
    pub description: String,
    /// Authors.
    pub authors: Vec<String>,
    /// Dependencies.
    pub dependencies: Vec<Dependency>,
    /// Main entry file.
    pub main: String,
}

/// A dependency specification.
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Package name.
    pub name: String,
    /// Version constraint.
    pub version: String,
    /// Optional.
    pub optional: bool,
}

/// Package source location.
#[derive(Debug, Clone)]
pub enum PackageSource {
    /// Local filesystem path.
    Local(PathBuf),
    /// Registry URL.
    Registry(String),
    /// Git repository.
    Git { url: String, branch: Option<String> },
}

/// A resolved package.
#[derive(Debug, Clone)]
pub struct Package {
    /// Manifest.
    pub manifest: PackageManifest,
    /// Source location.
    pub source: PackageSource,
    /// Resolved path (after install).
    pub path: Option<PathBuf>,
}

/// Package manager.
pub struct PackageManager {
    /// Package cache directory.
    cache_dir: PathBuf,
    /// Installed packages.
    installed: HashMap<String, Package>,
    /// Registry URL.
    registry_url: String,
}

impl Default for PackageManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PackageManager {
    /// Create a new package manager.
    pub fn new() -> Self {
        let cache_dir = dirs_or_default();
        Self {
            cache_dir,
            installed: HashMap::new(),
            registry_url: "https://packages.ourochronos.dev".to_string(),
        }
    }
    
    /// Create with custom cache directory.
    pub fn with_cache(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            installed: HashMap::new(),
            registry_url: "https://packages.ourochronos.dev".to_string(),
        }
    }
    
    /// Initialize a new package in the current directory.
    pub fn init(&self, name: &str, path: &PathBuf) -> Result<PackageManifest, String> {
        let manifest = PackageManifest {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            description: "An Ourochronos package".to_string(),
            authors: vec![],
            dependencies: vec![],
            main: "main.ouro".to_string(),
        };
        
        // Create manifest file
        let manifest_path = path.join("ouro.toml");
        let content = format!(
            r#"[package]
name = "{}"
version = "{}"
description = "{}"
main = "{}"

[dependencies]
"#,
            manifest.name, manifest.version, manifest.description, manifest.main
        );
        
        fs::write(&manifest_path, content)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;
        
        // Create main file
        let main_path = path.join(&manifest.main);
        if !main_path.exists() {
            fs::write(&main_path, "# Welcome to Ourochronos\n0 ORACLE 0 PROPHECY\n")
                .map_err(|e| format!("Failed to write main file: {}", e))?;
        }
        
        Ok(manifest)
    }
    
    /// Load a package manifest from a directory.
    pub fn load_manifest(&self, path: &PathBuf) -> Result<PackageManifest, String> {
        let manifest_path = path.join("ouro.toml");
        let content = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        
        Self::parse_manifest(&content)
    }
    
    /// Parse a manifest from TOML content.
    fn parse_manifest(content: &str) -> Result<PackageManifest, String> {
        // Simple TOML-like parsing (production would use toml crate)
        let mut name = String::new();
        let mut version = String::new();
        let mut description = String::new();
        let mut main = "main.ouro".to_string();
        let mut dependencies = Vec::new();
        
        let mut in_package = false;
        let mut in_deps = false;
        
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            if line == "[package]" {
                in_package = true;
                in_deps = false;
            } else if line == "[dependencies]" {
                in_package = false;
                in_deps = true;
            } else if in_package {
                if let Some((key, val)) = line.split_once('=') {
                    let key = key.trim();
                    let val = val.trim().trim_matches('"');
                    match key {
                        "name" => name = val.to_string(),
                        "version" => version = val.to_string(),
                        "description" => description = val.to_string(),
                        "main" => main = val.to_string(),
                        _ => {}
                    }
                }
            } else if in_deps {
                if let Some((dep_name, dep_ver)) = line.split_once('=') {
                    dependencies.push(Dependency {
                        name: dep_name.trim().to_string(),
                        version: dep_ver.trim().trim_matches('"').to_string(),
                        optional: false,
                    });
                }
            }
        }
        
        if name.is_empty() {
            return Err("Missing package name".to_string());
        }
        
        Ok(PackageManifest {
            name,
            version,
            description,
            authors: vec![],
            dependencies,
            main,
        })
    }
    
    /// Install a package by name.
    pub fn install(&mut self, name: &str) -> Result<Package, String> {
        // Check if already installed
        if let Some(pkg) = self.installed.get(name) {
            return Ok(pkg.clone());
        }
        
        // In a real implementation, this would:
        // 1. Query the registry for package metadata
        // 2. Download the package
        // 3. Resolve and install dependencies
        // 4. Install to cache directory
        
        let pkg_path = self.cache_dir.join(name);
        fs::create_dir_all(&pkg_path)
            .map_err(|e| format!("Failed to create package dir: {}", e))?;
        
        let package = Package {
            manifest: PackageManifest {
                name: name.to_string(),
                version: "0.0.0".to_string(),
                description: "Downloaded package".to_string(),
                authors: vec![],
                dependencies: vec![],
                main: "main.ouro".to_string(),
            },
            source: PackageSource::Registry(self.registry_url.clone()),
            path: Some(pkg_path),
        };
        
        self.installed.insert(name.to_string(), package.clone());
        Ok(package)
    }
    
    /// List installed packages.
    pub fn list(&self) -> Vec<&Package> {
        self.installed.values().collect()
    }
    
    /// Get cache directory.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

/// Get default cache directory.
fn dirs_or_default() -> PathBuf {
    std::env::var("OURO_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".ouro").join("packages")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_package_manager_creation() {
        let pm = PackageManager::new();
        assert!(pm.installed.is_empty());
    }
    
    #[test]
    fn test_manifest_parsing() {
        let content = r#"
[package]
name = "test-pkg"
version = "1.0.0"
description = "A test package"
main = "lib.ouro"

[dependencies]
std = "0.1.0"
"#;
        let manifest = PackageManager::parse_manifest(content).unwrap();
        assert_eq!(manifest.name, "test-pkg");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.main, "lib.ouro");
        assert_eq!(manifest.dependencies.len(), 1);
    }
    
    #[test]
    fn test_list_empty() {
        let pm = PackageManager::new();
        assert!(pm.list().is_empty());
    }
}
