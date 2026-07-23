//! Source-module to relocatable-object compilation.
//!
//! Ourochronos source does not yet contain an `EXPORT` declaration. The
//! language-level visibility rule is therefore deliberately simple and
//! closed: every procedure and foreign declaration is public to importers,
//! MANIFEST words and named temporal schemas are public, anonymous quotations
//! are private to their defining module, properties are retained as linked
//! metadata, and each top-level body is a module
//! initializer. A private synthetic procedure symbol records every direct
//! source dependency; ordinal object names then preserve the module graph's
//! dependency-first initializer order under the version-4 linker's stable
//! name ordering.

use crate::ast::{Effect, Procedure, Program, ProgramLocations};
use crate::bytecode::{BytecodeError, BytecodeProgram, Instruction};
use crate::hir::{HirError, HirProgram, ProcedureId};
use crate::linker::{
    ObjectError, ObjectExport, ObjectImport, ObjectModule, ObjectRelocation, RelocationKind,
    SymbolKind, SymbolSignature, SymbolTarget, MAX_OBJECT_NAME_BYTES,
};
use crate::module_graph::{ModuleGraph, SourceModule};
use crate::source::{SourceError, SourceId, SourceManager};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Compile every retained source in a canonical module graph into one
/// independently serializable `OUROOBJ` v4 module.
///
/// The returned vector is dependency-first, but [`crate::linker::link`] also
/// produces the same initializer order if the caller permutes it because the
/// stable object names carry their graph ordinal.
pub fn compile_objects(graph: &ModuleGraph) -> Result<Vec<ObjectModule>, ObjectCompileError> {
    let effective_prelude = effective_prelude(graph);
    let names = object_names(graph)?;
    let anchors = anchor_names(graph);
    let modules_by_source: HashMap<SourceId, &SourceModule> = graph
        .modules()
        .iter()
        .map(|module| (module.source, module))
        .collect();

    let mut objects =
        Vec::with_capacity(graph.modules().len() + usize::from(!effective_prelude.is_empty()));
    for module in graph.modules() {
        let object = compile_source_module(
            graph,
            module,
            &modules_by_source,
            &effective_prelude,
            &names,
            &anchors,
        )?;
        objects.push(object);
    }
    if !effective_prelude.is_empty() {
        objects.push(compile_prelude(graph.modules().len(), effective_prelude)?);
    }
    Ok(objects)
}

impl ModuleGraph {
    /// Convenience method for [`compile_objects`].
    pub fn compile_objects(&self) -> Result<Vec<ObjectModule>, ObjectCompileError> {
        compile_objects(self)
    }
}

fn compile_source_module(
    graph: &ModuleGraph,
    module: &SourceModule,
    modules: &HashMap<SourceId, &SourceModule>,
    prelude: &[Procedure],
    names: &HashMap<SourceId, String>,
    anchors: &HashMap<SourceId, String>,
) -> Result<ObjectModule, ObjectCompileError> {
    let source_name = names
        .get(&module.source)
        .cloned()
        .ok_or_else(|| internal(module.source, "missing deterministic object name"))?;
    let anchor = anchors
        .get(&module.source)
        .cloned()
        .ok_or_else(|| internal(module.source, "missing module dependency anchor"))?;

    let mut program = module.located.program.clone();
    let mut locations = module.located.locations.clone();
    let local_constant_count = program.manifests.len();
    let local_temporal_count = program.temporal_declarations.len();
    let source_procedure_count = program.procedures.len();
    let local_foreign_count = program.ffi_declarations.len();

    program.procedures.push(Procedure {
        name: anchor.clone(),
        params: Vec::new(),
        returns: 0,
        effects: vec![Effect::Pure],
        body: Vec::new(),
    });
    locations.procedures.push(None);
    let local_procedure_count = program.procedures.len();

    let visible = graph
        .visible_dependencies(module.source)
        .ok_or_else(|| internal(module.source, "missing module visibility set"))?;
    for dependency in visible {
        let imported = modules
            .get(dependency)
            .ok_or_else(|| internal(module.source, "visible dependency was not retained"))?;
        for declaration in &imported.located.program.manifests {
            program.manifests.push(declaration.clone());
            locations.manifests.push(None);
        }
        for declaration in &imported.located.program.temporal_declarations {
            program.temporal_declarations.push(declaration.clone());
            locations.temporals.push(None);
        }
        for procedure in &imported.located.program.procedures {
            program.procedures.push(interface_stub(procedure));
            locations.procedures.push(None);
        }
        for foreign in &imported.located.program.ffi_declarations {
            program.ffi_declarations.push(foreign.clone());
            locations.foreigns.push(None);
        }
    }
    for procedure in prelude {
        program.procedures.push(interface_stub(procedure));
        locations.procedures.push(None);
    }

    let hir = HirProgram::resolve_located(&program, &locations).map_err(|errors| {
        ObjectCompileError::Hir {
            module: source_name.clone(),
            errors,
        }
    })?;
    let (mut code, constant_references) =
        BytecodeProgram::compile_with_symbols(&hir).map_err(|error| {
            ObjectCompileError::Bytecode {
                module: source_name.clone(),
                error,
            }
        })?;

    let mut imports = Vec::new();
    let mut seen_anchor = HashSet::new();
    for edge in graph
        .edges()
        .iter()
        .filter(|edge| edge.importer == module.source)
    {
        if !seen_anchor.insert(edge.imported) {
            continue;
        }
        imports.push(ObjectImport {
            name: anchors
                .get(&edge.imported)
                .cloned()
                .ok_or_else(|| internal(module.source, "direct dependency has no anchor"))?,
            kind: SymbolKind::Procedure,
            signature: SymbolSignature::pure(0, 0),
        });
    }

    let mut constant_imports = vec![None; hir.constants.len()];
    for constant in hir.constants.iter().skip(local_constant_count) {
        let import = import_index(imports.len(), &source_name)?;
        imports.push(ObjectImport {
            name: normalize(&constant.name),
            kind: SymbolKind::Constant,
            signature: SymbolSignature::pure(0, 1),
        });
        constant_imports[constant.id.index()] = Some(import);
    }

    let mut procedure_imports = vec![None; hir.procedures.len()];
    for procedure in hir.procedures.iter().skip(local_procedure_count) {
        let import = import_index(imports.len(), &source_name)?;
        imports.push(ObjectImport {
            name: normalize(&procedure.name),
            kind: SymbolKind::Procedure,
            signature: procedure_signature(
                procedure.params.len(),
                procedure.returns,
                &procedure.effects,
                &source_name,
            )?,
        });
        procedure_imports[procedure.id.index()] = Some(import);
    }

    let mut foreign_imports = vec![None; hir.foreigns.len()];
    for foreign in hir.foreigns.iter().skip(local_foreign_count) {
        let import = import_index(imports.len(), &source_name)?;
        let descriptor = code.foreigns.get(foreign.id.index()).ok_or_else(|| {
            internal(
                module.source,
                "compiled foreign interface is missing its descriptor",
            )
        })?;
        imports.push(ObjectImport {
            name: normalize(&foreign.declaration.signature.name),
            kind: SymbolKind::Foreign,
            signature: SymbolSignature::foreign(descriptor),
        });
        foreign_imports[foreign.id.index()] = Some(import);
    }

    // Imported procedure stubs are compiled after all source-local units.
    // Remove those unreachable ranges while retaining their out-of-table call
    // operands at typed relocation sites.
    let retained_end = code
        .procedures
        .iter()
        .take(local_procedure_count)
        .map(|entry| entry.range.end)
        .chain(code.quotations.iter().map(|entry| entry.range.end))
        .chain(std::iter::once(code.main.end))
        .max()
        .unwrap_or(code.main.end);
    code.instructions.truncate(retained_end as usize);
    code.source_map
        .retain(|entry| entry.instruction < retained_end);
    code.procedures.truncate(local_procedure_count);
    code.foreigns.truncate(local_foreign_count);

    let mut relocations = Vec::new();
    for reference in constant_references {
        if reference.instruction >= retained_end {
            continue;
        }
        if let Some(import) = constant_imports
            .get(reference.constant.index())
            .copied()
            .flatten()
        {
            relocations.push(ObjectRelocation {
                instruction: reference.instruction,
                import,
                kind: RelocationKind::ConstantLiteral,
            });
        }
    }
    for (pc, instruction) in code.instructions.iter().enumerate() {
        let (import, kind) = match instruction {
            Instruction::CallProcedure(id) => (
                procedure_imports.get(id.index()).copied().flatten(),
                RelocationKind::ProcedureCall,
            ),
            Instruction::CallForeign(id) => (
                foreign_imports.get(id.index()).copied().flatten(),
                RelocationKind::ForeignCall,
            ),
            _ => continue,
        };
        if let Some(import) = import {
            relocations.push(ObjectRelocation {
                instruction: u32::try_from(pc).map_err(|_| ObjectCompileError::IndexOverflow {
                    module: source_name.clone(),
                    what: "instruction",
                })?,
                import,
                kind,
            });
        }
    }

    let mut local_hir = hir.clone();
    local_hir.constants.truncate(local_constant_count);
    local_hir.temporals.truncate(local_temporal_count);
    local_hir.procedures.truncate(local_procedure_count);
    local_hir.foreigns.truncate(local_foreign_count);

    let mut exports =
        Vec::with_capacity(local_constant_count + source_procedure_count + local_foreign_count + 1);
    for constant in local_hir.constants.iter().take(local_constant_count) {
        exports.push(ObjectExport {
            name: normalize(&constant.name),
            target: SymbolTarget::Constant(constant.value),
            signature: SymbolSignature::pure(0, 1),
        });
    }
    for procedure in local_hir.procedures.iter().take(source_procedure_count) {
        exports.push(ObjectExport {
            name: normalize(&procedure.name),
            target: SymbolTarget::Procedure(procedure.id),
            signature: procedure_signature(
                procedure.params.len(),
                procedure.returns,
                &procedure.effects,
                &source_name,
            )?,
        });
    }
    let anchor_id = ProcedureId::try_from_index(source_procedure_count).ok_or_else(|| {
        ObjectCompileError::IndexOverflow {
            module: source_name.clone(),
            what: "procedure",
        }
    })?;
    exports.push(ObjectExport {
        name: anchor,
        target: SymbolTarget::Procedure(anchor_id),
        signature: SymbolSignature::pure(0, 0),
    });
    for foreign in local_hir.foreigns.iter().take(local_foreign_count) {
        let descriptor = code.foreigns.get(foreign.id.index()).ok_or_else(|| {
            internal(
                module.source,
                "local foreign is missing its compiled descriptor",
            )
        })?;
        exports.push(ObjectExport {
            name: normalize(&foreign.declaration.signature.name),
            target: SymbolTarget::Foreign(foreign.id),
            signature: SymbolSignature::foreign(descriptor),
        });
    }

    let file = graph
        .sources()
        .get(module.source)
        .map_err(ObjectCompileError::Source)?;
    let mut local_sources = SourceManager::new();
    let local_source = local_sources.add_virtual(file.display_name(), file.text());
    debug_assert_eq!(local_source.index(), 0);
    for entry in &mut code.source_map {
        entry.span.source = local_source;
    }

    ObjectModule::from_relocatable_compiled(
        source_name.clone(),
        &local_hir,
        code,
        &local_sources,
        exports,
        imports,
        relocations,
    )
    .map_err(|error| ObjectCompileError::Object {
        module: source_name,
        error,
    })
}

fn compile_prelude(
    ordinal: usize,
    procedures: Vec<Procedure>,
) -> Result<ObjectModule, ObjectCompileError> {
    let name = format!("{ordinal:08}-prelude");
    let mut program = Program::new();
    program.procedures = procedures;
    let locations = ProgramLocations {
        procedures: vec![None; program.procedures.len()],
        ..ProgramLocations::default()
    };
    let hir = HirProgram::resolve_located(&program, &locations).map_err(|errors| {
        ObjectCompileError::Hir {
            module: name.clone(),
            errors,
        }
    })?;
    let code = BytecodeProgram::compile(&hir).map_err(|error| ObjectCompileError::Bytecode {
        module: name.clone(),
        error,
    })?;
    let exports = hir
        .procedures
        .iter()
        .map(|procedure| {
            Ok(ObjectExport {
                name: normalize(&procedure.name),
                target: SymbolTarget::Procedure(procedure.id),
                signature: procedure_signature(
                    procedure.params.len(),
                    procedure.returns,
                    &procedure.effects,
                    &name,
                )?,
            })
        })
        .collect::<Result<Vec<_>, ObjectCompileError>>()?;
    ObjectModule::from_relocatable_compiled(
        name.clone(),
        &hir,
        code,
        &SourceManager::new(),
        exports,
        Vec::new(),
        Vec::new(),
    )
    .map_err(|error| ObjectCompileError::Object {
        module: name,
        error,
    })
}

fn interface_stub(procedure: &Procedure) -> Procedure {
    Procedure {
        name: procedure.name.clone(),
        params: procedure.params.clone(),
        returns: procedure.returns,
        effects: procedure.effects.clone(),
        body: Vec::new(),
    }
}

fn effective_prelude(graph: &ModuleGraph) -> Vec<Procedure> {
    let source_names: HashSet<String> = graph
        .modules()
        .iter()
        .flat_map(|module| {
            module
                .located
                .program
                .manifests
                .iter()
                .map(|declaration| normalize(&declaration.name))
                .chain(
                    module
                        .located
                        .program
                        .temporal_declarations
                        .iter()
                        .map(|declaration| normalize(&declaration.name)),
                )
                .chain(
                    module
                        .located
                        .program
                        .procedures
                        .iter()
                        .map(|procedure| normalize(&procedure.name)),
                )
                .chain(
                    module
                        .located
                        .program
                        .ffi_declarations
                        .iter()
                        .map(|foreign| normalize(&foreign.signature.name)),
                )
        })
        .collect();
    graph
        .prelude()
        .iter()
        .filter(|procedure| !source_names.contains(&normalize(&procedure.name)))
        .cloned()
        .collect()
}

fn object_names(graph: &ModuleGraph) -> Result<HashMap<SourceId, String>, ObjectCompileError> {
    graph
        .modules()
        .iter()
        .enumerate()
        .map(|(ordinal, module)| {
            let file = graph
                .sources()
                .get(module.source)
                .map_err(ObjectCompileError::Source)?;
            let display = file.display_name();
            let prefix = format!("{ordinal:08}-");
            let digest = format!("-{:016x}", stable_hash(display.as_bytes()));
            let available = MAX_OBJECT_NAME_BYTES.saturating_sub(prefix.len() + digest.len());
            let mut end = display.len().min(available);
            while !display.is_char_boundary(end) {
                end -= 1;
            }
            Ok((
                module.source,
                format!("{prefix}{}{digest}", &display[..end]),
            ))
        })
        .collect()
}

fn anchor_names(graph: &ModuleGraph) -> HashMap<SourceId, String> {
    let mut occupied: HashSet<String> = graph
        .modules()
        .iter()
        .flat_map(|module| {
            module
                .located
                .program
                .manifests
                .iter()
                .map(|declaration| normalize(&declaration.name))
                .chain(
                    module
                        .located
                        .program
                        .temporal_declarations
                        .iter()
                        .map(|declaration| normalize(&declaration.name)),
                )
                .chain(
                    module
                        .located
                        .program
                        .procedures
                        .iter()
                        .map(|procedure| normalize(&procedure.name)),
                )
                .chain(
                    module
                        .located
                        .program
                        .ffi_declarations
                        .iter()
                        .map(|foreign| normalize(&foreign.signature.name)),
                )
        })
        .collect();
    let mut anchors = HashMap::new();
    for (ordinal, module) in graph.modules().iter().enumerate() {
        let mut anchor = format!("__ouro_module_init_{ordinal:08}");
        while !occupied.insert(normalize(&anchor)) {
            anchor.push('_');
        }
        anchors.insert(module.source, anchor);
    }
    anchors
}

fn procedure_signature(
    inputs: usize,
    outputs: usize,
    effects: &[Effect],
    module: &str,
) -> Result<SymbolSignature, ObjectCompileError> {
    let inputs = u32::try_from(inputs).map_err(|_| ObjectCompileError::IndexOverflow {
        module: module.to_string(),
        what: "procedure input",
    })?;
    let outputs = u32::try_from(outputs).map_err(|_| ObjectCompileError::IndexOverflow {
        module: module.to_string(),
        what: "procedure output",
    })?;
    let mut bits = 0u64;
    for effect in effects {
        bits |= match effect {
            Effect::Pure => 0,
            Effect::Reads(_) => 1 << 0,
            Effect::Writes(_) => 1 << 1,
            Effect::Temporal => 1 << 2,
            Effect::IO => 1 << 3,
            Effect::Alloc => 1 << 4,
            Effect::FFI => 1 << 5,
            Effect::FileIO => 1 << 6,
            Effect::Network => 1 << 7,
            Effect::System => 1 << 8,
        };
    }
    Ok(SymbolSignature {
        inputs,
        outputs,
        effects: bits,
        abi: 0,
    })
}

fn import_index(index: usize, module: &str) -> Result<u32, ObjectCompileError> {
    u32::try_from(index).map_err(|_| ObjectCompileError::IndexOverflow {
        module: module.to_string(),
        what: "import",
    })
}

fn normalize(name: &str) -> String {
    name.to_lowercase()
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn internal(source: SourceId, message: impl Into<String>) -> ObjectCompileError {
    ObjectCompileError::Internal {
        module: source.to_string(),
        message: message.into(),
    }
}

/// Failure while lowering a canonical source graph into relocatable objects.
#[derive(Debug)]
pub enum ObjectCompileError {
    /// A retained source could no longer be read from its graph.
    Source(SourceError),
    /// Typed name/quotation resolution failed for one source-local unit.
    Hir {
        module: String,
        errors: Vec<HirError>,
    },
    /// HIR could not be represented as bounded bytecode.
    Bytecode {
        module: String,
        error: BytecodeError,
    },
    /// The constructed relocation-aware object failed validation.
    Object { module: String, error: ObjectError },
    /// A typed artifact index exceeded its stable representation.
    IndexOverflow { module: String, what: &'static str },
    /// A retained graph invariant was unexpectedly absent.
    Internal { module: String, message: String },
}

impl fmt::Display for ObjectCompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => error.fmt(formatter),
            Self::Hir { module, errors } => {
                write!(formatter, "module {module:?} HIR resolution failed")?;
                for error in errors {
                    write!(formatter, "\n  - {error}")?;
                }
                Ok(())
            }
            Self::Bytecode { module, error } => {
                write!(
                    formatter,
                    "module {module:?} bytecode lowering failed: {error}"
                )
            }
            Self::Object { module, error } => {
                write!(
                    formatter,
                    "module {module:?} object construction failed: {error}"
                )
            }
            Self::IndexOverflow { module, what } => {
                write!(formatter, "module {module:?} {what} index overflow")
            }
            Self::Internal { module, message } => {
                write!(
                    formatter,
                    "module {module:?} object compiler invariant: {message}"
                )
            }
        }
    }
}

impl Error for ObjectCompileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Bytecode { error, .. } => Some(error),
            Self::Object { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode_vm::BytecodeVm;
    use crate::core::{OutputItem, PagedMemory};
    use crate::linker::{link, link_with_metadata};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ourochronos-object-compiler-tests-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, contents: &str) -> PathBuf {
            let path = self.0.join(relative);
            fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn output(code: &BytecodeProgram) -> Vec<u64> {
        BytecodeVm::new()
            .run(code, &PagedMemory::new())
            .unwrap()
            .output
            .into_iter()
            .map(|item| match item {
                OutputItem::Val(value) => value.val,
                OutputItem::Char(value) => u64::from(value),
            })
            .collect()
    }

    #[test]
    fn per_source_objects_roundtrip_and_match_merged_initializers_and_quotes() {
        let temp = TempDir::new();
        temp.write("alpha.ouro", "[ 7 OUTPUT ] EXEC\n");
        temp.write(
            "zlib.ouro",
            "MANIFEST ANSWER = 42;\n\
             TEMPORAL answer_cell @ 0 DEFAULT 42;\n\
             PROPERTY answer { ALL_FIXED CELL answer_cell EQ 42 OR NOT CELL 1 NE 0; }\n\
             PROCEDURE exported { [ ANSWER OUTPUT ] EXEC }\n\
             ANSWER OUTPUT\n",
        );
        let root = temp.write(
            "root.ouro",
            "IMPORT \"alpha.ouro\"\nIMPORT \"zlib.ouro\"\n1 OUTPUT\nANSWER OUTPUT\nanswer_cell POP\nexported\n",
        );
        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(graph.modules().len(), 3);
        assert!(graph
            .modules()
            .iter()
            .all(|module| module.located.locations.matches(&module.located.program)));

        let merged_hir = graph.resolve_hir().unwrap();
        let merged = BytecodeProgram::compile(&merged_hir).unwrap();
        let objects = compile_objects(&graph).unwrap();
        assert_eq!(objects.len(), 3);
        assert!(objects.windows(2).all(|pair| pair[0].name < pair[1].name));
        assert!(objects
            .iter()
            .all(|object| object.metadata.source_files.len() == 1));

        let provider = objects
            .iter()
            .find(|object| {
                object
                    .exports
                    .iter()
                    .any(|export| export.name == "exported")
            })
            .unwrap();
        assert!(provider.metadata.constants.contains(&42));
        assert_eq!(provider.metadata.properties.len(), 1);
        assert_eq!(provider.metadata.named_temporals.len(), 1);
        assert_eq!(provider.metadata.named_temporals[0].default, 42);
        assert_eq!(
            provider.metadata.properties[0].touched_addresses,
            vec![0, 1]
        );
        assert_eq!(provider.code.quotations.len(), 1);

        let consumer = objects
            .iter()
            .find(|object| {
                object
                    .imports
                    .iter()
                    .any(|import| import.name == "exported")
            })
            .unwrap();
        assert!(consumer.relocations.iter().any(|relocation| {
            relocation.kind == RelocationKind::ProcedureCall
                && consumer.imports[relocation.import as usize].name == "exported"
        }));
        assert!(consumer.relocations.iter().any(|relocation| {
            relocation.kind == RelocationKind::ConstantLiteral
                && consumer.imports[relocation.import as usize].name == "answer"
        }));

        let mut decoded = objects
            .iter()
            .map(|object| ObjectModule::from_bytes(&object.to_bytes().unwrap()).unwrap())
            .collect::<Vec<_>>();
        decoded.reverse();
        let linked = link_with_metadata(&decoded).unwrap();
        assert_eq!(linked.metadata.properties.len(), 1);
        assert_eq!(linked.metadata.named_temporals.len(), 1);
        assert_eq!(linked.metadata.properties[0].touched_addresses, vec![0, 1]);
        assert_eq!(
            linked.metadata.property_named_slice("answer").unwrap(),
            linked.metadata.named_temporals
        );
        assert_eq!(output(&linked.code), vec![7, 42, 1, 42, 42]);
        assert_eq!(output(&linked.code), output(&merged));

        // alpha owns linked quote 0, so zlib's private local quote 0 must have
        // been relocated when its exported procedure was linked.
        assert!(linked.code.procedures.iter().any(|procedure| {
            linked.code.instructions
                [procedure.range.start as usize..procedure.range.end as usize]
                .iter()
                .any(|instruction| {
                    matches!(instruction, Instruction::PushQuote(id) if id.as_u64() == 1)
                })
        }));
    }

    #[test]
    fn imported_foreign_uses_exact_descriptor_and_typed_call_relocation() {
        let temp = TempDir::new();
        temp.write(
            "foreign.ouro",
            "FOREIGN \"process\" {\n\
             PROC add_signed AS \"host_add\" (a: u64, b: i64 -- i64) PURE;\n\
             }\n",
        );
        let root = temp.write(
            "root.ouro",
            "IMPORT \"foreign.ouro\"\n50 8 add_signed POP\n",
        );
        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        let objects = compile_objects(&graph).unwrap();
        let provider = objects
            .iter()
            .find(|object| object.code.foreigns.len() == 1)
            .unwrap();
        let consumer = objects
            .iter()
            .find(|object| {
                object
                    .relocations
                    .iter()
                    .any(|relocation| relocation.kind == RelocationKind::ForeignCall)
            })
            .unwrap();
        let export = provider
            .exports
            .iter()
            .find(|export| export.name == "add_signed")
            .unwrap();
        let import = consumer
            .imports
            .iter()
            .find(|import| import.name == "add_signed")
            .unwrap();
        assert_eq!(export.signature, import.signature);
        assert_ne!(export.signature.abi, 0);
        let linked = link(&objects).unwrap();
        assert_eq!(linked.foreigns, provider.code.foreigns);
    }

    #[test]
    fn unresolved_link_errors_retain_the_importing_source_name() {
        let temp = TempDir::new();
        temp.write("library.ouro", "PROCEDURE answer { 42 }\n");
        let root = temp.write("root.ouro", "IMPORT \"library.ouro\"\nanswer POP\n");
        let graph = ModuleGraph::load(&root, Vec::new()).unwrap();
        let mut objects = compile_objects(&graph).unwrap();
        let consumer = objects
            .iter_mut()
            .find(|object| object.name.contains("root.ouro"))
            .unwrap();
        let imported = consumer
            .imports
            .iter_mut()
            .find(|import| import.name == "answer")
            .unwrap();
        imported.name = "missing_answer".to_string();
        let error = link(&objects).unwrap_err().to_string();
        assert!(error.contains("root.ouro"), "{error}");
        assert!(error.contains("missing_answer"), "{error}");
    }
}
