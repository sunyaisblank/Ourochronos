//! Canonical, importer-relative source loading and deterministic module linking.
//!
//! Parsing a module never performs filesystem I/O. The graph owns loading,
//! canonical deduplication, cycle reporting, dependency order, and the single
//! relocation pass that joins module-local quotation tables.

use crate::ast::{LocatedProgram, Procedure, Program, ProgramLocations, Stmt};
use crate::hir::{HirError, HirProgram};
use crate::lexer::{lex_with_token_limit, LocatedToken, MAX_SOURCE_TOKENS};
use crate::parser::{Parser, Token, MAX_EXPANDED_STATEMENTS};
use crate::source::{SourceError, SourceId, SourceManager, SourceSpan};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::path::Path;

/// Maximum number of distinct source modules in one graph.
pub const MAX_MODULES: usize = 256;
/// Maximum number of import declarations accepted from one module.
pub const MAX_IMPORTS_PER_MODULE: usize = 128;
/// Maximum number of active modules in an import chain, including the root.
pub const MAX_IMPORT_DEPTH: usize = 64;
/// Maximum number of import edges retained across one graph.
pub const MAX_IMPORT_EDGES: usize = 4_096;
/// Maximum combined UTF-8 source bytes retained by one module graph.
pub const MAX_GRAPH_SOURCE_BYTES: usize = 64 * 1024 * 1024;
/// Maximum aggregate tokens retained or processed across a module graph.
pub const MAX_GRAPH_SOURCE_TOKENS: usize = MAX_SOURCE_TOKENS;
/// Maximum aggregate statement nodes materialized across all source modules.
pub const MAX_GRAPH_EXPANDED_STATEMENTS: usize = MAX_EXPANDED_STATEMENTS;

/// One resolved import edge in source order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportEdge {
    pub importer: SourceId,
    pub imported: SourceId,
    pub requested: String,
    pub span: SourceSpan,
}

/// One parsed source module retained independently of the linked compatibility
/// program. Entries in [`ModuleGraph::modules`] are dependency-first and each
/// quotation table remains local to its source.
#[derive(Debug, Clone)]
pub struct SourceModule {
    /// Identity in the graph's [`SourceManager`].
    pub source: SourceId,
    /// Exact source-local AST and location sidecar.
    pub located: LocatedProgram,
}

/// A fully loaded and linked source graph.
#[derive(Debug)]
pub struct ModuleGraph {
    sources: SourceManager,
    root: SourceId,
    order: Vec<SourceId>,
    edges: Vec<ImportEdge>,
    modules: Vec<SourceModule>,
    visibility: HashMap<SourceId, Vec<SourceId>>,
    prelude: Vec<Procedure>,
    program: Program,
    locations: ProgramLocations,
}

impl ModuleGraph {
    /// Load `root`, resolve its complete import graph, and link every module
    /// initializer and declaration exactly once. `prelude` supplies callable
    /// signatures during parsing and definitions after source modules link.
    pub fn load(root: impl AsRef<Path>, prelude: Vec<Procedure>) -> Result<Self, ModuleError> {
        let mut builder = GraphBuilder::new(prelude);
        let root = builder
            .sources
            .load_file(root)
            .map_err(ModuleError::Source)?;
        Self::finish_load(builder, root)
    }

    /// Load a canonical disk-backed graph while replacing only the root
    /// source's text with an editor overlay. Dependency files are read through
    /// the ordinary importer-relative loader.
    pub fn load_with_root_text(
        root: impl AsRef<Path>,
        root_text: impl Into<String>,
        prelude: Vec<Procedure>,
    ) -> Result<Self, ModuleError> {
        let mut builder = GraphBuilder::new(prelude);
        let root = builder
            .sources
            .load_file_overlay(root, root_text)
            .map_err(ModuleError::Source)?;
        Self::finish_load(builder, root)
    }

    fn finish_load(mut builder: GraphBuilder, root: SourceId) -> Result<Self, ModuleError> {
        builder.visit(root)?;
        let modules =
            builder
                .order
                .iter()
                .map(|source| {
                    let program = builder.programs.get(source).cloned().ok_or_else(|| {
                        ModuleError::Internal {
                            message: format!("missing parsed module {source}"),
                        }
                    })?;
                    let locations = builder.locations.get(source).cloned().ok_or_else(|| {
                        ModuleError::Internal {
                            message: format!("missing parsed locations for module {source}"),
                        }
                    })?;
                    Ok(SourceModule {
                        source: *source,
                        located: LocatedProgram { program, locations },
                    })
                })
                .collect::<Result<Vec<_>, ModuleError>>()?;
        let visibility = builder.visibility.clone();
        let prelude = builder.prelude.clone();
        let (program, locations) = link_programs(
            &builder.sources,
            &builder.order,
            builder.programs,
            builder.locations,
            builder.prelude,
        )?;
        Ok(Self {
            sources: builder.sources,
            root,
            order: builder.order,
            edges: builder.edges,
            modules,
            visibility,
            prelude,
            program,
            locations,
        })
    }

    pub const fn root(&self) -> SourceId {
        self.root
    }

    /// Deterministic dependency-first order. The root is last.
    pub fn order(&self) -> &[SourceId] {
        &self.order
    }

    pub fn edges(&self) -> &[ImportEdge] {
        &self.edges
    }

    /// Parsed source-local modules in the same dependency-first order as
    /// [`Self::order`]. No declarations or quotation identities are merged in
    /// this view.
    pub fn modules(&self) -> &[SourceModule] {
        &self.modules
    }

    /// Transitive modules whose public interfaces were visible while parsing
    /// `source`, in deterministic dependency-first order.
    pub fn visible_dependencies(&self, source: SourceId) -> Option<&[SourceId]> {
        self.visibility.get(&source).map(Vec::as_slice)
    }

    /// Procedure definitions supplied outside source files.
    pub fn prelude(&self) -> &[Procedure] {
        &self.prelude
    }

    pub fn sources(&self) -> &SourceManager {
        &self.sources
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Exact locations parallel to [`ModuleGraph::program`].
    pub fn locations(&self) -> &ProgramLocations {
        &self.locations
    }

    /// Resolve the linked graph without dropping file-backed locations.
    pub fn resolve_hir(&self) -> Result<HirProgram, Vec<HirError>> {
        HirProgram::resolve_located(&self.program, &self.locations)
    }

    pub fn into_program(self) -> Program {
        self.program
    }

    /// Consume the graph without discarding its source-location sidecar.
    pub fn into_located_program(self) -> LocatedProgram {
        LocatedProgram {
            program: self.program,
            locations: self.locations,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Done,
}

struct GraphBuilder {
    sources: SourceManager,
    prelude: Vec<Procedure>,
    states: HashMap<SourceId, VisitState>,
    stack: Vec<SourceId>,
    order: Vec<SourceId>,
    edges: Vec<ImportEdge>,
    programs: HashMap<SourceId, Program>,
    locations: HashMap<SourceId, ProgramLocations>,
    /// Transitive dependency visibility for each completed module, excluding
    /// the module itself and ordered dependency-first.
    visibility: HashMap<SourceId, Vec<SourceId>>,
    limits: GraphLimits,
    source_bytes: usize,
    source_tokens: usize,
    expanded_statements: usize,
}

#[derive(Debug, Clone, Copy)]
struct GraphLimits {
    modules: usize,
    imports_per_module: usize,
    depth: usize,
    edges: usize,
    source_bytes: usize,
    source_tokens: usize,
    expanded_statements: usize,
}

impl Default for GraphLimits {
    fn default() -> Self {
        Self {
            modules: MAX_MODULES,
            imports_per_module: MAX_IMPORTS_PER_MODULE,
            depth: MAX_IMPORT_DEPTH,
            edges: MAX_IMPORT_EDGES,
            source_bytes: MAX_GRAPH_SOURCE_BYTES,
            source_tokens: MAX_GRAPH_SOURCE_TOKENS,
            expanded_statements: MAX_GRAPH_EXPANDED_STATEMENTS,
        }
    }
}

impl GraphBuilder {
    fn new(prelude: Vec<Procedure>) -> Self {
        Self::new_with_limits(prelude, GraphLimits::default())
    }

    fn new_with_limits(prelude: Vec<Procedure>, limits: GraphLimits) -> Self {
        Self {
            sources: SourceManager::new(),
            prelude,
            states: HashMap::new(),
            stack: Vec::new(),
            order: Vec::new(),
            edges: Vec::new(),
            programs: HashMap::new(),
            locations: HashMap::new(),
            visibility: HashMap::new(),
            limits,
            source_bytes: 0,
            source_tokens: 0,
            expanded_statements: 0,
        }
    }

    fn visit(&mut self, source: SourceId) -> Result<(), ModuleError> {
        match self.states.get(&source) {
            Some(VisitState::Done) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(self.cycle_error(source, None));
            }
            None => {}
        }

        let file_bytes = self
            .sources
            .get(source)
            .map_err(ModuleError::Source)?
            .text()
            .len();
        let observed = self.source_bytes.saturating_add(file_bytes);
        if observed > self.limits.source_bytes {
            return Err(ModuleError::GraphSourceBytesExceeded {
                limit: self.limits.source_bytes,
                observed,
            });
        }
        self.source_bytes = observed;

        self.states.insert(source, VisitState::Visiting);
        self.stack.push(source);

        let remaining_tokens = self.limits.source_tokens.saturating_sub(self.source_tokens);
        let located = {
            let file = self.sources.get(source).map_err(ModuleError::Source)?;
            lex_with_token_limit(source, file.text(), remaining_tokens).map_err(|errors| {
                ModuleError::Lex {
                    source: file.display_name().to_string(),
                    errors: errors.into_iter().map(|error| error.to_string()).collect(),
                }
            })?
        };
        self.source_tokens = self
            .source_tokens
            .checked_add(located.len())
            .ok_or_else(|| ModuleError::Internal {
                message: "module-graph token count overflow".to_string(),
            })?;
        let imports = discover_imports(&located);
        if imports.len() > self.limits.imports_per_module {
            return Err(ModuleError::TooManyImports {
                source: self.source_name(source),
                limit: self.limits.imports_per_module,
                observed: imports.len(),
            });
        }
        let mut direct_dependencies = Vec::with_capacity(imports.len());
        for (requested, span) in &imports {
            if self.edges.len() >= self.limits.edges {
                return Err(ModuleError::TooManyImportEdges {
                    limit: self.limits.edges,
                });
            }
            let canonical_path = self
                .sources
                .resolve_import_path(source, requested)
                .map_err(ModuleError::Source)?;
            if self.sources.canonical_source_id(&canonical_path).is_none()
                && self.sources.len() >= self.limits.modules
            {
                return Err(ModuleError::TooManyModules {
                    limit: self.limits.modules,
                });
            }
            let imported = self
                .sources
                .load_canonical_file(canonical_path)
                .map_err(ModuleError::Source)?;
            self.edges.push(ImportEdge {
                importer: source,
                imported,
                requested: requested.clone(),
                span: *span,
            });
            if self.states.get(&imported) == Some(&VisitState::Visiting) {
                return Err(self.cycle_error(imported, Some(*span)));
            }
            if !self.states.contains_key(&imported) && self.stack.len() >= self.limits.depth {
                return Err(ModuleError::ImportDepthExceeded {
                    source: self.source_name(imported),
                    limit: self.limits.depth,
                });
            }
            self.visit(imported)?;
            direct_dependencies.push(imported);
        }

        let mut visible = Vec::new();
        let mut seen = HashSet::new();
        for dependency in direct_dependencies {
            if let Some(transitive) = self.visibility.get(&dependency) {
                for &module in transitive {
                    if seen.insert(module) {
                        visible.push(module);
                    }
                }
            }
            if seen.insert(dependency) {
                visible.push(dependency);
            }
        }

        let tokens: Vec<Token> = located.iter().map(|item| item.token.clone()).collect();
        let spans: Vec<SourceSpan> = located.iter().map(|item| item.span).collect();
        let remaining_statements = self
            .limits
            .expanded_statements
            .saturating_sub(self.expanded_statements);
        let mut parser = Parser::new_with_deferred_imports_spans_and_statement_limit(
            &tokens,
            &spans,
            remaining_statements,
        )
        .map_err(|message| ModuleError::Internal {
            message: format!("could not construct located parser: {message}"),
        })?;
        parser.register_procedure_interfaces(&self.prelude);
        for dependency in &visible {
            let program = self
                .programs
                .get(dependency)
                .expect("completed visible module has a parsed program");
            parser.register_imported_interface(program);
        }
        let located_program =
            parser
                .parse_located_program()
                .map_err(|message| ModuleError::LocatedParse {
                    source: self.source_name(source),
                    message,
                    span: parser.error_source_span(),
                })?;
        self.expanded_statements = self
            .expanded_statements
            .checked_add(parser.expanded_statement_count())
            .ok_or_else(|| ModuleError::Internal {
                message: "module-graph expanded statement count overflow".to_string(),
            })?;
        let parsed_imports = parser.take_deferred_imports();
        let discovered_imports: Vec<String> = imports.into_iter().map(|(path, _)| path).collect();
        if parsed_imports != discovered_imports {
            return Err(ModuleError::Internal {
                message: format!(
                    "import discovery disagreed with parser for {}",
                    self.source_name(source)
                ),
            });
        }

        self.programs.insert(source, located_program.program);
        self.locations.insert(source, located_program.locations);
        self.visibility.insert(source, visible);
        self.states.insert(source, VisitState::Done);
        self.order.push(source);
        let popped = self.stack.pop();
        debug_assert_eq!(popped, Some(source));
        Ok(())
    }

    fn source_name(&self, source: SourceId) -> String {
        self.sources
            .get(source)
            .map(|file| file.display_name().to_string())
            .unwrap_or_else(|_| source.to_string())
    }

    fn cycle_error(&self, repeated: SourceId, span: Option<SourceSpan>) -> ModuleError {
        let start = self
            .stack
            .iter()
            .position(|source| *source == repeated)
            .unwrap_or(0);
        let mut chain: Vec<String> = self.stack[start..]
            .iter()
            .map(|source| self.source_name(*source))
            .collect();
        chain.push(self.source_name(repeated));
        ModuleError::ImportCycle { chain, span }
    }
}

fn discover_imports(tokens: &[LocatedToken]) -> Vec<(String, SourceSpan)> {
    let mut imports = Vec::new();
    let mut index = 0;
    while index + 1 < tokens.len() {
        if matches!(&tokens[index].token, Token::Word(word) if word.eq_ignore_ascii_case("IMPORT"))
        {
            if let Token::StringLit(path) = &tokens[index + 1].token {
                imports.push((path.clone(), tokens[index + 1].span));
                index += 2;
                continue;
            }
        }
        index += 1;
    }
    imports
}

fn link_programs(
    sources: &SourceManager,
    order: &[SourceId],
    mut modules: HashMap<SourceId, Program>,
    mut module_locations: HashMap<SourceId, ProgramLocations>,
    prelude: Vec<Procedure>,
) -> Result<(Program, ProgramLocations), ModuleError> {
    let mut linked = Program::new();
    let mut linked_locations = ProgramLocations::default();
    let mut symbols: HashMap<String, (String, Option<SourceSpan>)> = HashMap::new();
    let mut properties: HashMap<String, (String, Option<SourceSpan>)> = HashMap::new();
    let mut temporal_addresses: HashMap<u64, (String, String, Option<SourceSpan>)> = HashMap::new();
    let mut singleton_owners: HashMap<&'static str, String> = HashMap::new();

    for &source in order {
        let source_name = sources
            .get(source)
            .map_err(ModuleError::Source)?
            .display_name()
            .to_string();
        let mut module = modules
            .remove(&source)
            .ok_or_else(|| ModuleError::Internal {
                message: format!("missing parsed module {source}"),
            })?;
        let locations = module_locations
            .remove(&source)
            .ok_or_else(|| ModuleError::Internal {
                message: format!("missing parsed locations for module {source}"),
            })?;
        if !locations.matches(&module) {
            return Err(ModuleError::Internal {
                message: format!("location sidecar does not match module {source}"),
            });
        }

        for (declaration, location) in module.manifests.into_iter().zip(locations.manifests) {
            let symbol = declaration.name.to_lowercase();
            register_symbol(&mut symbols, &symbol, &source_name, location)?;
            linked.manifests.push(declaration);
            linked_locations.manifests.push(location);
        }

        for (declaration, location) in module
            .temporal_declarations
            .into_iter()
            .zip(locations.temporals)
        {
            let symbol = declaration.name.to_lowercase();
            register_symbol(&mut symbols, &symbol, &source_name, location)?;
            if let Some((first_name, first_source, first_span)) = temporal_addresses.insert(
                declaration.address,
                (symbol.clone(), source_name.clone(), location),
            ) {
                return Err(ModuleError::DuplicateLocatedSymbol {
                    symbol: format!(
                        "TEMPORAL address {} ({first_name} and {symbol})",
                        declaration.address
                    ),
                    first: first_source,
                    second: source_name.clone(),
                    spans: Box::new((first_span, location)),
                });
            }
            linked.temporal_declarations.push(declaration);
            linked_locations.temporals.push(location);
        }

        let quote_offset = linked.quotes.len();
        relocate_quote_ids(&mut module.body, quote_offset)?;
        for procedure in &mut module.procedures {
            relocate_quote_ids(&mut procedure.body, quote_offset)?;
        }
        for quote in &mut module.quotes {
            relocate_quote_ids(quote, quote_offset)?;
        }

        for (procedure, location) in module.procedures.into_iter().zip(locations.procedures) {
            let symbol = procedure.name.to_lowercase();
            register_symbol(
                &mut symbols,
                &symbol,
                &source_name,
                location.as_ref().map(|locations| locations.declaration),
            )?;
            linked.procedures.push(procedure);
            linked_locations.procedures.push(location);
        }
        for (declaration, location) in module.ffi_declarations.into_iter().zip(locations.foreigns) {
            let symbol = declaration.signature.name.to_lowercase();
            register_symbol(&mut symbols, &symbol, &source_name, location)?;
            linked.ffi_declarations.push(declaration);
            linked_locations.foreigns.push(location);
        }
        for (property, location) in module
            .temporal_properties
            .into_iter()
            .zip(locations.properties)
        {
            let name = property.name.to_lowercase();
            if let Some((first, first_span)) =
                properties.insert(name.clone(), (source_name.clone(), location))
            {
                return Err(ModuleError::DuplicateLocatedSymbol {
                    symbol: format!("PROPERTY {name}"),
                    first,
                    second: source_name.clone(),
                    spans: Box::new((first_span, location)),
                });
            }
            linked.temporal_properties.push(property);
            linked_locations.properties.push(location);
        }
        merge_singleton(
            "FAMILY",
            &mut linked.family_declaration,
            module.family_declaration,
            &source_name,
            &mut singleton_owners,
        )?;
        merge_singleton(
            "MARKOV",
            &mut linked.markov_declaration,
            module.markov_declaration,
            &source_name,
            &mut singleton_owners,
        )?;
        merge_singleton(
            "QCHANNEL",
            &mut linked.quantum_declaration,
            module.quantum_declaration,
            &source_name,
            &mut singleton_owners,
        )?;
        linked.quotes.extend(module.quotes);
        linked_locations.quotes.extend(locations.quotes);
        linked.body.extend(module.body);
        linked_locations.body.extend(locations.body);
    }

    // Source definitions intentionally shadow the prelude, matching the
    // parser's historical local-definition behavior while avoiding duplicate
    // linked definitions. Prelude-prelude duplicates are rejected.
    let mut prelude_seen = HashSet::new();
    for mut procedure in prelude {
        procedure.name = procedure.name.to_lowercase();
        if symbols.contains_key(&procedure.name) {
            continue;
        }
        if !prelude_seen.insert(procedure.name.clone()) {
            return Err(ModuleError::DuplicateSymbol {
                symbol: procedure.name,
                first: "prelude".to_string(),
                second: "prelude".to_string(),
            });
        }
        linked.procedures.push(procedure);
        linked_locations.procedures.push(None);
    }

    debug_assert!(linked_locations.matches(&linked));
    Ok((linked, linked_locations))
}

fn register_symbol(
    symbols: &mut HashMap<String, (String, Option<SourceSpan>)>,
    symbol: &str,
    source: &str,
    span: Option<SourceSpan>,
) -> Result<(), ModuleError> {
    if let Some((first, first_span)) =
        symbols.insert(symbol.to_string(), (source.to_string(), span))
    {
        return Err(ModuleError::DuplicateLocatedSymbol {
            symbol: symbol.to_string(),
            first,
            second: source.to_string(),
            spans: Box::new((first_span, span)),
        });
    }
    Ok(())
}

fn merge_singleton<T>(
    kind: &'static str,
    target: &mut Option<T>,
    incoming: Option<T>,
    source: &str,
    owners: &mut HashMap<&'static str, String>,
) -> Result<(), ModuleError> {
    let Some(value) = incoming else {
        return Ok(());
    };
    if target.is_some() {
        return Err(ModuleError::DuplicateSingleton {
            kind,
            first: owners
                .get(kind)
                .cloned()
                .unwrap_or_else(|| "unknown source".to_string()),
            second: source.to_string(),
        });
    }
    *target = Some(value);
    owners.insert(kind, source.to_string());
    Ok(())
}

fn relocate_quote_ids(statements: &mut [Stmt], offset: usize) -> Result<(), ModuleError> {
    for statement in statements {
        match statement {
            Stmt::PushQuote(id) => {
                *id = id.checked_add(offset).ok_or(ModuleError::QuoteIdOverflow {
                    id: id.as_u64(),
                    offset,
                })?;
            }
            Stmt::If {
                then_branch,
                else_branch,
            } => {
                relocate_quote_ids(then_branch, offset)?;
                if let Some(branch) = else_branch {
                    relocate_quote_ids(branch, offset)?;
                }
            }
            Stmt::While { cond, body } => {
                relocate_quote_ids(cond, offset)?;
                relocate_quote_ids(body, offset)?;
            }
            Stmt::Block(body) | Stmt::TemporalScope { body, .. } => {
                relocate_quote_ids(body, offset)?;
            }
            Stmt::Op(_)
            | Stmt::Push(_)
            | Stmt::PushConstant { .. }
            | Stmt::ReadTemporal { .. }
            | Stmt::Call { .. } => {}
        }
    }
    Ok(())
}

/// Source/module loading and linking failure.
#[derive(Debug)]
pub enum ModuleError {
    Source(SourceError),
    TooManyModules {
        limit: usize,
    },
    TooManyImports {
        source: String,
        limit: usize,
        observed: usize,
    },
    TooManyImportEdges {
        limit: usize,
    },
    GraphSourceBytesExceeded {
        limit: usize,
        observed: usize,
    },
    ImportDepthExceeded {
        source: String,
        limit: usize,
    },
    Lex {
        source: String,
        errors: Vec<String>,
    },
    Parse {
        source: String,
        message: String,
    },
    LocatedParse {
        source: String,
        message: String,
        span: Option<SourceSpan>,
    },
    ImportCycle {
        chain: Vec<String>,
        span: Option<SourceSpan>,
    },
    DuplicateSymbol {
        symbol: String,
        first: String,
        second: String,
    },
    DuplicateLocatedSymbol {
        symbol: String,
        first: String,
        second: String,
        spans: Box<(Option<SourceSpan>, Option<SourceSpan>)>,
    },
    DuplicateSingleton {
        kind: &'static str,
        first: String,
        second: String,
    },
    QuoteIdOverflow {
        id: u64,
        offset: usize,
    },
    Internal {
        message: String,
    },
}

impl fmt::Display for ModuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => error.fmt(formatter),
            Self::TooManyModules { limit } => {
                write!(formatter, "module graph exceeds the {limit}-module limit")
            }
            Self::TooManyImports {
                source,
                limit,
                observed,
            } => write!(
                formatter,
                "module {source} has {observed} imports, exceeding the {limit}-import per-module limit"
            ),
            Self::TooManyImportEdges { limit } => {
                write!(formatter, "module graph exceeds the {limit}-import edge limit")
            }
            Self::GraphSourceBytesExceeded { limit, observed } => write!(
                formatter,
                "module graph retains {observed} source bytes, exceeding the {limit}-byte graph limit"
            ),
            Self::ImportDepthExceeded { source, limit } => write!(
                formatter,
                "importing {source} would exceed the {limit}-module import-depth limit"
            ),
            Self::Lex { source, errors } => {
                write!(formatter, "lexing {source} failed")?;
                for error in errors {
                    write!(formatter, "\n  - {error}")?;
                }
                Ok(())
            }
            Self::Parse { source, message } => {
                write!(formatter, "failed to parse {source}: {message}")
            }
            Self::LocatedParse {
                source,
                message,
                span,
            } => {
                write!(formatter, "failed to parse {source}: {message}")?;
                if let Some(span) = span {
                    write!(
                        formatter,
                        " (at source {} bytes {}..{})",
                        span.source.index(),
                        span.range.start,
                        span.range.end
                    )?;
                }
                Ok(())
            }
            Self::ImportCycle { chain, span } => {
                write!(formatter, "import cycle: {}", chain.join(" -> "))?;
                if let Some(span) = span {
                    write!(
                        formatter,
                        " (at source {} bytes {}..{})",
                        span.source.index(),
                        span.range.start,
                        span.range.end
                    )?;
                }
                Ok(())
            }
            Self::DuplicateSymbol {
                symbol,
                first,
                second,
            } => write!(
                formatter,
                "duplicate linked symbol '{symbol}' in {first} and {second}"
            ),
            Self::DuplicateLocatedSymbol {
                symbol,
                first,
                second,
                spans,
            } => {
                write!(
                    formatter,
                    "duplicate linked symbol '{symbol}' in {first} and {second}"
                )?;
                if let Some(span) = spans.0 {
                    write!(
                        formatter,
                        " (first at source {} bytes {}..{})",
                        span.source.index(),
                        span.range.start,
                        span.range.end
                    )?;
                }
                if let Some(span) = spans.1 {
                    write!(
                        formatter,
                        " (second at source {} bytes {}..{})",
                        span.source.index(),
                        span.range.start,
                        span.range.end
                    )?;
                }
                Ok(())
            }
            Self::DuplicateSingleton {
                kind,
                first,
                second,
            } => write!(
                formatter,
                "multiple {kind} declarations in {first} and {second}"
            ),
            Self::QuoteIdOverflow { id, offset } => write!(
                formatter,
                "quotation ID {id} overflows while applying module offset {offset}"
            ),
            Self::Internal { message } => formatter.write_str(message),
        }
    }
}

impl Error for ModuleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{BytecodeProgram, Instruction};
    use crate::core::{Memory, OutputItem};
    use crate::vm::{EpochStatus, Executor};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ourochronos-module-tests-{}-{sequence}",
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

    fn values(program: Program) -> Vec<u64> {
        let program = program.inline_procedures();
        let mut executor = Executor::new();
        executor.config.immediate_output = false;
        let result = executor.run_epoch(&program, &Memory::new());
        assert_eq!(result.status, EpochStatus::Finished);
        result
            .output
            .into_iter()
            .map(|item| match item {
                OutputItem::Val(value) => value.val,
                OutputItem::Char(value) => value as u64,
            })
            .collect()
    }

    fn load_with_limits(root: &Path, limits: GraphLimits) -> Result<ModuleGraph, ModuleError> {
        let mut builder = GraphBuilder::new_with_limits(Vec::new(), limits);
        let root = builder
            .sources
            .load_file(root)
            .map_err(ModuleError::Source)?;
        ModuleGraph::finish_load(builder, root)
    }

    #[test]
    fn diamond_imports_are_relative_deduplicated_and_dependency_first() {
        let temp = TempDir::new();
        temp.write("lib/shared.ouro", "7 OUTPUT\n");
        temp.write("left/a.ouro", "IMPORT \"../lib/shared.ouro\"\n1 OUTPUT\n");
        temp.write(
            "right/b.ouro",
            "IMPORT \"../lib/./shared.ouro\"\n2 OUTPUT\n",
        );
        let root = temp.write(
            "root.ouro",
            "IMPORT \"left/a.ouro\"\nIMPORT \"right/b.ouro\"\n3 OUTPUT\n",
        );

        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(graph.order().len(), 4);
        assert_eq!(graph.edges().len(), 4);
        assert_eq!(values(graph.into_program()), vec![7, 1, 2, 3]);
    }

    #[test]
    fn imported_procedures_and_nested_quotes_link_with_typed_relocation() {
        let temp = TempDir::new();
        temp.write(
            "lib.ouro",
            "PROCEDURE answer { [ [ 42 OUTPUT ] EXEC ] EXEC 0 OUTPUT }\n",
        );
        let root = temp.write(
            "root.ouro",
            "[ 9 OUTPUT ] EXEC\nIMPORT \"lib.ouro\"\nanswer\n",
        );

        // IMPORT is declaration syntax and must precede executable body.
        fs::write(&root, "IMPORT \"lib.ouro\"\n[ 9 OUTPUT ] EXEC\nanswer\n").unwrap();
        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(values(graph.into_program()), vec![9, 42, 0]);
    }

    #[test]
    fn cycles_report_the_complete_canonical_trace() {
        let temp = TempDir::new();
        let a = temp.write("a.ouro", "IMPORT \"b.ouro\"\n");
        temp.write("b.ouro", "IMPORT \"c.ouro\"\n");
        temp.write("c.ouro", "IMPORT \"a.ouro\"\n");

        let error = ModuleGraph::load(a, Vec::new()).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("import cycle"), "{text}");
        assert!(text.contains("a.ouro"), "{text}");
        assert!(text.contains("b.ouro"), "{text}");
        assert!(text.contains("c.ouro"), "{text}");
    }

    #[test]
    fn import_depth_is_rejected_before_the_next_recursive_visit() {
        let temp = TempDir::new();
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\n");
        temp.write("a.ouro", "IMPORT \"b.ouro\"\n");
        temp.write("b.ouro", "IMPORT \"c.ouro\"\n");
        let c = temp.write("c.ouro", "0 POP\n");
        let limits = GraphLimits {
            depth: 3,
            ..GraphLimits::default()
        };

        let error = load_with_limits(&root, limits).unwrap_err();
        assert!(matches!(
            error,
            ModuleError::ImportDepthExceeded { source, limit: 3 }
                if source == fs::canonicalize(c).unwrap().display().to_string()
        ));
    }

    #[test]
    fn per_module_import_fanout_is_bounded_before_loading_targets() {
        let temp = TempDir::new();
        let root = temp.write(
            "root.ouro",
            "IMPORT \"missing-a.ouro\"\nIMPORT \"missing-b.ouro\"\nIMPORT \"missing-c.ouro\"\n",
        );
        let limits = GraphLimits {
            imports_per_module: 2,
            ..GraphLimits::default()
        };

        let error = load_with_limits(&root, limits).unwrap_err();
        assert!(matches!(
            error,
            ModuleError::TooManyImports {
                limit: 2,
                observed: 3,
                ..
            }
        ));
    }

    #[test]
    fn distinct_module_count_is_bounded_before_reading_an_extra_module() {
        let temp = TempDir::new();
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\nIMPORT \"b.ouro\"\n");
        temp.write("a.ouro", "1 POP\n");
        temp.write("b.ouro", "2 POP\n");
        let limits = GraphLimits {
            modules: 2,
            ..GraphLimits::default()
        };

        assert!(matches!(
            load_with_limits(&root, limits),
            Err(ModuleError::TooManyModules { limit: 2 })
        ));
    }

    #[test]
    fn total_import_edge_count_is_bounded_deterministically() {
        let temp = TempDir::new();
        let root = temp.write(
            "root.ouro",
            "IMPORT \"a.ouro\"\nIMPORT \"a.ouro\"\nIMPORT \"a.ouro\"\n",
        );
        temp.write("a.ouro", "0 POP\n");
        let limits = GraphLimits {
            edges: 2,
            ..GraphLimits::default()
        };

        assert!(matches!(
            load_with_limits(&root, limits),
            Err(ModuleError::TooManyImportEdges { limit: 2 })
        ));
    }

    #[test]
    fn aggregate_graph_source_bytes_are_bounded_with_small_fixtures() {
        let temp = TempDir::new();
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\n");
        temp.write("a.ouro", "123456789 POP\n");
        let root_bytes = fs::read(&root).unwrap().len();
        let limits = GraphLimits {
            source_bytes: root_bytes + 4,
            ..GraphLimits::default()
        };

        assert!(matches!(
            load_with_limits(&root, limits),
            Err(ModuleError::GraphSourceBytesExceeded { limit, observed })
                if limit == root_bytes + 4 && observed > limit
        ));
    }

    #[test]
    fn aggregate_graph_tokens_share_one_budget_across_imports() {
        let temp = TempDir::new();
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\" 3\n");
        temp.write("a.ouro", "1 2\n");
        let limits = GraphLimits {
            source_tokens: 4,
            ..GraphLimits::default()
        };

        let error = load_with_limits(&root, limits).unwrap_err();
        let ModuleError::Lex { source, errors } = error else {
            panic!("expected aggregate token rejection");
        };
        assert!(source.ends_with("a.ouro"), "{source}");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("1-token limit"), "{}", errors[0]);
    }

    #[test]
    fn aggregate_graph_statement_budget_covers_dependency_asts() {
        let temp = TempDir::new();
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\" \"y\" OUTPUT\n");
        temp.write("a.ouro", "\"x\" OUTPUT\n");
        let limits = GraphLimits {
            expanded_statements: 5,
            ..GraphLimits::default()
        };

        let error = load_with_limits(&root, limits).unwrap_err();
        let ModuleError::LocatedParse {
            source, message, ..
        } = error
        else {
            panic!("expected aggregate statement rejection");
        };
        assert!(source.ends_with("root.ouro"), "{source}");
        assert_eq!(
            message,
            "expanded statement budget exceeds 2 while parsing string literal"
        );
    }

    #[test]
    fn duplicate_source_symbols_are_link_errors() {
        let temp = TempDir::new();
        temp.write("a.ouro", "PROCEDURE same { 1 }\n");
        temp.write("b.ouro", "PROCEDURE same { 2 }\n");
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\nIMPORT \"b.ouro\"\n0 POP\n");

        let error = ModuleGraph::load(root, Vec::new()).unwrap_err();
        assert!(error.to_string().contains("duplicate linked symbol 'same'"));
        let ModuleError::DuplicateLocatedSymbol { spans, .. } = error else {
            panic!("expected declaration locations on the duplicate symbol");
        };
        let (Some(first), Some(second)) = *spans else {
            panic!("expected both declaration spans");
        };
        assert_ne!(first.source, second.source);
        assert_eq!(first.range, crate::source::TextRange::new(0, 20));
        assert_eq!(second.range, crate::source::TextRange::new(0, 20));
    }

    #[test]
    fn imported_properties_and_initializers_are_preserved() {
        let temp = TempDir::new();
        temp.write(
            "library.ouro",
            "PROPERTY answer { ALL_FIXED CELL 0 EQ 42; }\n42 OUTPUT\n",
        );
        let root = temp.write("root.ouro", "IMPORT \"library.ouro\"\n1 OUTPUT\n");

        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(graph.program().temporal_properties.len(), 1);
        assert_eq!(values(graph.into_program()), vec![42, 1]);
    }

    #[test]
    fn imported_named_temporals_are_typed_visible_and_located() {
        let temp = TempDir::new();
        temp.write("cells.ouro", "TEMPORAL future @ 7 DEFAULT 99;\n");
        let root = temp.write(
            "root.ouro",
            "IMPORT \"cells.ouro\"\nPROPERTY safe { ALL_FIXED CELL future EQ 0; }\nfuture POP\n",
        );

        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(graph.program().temporal_declarations.len(), 1);
        assert_eq!(graph.program().temporal_declarations[0].default, 99);
        let hir = graph.resolve_hir().unwrap();
        assert_eq!(hir.temporals.len(), 1);
        assert!(hir.temporals[0].location.is_some());
        assert_eq!(
            hir.properties[0].cells[0].temporal,
            Some(hir.temporals[0].id)
        );
        assert!(matches!(
            hir.body[0].kind,
            crate::hir::HirStmtKind::ReadTemporal { id, address: 7 }
                if id == hir.temporals[0].id
        ));
        let reference_location = hir.body[0].location.unwrap();
        let code = crate::bytecode::BytecodeProgram::compile(&hir).unwrap();
        assert_eq!(
            code.source_map
                .iter()
                .filter(|entry| entry.span == reference_location)
                .count(),
            2
        );
        assert!(graph.locations().temporals[0].is_some());
    }

    #[test]
    fn duplicate_temporal_addresses_report_both_sources() {
        let temp = TempDir::new();
        temp.write("a.ouro", "TEMPORAL alpha @ 3 DEFAULT 0;\n");
        temp.write("b.ouro", "TEMPORAL beta @ 3 DEFAULT 1;\n");
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\nIMPORT \"b.ouro\"\n");

        let error = ModuleGraph::load(root, Vec::new()).unwrap_err();
        assert!(error.to_string().contains("TEMPORAL address 3"));
        let ModuleError::DuplicateLocatedSymbol { spans, .. } = error else {
            panic!("expected located duplicate temporal address");
        };
        assert!(spans.0.is_some() && spans.1.is_some());
    }

    #[test]
    fn imported_manifests_are_retained_located_and_typed() {
        let temp = TempDir::new();
        temp.write("library.ouro", "MANIFEST ANSWER = 42;\n");
        let root = temp.write("root.ouro", "IMPORT \"library.ouro\"\nANSWER OUTPUT\n");

        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert_eq!(graph.program().manifests.len(), 1);
        let declaration_span = graph.locations().manifests[0].unwrap();
        assert_eq!(
            graph.sources().slice(declaration_span).unwrap(),
            "MANIFEST ANSWER = 42;"
        );
        let hir = graph.resolve_hir().unwrap();
        assert_eq!(hir.constants.len(), 1);
        assert_eq!(hir.constants[0].name, "ANSWER");
        assert!(matches!(
            hir.body[0].kind,
            crate::hir::HirStmtKind::PushConstant { id, value: 42 }
                if id == crate::hir::ConstantId::try_from_index(0).unwrap()
        ));
    }

    #[test]
    fn duplicate_manifest_symbols_report_both_source_locations() {
        let temp = TempDir::new();
        temp.write("a.ouro", "MANIFEST SAME = 1;\n");
        temp.write("b.ouro", "MANIFEST SAME = 2;\n");
        let root = temp.write("root.ouro", "IMPORT \"a.ouro\"\nIMPORT \"b.ouro\"\n");

        let error = ModuleGraph::load(root, Vec::new()).unwrap_err();
        let ModuleError::DuplicateLocatedSymbol { symbol, spans, .. } = error else {
            panic!("expected a located duplicate MANIFEST error");
        };
        assert_eq!(symbol, "same");
        assert!(spans.0.is_some());
        assert!(spans.1.is_some());
    }

    #[test]
    fn linked_hir_and_bytecode_retain_locations_across_modules_and_quotes() {
        let temp = TempDir::new();
        temp.write(
            "lib.ouro",
            "PROCEDURE café { 1 IF { [ 2 OUTPUT ] EXEC } ELSE { 3 OUTPUT } }\n",
        );
        let root = temp.write("root.ouro", "IMPORT \"lib.ouro\"\ncafé\n");

        let graph = ModuleGraph::load(root, Vec::new()).unwrap();
        assert!(graph.locations().matches(graph.program()));
        let hir = graph.resolve_hir().unwrap();
        let bytecode = BytecodeProgram::compile(&hir).unwrap();
        assert!(!bytecode.source_map.is_empty());

        let mapped_text = bytecode
            .source_map
            .iter()
            .map(|entry| graph.sources().slice(entry.span).unwrap())
            .collect::<Vec<_>>();
        assert!(mapped_text.contains(&"café"));
        assert!(mapped_text.contains(&"2"));
        assert!(mapped_text.iter().any(|text| text.starts_with("IF")));

        let call_pc = bytecode
            .instructions
            .iter()
            .position(|instruction| matches!(instruction, Instruction::CallProcedure(_)))
            .unwrap() as u32;
        let call_span = bytecode
            .source_map
            .iter()
            .find(|entry| entry.instruction == call_pc)
            .unwrap()
            .span;
        assert_eq!(graph.sources().slice(call_span).unwrap(), "café");
    }

    #[test]
    fn parse_errors_report_exact_utf8_byte_spans() {
        let temp = TempDir::new();
        let root = temp.write("bad.ouro", "\"λ\" OUTPUT\nBOGUS 1\n");

        let error = ModuleGraph::load(root, Vec::new()).unwrap_err();
        let ModuleError::LocatedParse {
            span: Some(span), ..
        } = error
        else {
            panic!("expected a source-located parse error");
        };
        assert_eq!(span.range.start, "\"λ\" OUTPUT\n".len());
        assert_eq!(span.range.end - span.range.start, "BOGUS".len());
    }
}
