//! Shared, mandatory frontend and bytecode pipeline for developer tooling.
//!
//! Keeping this path in one module prevents the REPL and language server from
//! acquiring their own parser, checker, or interpreter semantics.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::ast::{LocatedProgram, ProgramLocations, Stmt, StmtLocations};
use crate::bytecode::BytecodeProgram;
use crate::bytecode_verifier::{verify_default, BytecodeVerificationReport};
use crate::hir::HirProgram;
use crate::lexer::{lex, LocatedToken, Token};
use crate::linker::link;
use crate::module_graph::{ModuleError, ModuleGraph};
use crate::parser::Parser;
use crate::semantics::{check as check_semantics, SemanticsReport};
use crate::source::{SourceManager, SourceSpan};
use crate::temporal::region::TemporalRegionReport;
use crate::types::{type_check, TypeCheckResult};

/// Compiler phase responsible for a tooling diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolingPhase {
    Lex,
    Parse,
    Resolve,
    Semantics,
    Types,
    Regions,
    Bytecode,
    Verify,
}

impl ToolingPhase {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Lex => "L001",
            Self::Parse => "P001",
            Self::Resolve => "R001",
            Self::Semantics => "S001",
            Self::Types => "T001",
            Self::Regions => "G001",
            Self::Bytecode => "B001",
            Self::Verify => "V001",
        }
    }
}

/// One hard compiler diagnostic, retaining an exact source span when the
/// producing phase can identify one.
#[derive(Debug, Clone)]
pub(crate) struct ToolingDiagnostic {
    pub(crate) phase: ToolingPhase,
    pub(crate) message: String,
    pub(crate) span: Option<SourceSpan>,
}

/// Result of the complete tooling compiler pipeline. Earlier products remain
/// available after a later mandatory phase reports errors, which lets an IDE
/// continue to offer symbols and semantic tokens for an invalid document.
#[derive(Debug, Clone)]
pub(crate) struct ToolingAnalysis {
    pub(crate) primary_source: Option<crate::source::SourceId>,
    pub(crate) tokens: Vec<LocatedToken>,
    pub(crate) located: Option<LocatedProgram>,
    pub(crate) hir: Option<HirProgram>,
    pub(crate) semantics: Option<SemanticsReport>,
    pub(crate) types: Option<TypeCheckResult>,
    pub(crate) regions: Option<TemporalRegionReport>,
    pub(crate) bytecode: Option<BytecodeProgram>,
    pub(crate) verification: Option<BytecodeVerificationReport>,
    pub(crate) diagnostics: Vec<ToolingDiagnostic>,
}

impl ToolingAnalysis {
    fn empty() -> Self {
        Self {
            primary_source: None,
            tokens: Vec::new(),
            located: None,
            hir: None,
            semantics: None,
            types: None,
            regions: None,
            bytecode: None,
            verification: None,
            diagnostics: Vec::new(),
        }
    }

    pub(crate) fn is_executable(&self) -> bool {
        self.diagnostics.is_empty() && self.bytecode.is_some() && self.verification.is_some()
    }
}

/// Run the same located frontend, HIR resolution, mandatory analyses,
/// bytecode lowering, and independent verifier used by production execution.
pub(crate) fn analyze_virtual_source(
    display_name: &str,
    text: &str,
    memory_cells: usize,
) -> ToolingAnalysis {
    let mut result = ToolingAnalysis::empty();
    if text.len() > crate::source::MAX_SOURCE_FILE_BYTES {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Parse,
            message: format!(
                "virtual source is {} bytes; limit is {}",
                text.len(),
                crate::source::MAX_SOURCE_FILE_BYTES
            ),
            span: None,
        });
        return result;
    }
    let mut sources = SourceManager::new();
    let source = sources.add_virtual(display_name, text);
    result.primary_source = Some(source);
    result.tokens = match lex(source, text) {
        Ok(tokens) => tokens,
        Err(errors) => {
            result
                .diagnostics
                .extend(errors.into_iter().map(|error| ToolingDiagnostic {
                    phase: ToolingPhase::Lex,
                    message: error.kind.to_string(),
                    span: Some(error.span),
                }));
            return result;
        }
    };

    let tokens: Vec<Token> = result
        .tokens
        .iter()
        .map(|token| token.token.clone())
        .collect();
    let spans: Vec<SourceSpan> = result.tokens.iter().map(|token| token.span).collect();
    let mut parser = match Parser::new_with_deferred_imports_and_spans(&tokens, &spans) {
        Ok(parser) => parser,
        Err(message) => {
            result.diagnostics.push(ToolingDiagnostic {
                phase: ToolingPhase::Parse,
                message,
                span: None,
            });
            return result;
        }
    };

    // Source declarations shadow the standard prelude, matching the module
    // graph. Filter before registering interfaces so a local declaration's
    // stack signature is also authoritative while parsing its callers.
    let local_procedures = local_procedure_names(&tokens);
    let mut prelude = crate::StdLib::procedures();
    prelude.retain(|procedure| !local_procedures.contains(&procedure.name.to_lowercase()));
    parser.register_procedure_interfaces(&prelude);

    let mut located = match parser.parse_located_program() {
        Ok(located) => located,
        Err(message) => {
            result.diagnostics.push(ToolingDiagnostic {
                phase: ToolingPhase::Parse,
                message,
                span: parser.error_source_span(),
            });
            return result;
        }
    };
    let imports = parser.take_deferred_imports();
    if !imports.is_empty() {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Parse,
            message: format!(
                "IMPORT requires a file-backed module graph for importer-relative resolution (found: {})",
                imports.join(", ")
            ),
            span: None,
        });
        return result;
    }
    for mut procedure in prelude {
        procedure.name = procedure.name.to_lowercase();
        located.program.procedures.push(procedure);
        located.locations.procedures.push(None);
    }
    debug_assert!(located.locations.matches(&located.program));

    analyze_located(result, located, memory_cells, Lowering::Monolithic)
}

/// Analyze a disk-backed entry through the canonical importer-relative module
/// graph and the same per-source object linker used by production compilation.
/// `root_text` overlays an open editor buffer without changing its filesystem
/// identity or the base directory used by its imports.
pub(crate) fn analyze_file_source(
    path: &Path,
    root_text: Option<&str>,
    memory_cells: usize,
) -> ToolingAnalysis {
    let graph = match root_text {
        Some(text) => ModuleGraph::load_with_root_text(path, text, crate::StdLib::procedures()),
        None => ModuleGraph::load(path, crate::StdLib::procedures()),
    };
    let graph = match graph {
        Ok(graph) => graph,
        Err(error) => {
            let mut result = ToolingAnalysis::empty();
            result.diagnostics.push(ToolingDiagnostic {
                phase: module_error_phase(&error),
                message: error.to_string(),
                span: module_error_span(&error),
            });
            return result;
        }
    };

    let root = graph.root();
    let root_source = graph
        .sources()
        .get(root)
        .expect("a loaded module graph retains its root source");
    let mut result = ToolingAnalysis::empty();
    result.primary_source = Some(root);
    result.tokens = match lex(root, root_source.text()) {
        Ok(tokens) => tokens,
        Err(errors) => {
            result
                .diagnostics
                .extend(errors.into_iter().map(|error| ToolingDiagnostic {
                    phase: ToolingPhase::Lex,
                    message: error.kind.to_string(),
                    span: Some(error.span),
                }));
            return result;
        }
    };
    let located = LocatedProgram {
        program: graph.program().clone(),
        locations: graph.locations().clone(),
    };
    analyze_located(result, located, memory_cells, Lowering::ModuleGraph(&graph))
}

enum Lowering<'a> {
    Monolithic,
    ModuleGraph(&'a ModuleGraph),
}

fn analyze_located(
    mut result: ToolingAnalysis,
    located: LocatedProgram,
    memory_cells: usize,
    lowering: Lowering<'_>,
) -> ToolingAnalysis {
    let type_result = type_check(&located.program);
    let statement_spans = checker_statement_spans(&located.program, &located.locations);
    for error in &type_result.errors {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Types,
            message: error.to_string(),
            span: error
                .location
                .and_then(|index| statement_spans.get(&index).copied()),
        });
    }
    for violation in &type_result.effect_violations {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Types,
            message: format!(
                "procedure '{}' declares {:?} but has effects {:?}",
                violation.procedure_name, violation.declared, violation.actual
            ),
            span: violation
                .location
                .and_then(|index| statement_spans.get(&index).copied()),
        });
    }
    for violation in &type_result.linear_violations {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Types,
            message: violation.to_string(),
            span: statement_spans.get(&violation.stmt_index).copied(),
        });
    }

    let region_result = TemporalRegionReport::analyze(&located.program, memory_cells);
    let region_spans = temporal_region_spans(&located.program, &located.locations);
    for issue in &region_result.issues {
        result.diagnostics.push(ToolingDiagnostic {
            phase: ToolingPhase::Regions,
            message: issue.message.clone(),
            span: region_spans.get(issue.region).copied().flatten(),
        });
    }

    let hir = match HirProgram::resolve_located(&located.program, &located.locations) {
        Ok(hir) => Some(hir),
        Err(errors) => {
            result
                .diagnostics
                .extend(errors.into_iter().map(|error| ToolingDiagnostic {
                    phase: ToolingPhase::Resolve,
                    message: error.to_string(),
                    span: error.location,
                }));
            None
        }
    };

    let semantics = hir.as_ref().map(check_semantics);
    if let Some(report) = &semantics {
        result.diagnostics.extend(report.errors.iter().map(|error| {
            ToolingDiagnostic {
                phase: ToolingPhase::Semantics,
                message: format!("{:?}", error.kind),
                span: error
                    .site
                    .location
                    .or_else(|| first_user_span(&located.locations)),
            }
        }));
    }

    result.types = Some(type_result);
    result.regions = Some(region_result);
    result.hir = hir;
    result.semantics = semantics;
    result.located = Some(located);

    if !result.diagnostics.is_empty() {
        return result;
    }

    let lowered = match lowering {
        Lowering::Monolithic => BytecodeProgram::compile(
            result
                .hir
                .as_ref()
                .expect("successful resolution stores HIR"),
        )
        .map_err(|error| error.to_string()),
        Lowering::ModuleGraph(graph) => graph
            .compile_objects()
            .map_err(|error| error.to_string())
            .and_then(|objects| link(&objects).map_err(|error| error.to_string())),
    };
    let bytecode = match lowered {
        Ok(bytecode) => bytecode,
        Err(error) => {
            result.diagnostics.push(ToolingDiagnostic {
                phase: ToolingPhase::Bytecode,
                message: error,
                span: None,
            });
            return result;
        }
    };
    let verification = match verify_default(&bytecode) {
        Ok(report) => report,
        Err(error) => {
            result.diagnostics.push(ToolingDiagnostic {
                phase: ToolingPhase::Verify,
                message: error.to_string(),
                span: None,
            });
            result.bytecode = Some(bytecode);
            return result;
        }
    };
    result.bytecode = Some(bytecode);
    result.verification = Some(verification);
    result
}

fn module_error_phase(error: &ModuleError) -> ToolingPhase {
    match error {
        ModuleError::Lex { .. } => ToolingPhase::Lex,
        _ => ToolingPhase::Parse,
    }
}

fn module_error_span(error: &ModuleError) -> Option<SourceSpan> {
    match error {
        ModuleError::LocatedParse { span, .. } | ModuleError::ImportCycle { span, .. } => *span,
        ModuleError::DuplicateLocatedSymbol { spans, .. } => spans.1.or(spans.0),
        _ => None,
    }
}

fn local_procedure_names(tokens: &[Token]) -> HashSet<String> {
    tokens
        .windows(2)
        .filter_map(|pair| match pair {
            [Token::Word(keyword), Token::Word(name)]
                if keyword.eq_ignore_ascii_case("PROCEDURE")
                    || keyword.eq_ignore_ascii_case("PROC") =>
            {
                Some(name.to_lowercase())
            }
            _ => None,
        })
        .collect()
}

fn first_user_span(locations: &ProgramLocations) -> Option<SourceSpan> {
    locations
        .body
        .first()
        .map(|location| location.span)
        .or_else(|| {
            locations
                .procedures
                .iter()
                .flatten()
                .next()
                .map(|location| location.declaration)
        })
}

/// Simulate the legacy type checker's statement counter closely enough to
/// attach its index diagnostics to retained source. Nested statements can
/// share an index with their parent; the innermost available span wins.
fn checker_statement_spans(
    program: &crate::ast::Program,
    locations: &ProgramLocations,
) -> HashMap<usize, SourceSpan> {
    let mut spans = HashMap::new();
    let mut index = 0usize;
    for (procedure, location) in program.procedures.iter().zip(&locations.procedures) {
        map_checker_block(
            &procedure.body,
            location.as_ref().map(|location| location.body.as_slice()),
            &mut index,
            &mut spans,
        );
    }
    map_checker_block(
        &program.body,
        Some(locations.body.as_slice()),
        &mut index,
        &mut spans,
    );
    spans
}

fn map_checker_block(
    statements: &[Stmt],
    locations: Option<&[StmtLocations]>,
    index: &mut usize,
    spans: &mut HashMap<usize, SourceSpan>,
) {
    for (statement_index, statement) in statements.iter().enumerate() {
        let location = locations.and_then(|locations| locations.get(statement_index));
        if let Some(location) = location {
            spans.insert(*index, location.span);
        }
        for (child_index, child) in statement.child_blocks().into_iter().enumerate() {
            map_checker_block(
                child,
                location
                    .and_then(|location| location.child_blocks.get(child_index).map(Vec::as_slice)),
                index,
                spans,
            );
        }
        *index = index.saturating_add(1);
    }
}

fn temporal_region_spans(
    program: &crate::ast::Program,
    locations: &ProgramLocations,
) -> Vec<Option<SourceSpan>> {
    let mut result = Vec::new();
    collect_region_spans(&program.body, Some(&locations.body), &mut result);
    for (procedure, locations) in program.procedures.iter().zip(&locations.procedures) {
        collect_region_spans(
            &procedure.body,
            locations
                .as_ref()
                .map(|locations| locations.body.as_slice()),
            &mut result,
        );
    }
    for (quote, locations) in program.quotes.iter().zip(&locations.quotes) {
        collect_region_spans(quote, Some(locations), &mut result);
    }
    result
}

fn collect_region_spans(
    statements: &[Stmt],
    locations: Option<&[StmtLocations]>,
    result: &mut Vec<Option<SourceSpan>>,
) {
    for (statement_index, statement) in statements.iter().enumerate() {
        let location = locations.and_then(|locations| locations.get(statement_index));
        if matches!(statement, Stmt::TemporalScope { .. }) {
            result.push(location.map(|location| location.span));
            // Region analysis treats the contents as a stronger boundary and
            // does not recursively discover nested host regions.
            continue;
        }
        for (child_index, child) in statement.child_blocks().into_iter().enumerate() {
            collect_region_spans(
                child,
                location
                    .and_then(|location| location.child_blocks.get(child_index).map(Vec::as_slice)),
                result,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_pipeline_retains_locations_and_verifies_bytecode() {
        let analysis = analyze_virtual_source("repl", "1 2 ADD OUTPUT", 16);
        assert!(
            analysis.diagnostics.is_empty(),
            "{:?}",
            analysis.diagnostics
        );
        assert!(analysis.is_executable());
        assert!(analysis
            .bytecode
            .as_ref()
            .is_some_and(|bytecode| !bytecode.source_map.is_empty()));
    }

    #[test]
    fn lexical_failure_is_exact_after_utf8() {
        let analysis = analyze_virtual_source("lsp", "\"λ\" ?", 16);
        let diagnostic = analysis.diagnostics.first().unwrap();
        assert_eq!(diagnostic.phase, ToolingPhase::Lex);
        assert_eq!(diagnostic.span.unwrap().range.start, 5);
        assert!(analysis.bytecode.is_none());
    }

    #[test]
    fn virtual_sources_reject_imports_without_reading_process_cwd() {
        let analysis =
            analyze_virtual_source("repl", "IMPORT \"examples/simple.ouro\" 99 OUTPUT", 16);
        let diagnostic = analysis.diagnostics.first().expect("import diagnostic");
        assert_eq!(diagnostic.phase, ToolingPhase::Parse);
        assert!(diagnostic.message.contains("file-backed module graph"));
        assert!(analysis.bytecode.is_none());
    }
}
