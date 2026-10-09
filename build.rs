//! Build-time enforcement for the cross-component alignment architecture.
//!
//! Runtime tests prove behavior, while this gate prevents a production source
//! facade from silently reintroducing a weaker compiler path that those tests
//! do not yet exercise. Keep the checks narrow and structural: semantic
//! correctness remains the responsibility of Rust's type system and the test
//! suite.

use std::fs;

struct Required<'a> {
    file: &'a str,
    markers: &'a [&'a str],
}

fn main() {
    let requirements = [
        Required {
            file: "src/admission.rs",
            markers: &[
                "pub struct AdmittedProgram",
                "PreparedBytecode::new(bytecode)",
                "bytecode_vm_supports(*opcode)",
            ],
        },
        Required {
            file: "src/vm/executor.rs",
            markers: &["admit_program(", ".run_prepared(admitted.executable()"],
        },
        Required {
            file: "src/vm/fast_vm.rs",
            markers: &["admit_program(", ".run_prepared(admitted.executable()"],
        },
        Required {
            file: "src/temporal/timeloop.rs",
            markers: &["admit_program(program, AdmissionConfig { memory_cells })"],
        },
        Required {
            file: "src/temporal/global_solver.rs",
            markers: &[
                "admit_program(",
                "Self::compile_bytecode(admitted.program(), config)",
                "z3_config.set_proof_generation(true)",
                "solver_query: solver_smt",
                "certificate.verify_with_z3(config.solver_timeout_ms)",
            ],
        },
        Required {
            file: "src/temporal/transition_graph.rs",
            markers: &[
                "admit_program(",
                "BytecodeTransitionAnalyzer::analyze(admitted.program()",
            ],
        },
        Required {
            file: "src/halting.rs",
            markers: &["admit_program(program, AdmissionConfig { memory_cells })"],
        },
        Required {
            file: "src/temporal/smt_encoder.rs",
            markers: &["GlobalFixedPointSolver::compile("],
        },
        Required {
            file: "src/object_compiler.rs",
            markers: &["analyze_resolved_program(", "seal_analyzed(source, linked)"],
        },
        Required {
            file: "src/main.rs",
            markers: &[
                "graph.compile_objects_with_memory(memory_cells)",
                "PspaceUniformFamilyGenerator::certify_bytecode(",
                "ProjectionFamilyCertificate::prove(",
                "theorem.generate_circuit(input_bits)",
                "let result = generated.verify()",
                "Internal family-theorem mismatch",
                "theorem.cross_check_finite(circuit, certificate)",
            ],
        },
        Required {
            file: "src/family_verifier.rs",
            markers: &[
                "admit_program(",
                "BytecodeTransitionAnalyzer::analyze_with_frozen_input(",
                "conservative_chronology_bits(",
                "unanimous_boolean_readout(&analysis)",
                "certificate.check_structure()",
                "maximum_stack_depths: analysis.stack_depths.clone()",
            ],
        },
        Required {
            file: "src/uniform_family.rs",
            markers: &[
                "admit_program(",
                "certificate.check_structure()",
                "PspaceFamilyVerifier::verify_bytecode(",
                "canonical_generation_bounds(",
                "recognize_projection_template(&program)",
                "ProjectionCellSource::Temporal",
                "ProjectionCellSource::Xor",
                "ProjectionCellSource::ShiftRight",
                "expected_circuit_wire(",
                "projection_circuit_wire_bound(",
                "pub fn cross_check_finite(",
                "proved_for_all_nonempty_inputs",
            ],
        },
        Required {
            file: "src/temporal/ir.rs",
            markers: &["pub(crate) fn compile("],
        },
    ];

    for requirement in requirements {
        println!("cargo:rerun-if-changed={}", requirement.file);
        let source = fs::read_to_string(requirement.file).unwrap_or_else(|error| {
            panic!("alignment gate cannot read {}: {error}", requirement.file)
        });
        for marker in requirement.markers {
            assert!(
                source.contains(marker),
                "alignment gate: {} must retain architecture marker {:?}",
                requirement.file,
                marker
            );
        }
    }

    let forbidden_raw_source_compilers = [
        "src/vm/executor.rs",
        "src/vm/fast_vm.rs",
        "src/temporal/timeloop.rs",
        "src/temporal/global_solver.rs",
        "src/temporal/transition_graph.rs",
        "src/temporal/smt_encoder.rs",
        "src/halting.rs",
        "src/family_verifier.rs",
        "src/uniform_family.rs",
    ];
    for file in forbidden_raw_source_compilers {
        let source = fs::read_to_string(file)
            .unwrap_or_else(|error| panic!("alignment gate cannot read {file}: {error}"));
        let production = source
            .split_once("\n#[cfg(test)]\nmod tests")
            .map_or(source.as_str(), |(production, _)| production);
        for forbidden in ["HirProgram::resolve", "BytecodeProgram::compile"] {
            assert!(
                !production.contains(forbidden),
                "alignment gate: {file} bypasses canonical admission through {forbidden}"
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
