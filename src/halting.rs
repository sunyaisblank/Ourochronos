//! Sound finite observations for the classical halting problem.
//!
//! This analyzer never claims unrestricted nonhalting. Exhausting `T`
//! instructions yields `NotHaltedWithinBound`, which is an unknown result for
//! the unbounded run.

use crate::admission::{admit_program, AdmissionConfig};
use crate::ast::{EffectClass, OpCode, Program};
use crate::bytecode::{BytecodeProgram, Instruction};
use crate::bytecode_vm::{
    bytecode_vm_supports, BytecodeVm, BytecodeVmConfig, BytecodeVmError, BytecodeVmStatus,
};
use crate::checkpoint::{
    CheckpointError, CheckpointOutcome, CheckpointPauseReason, ClassicalCheckpoint,
    ClassicalCheckpointConfig,
};
use crate::core::{OutputItem, PagedMemory};
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub enum BoundedHaltingResult {
    Halted {
        instructions: u64,
        output: Vec<OutputItem>,
    },
    NotHaltedWithinBound {
        instruction_bound: u64,
    },
    Unsupported {
        reason: String,
    },
    RuntimeError {
        message: String,
        instructions: u64,
    },
}

pub struct BoundedHaltingAnalyzer;

/// Observations from a verified, resumable deterministic classical query.
/// Every pause remains UNKNOWN for the unrestricted halting question.
#[derive(Debug)]
pub enum ResumableHaltingResult {
    Halted {
        instructions: u64,
        output: Vec<OutputItem>,
    },
    Unknown {
        instructions: u64,
        reason: CheckpointPauseReason,
        checkpoint: ClassicalCheckpoint,
    },
    RuntimeError {
        message: String,
        instructions: u64,
    },
}

impl BoundedHaltingAnalyzer {
    /// Start the explicitly declared checkpoint core. This narrower recovery
    /// boundary does not change the existing one-shot analyzer's coverage.
    pub fn start_checkpoint(
        program: &BytecodeProgram,
        config: ClassicalCheckpointConfig,
    ) -> Result<ClassicalCheckpoint, CheckpointError> {
        ClassicalCheckpoint::start(program, config)
    }

    pub fn analyze_checkpoint(
        checkpoint: ClassicalCheckpoint,
        allowance: u64,
        cancellation: Option<&AtomicBool>,
    ) -> Result<ResumableHaltingResult, CheckpointError> {
        match checkpoint.run_slice(allowance, cancellation)? {
            CheckpointOutcome::Complete(execution) => Ok(ResumableHaltingResult::Halted {
                instructions: execution.instructions_executed,
                output: execution.output,
            }),
            CheckpointOutcome::Paused { checkpoint, reason } => {
                Ok(ResumableHaltingResult::Unknown {
                    instructions: checkpoint.instructions_executed(),
                    reason,
                    checkpoint,
                })
            }
            CheckpointOutcome::Fault {
                error,
                instructions,
            } => Ok(ResumableHaltingResult::RuntimeError {
                message: error.to_string(),
                instructions,
            }),
        }
    }

    /// Observe one deterministic, non-temporal program for at most `T`
    /// instructions. Both falling off the end and explicit HALT count as halt.
    pub fn analyze(
        program: &Program,
        instruction_bound: u64,
        memory_cells: usize,
    ) -> BoundedHaltingResult {
        if instruction_bound == 0 {
            return BoundedHaltingResult::Unsupported {
                reason: "instruction bound must be greater than zero".into(),
            };
        }
        if memory_cells == 0 {
            return BoundedHaltingResult::Unsupported {
                reason: "memory width must be greater than zero".into(),
            };
        }
        let admitted = match admit_program(program, AdmissionConfig { memory_cells }) {
            Ok(admitted) => admitted,
            Err(error) => {
                return BoundedHaltingResult::Unsupported {
                    reason: format!("source admission failed: {error}"),
                };
            }
        };
        Self::analyze_bytecode(admitted.program(), instruction_bound, memory_cells)
    }

    /// Observe the authoritative linked executable representation for at most
    /// `T` fetched instructions.
    pub fn analyze_bytecode(
        program: &BytecodeProgram,
        instruction_bound: u64,
        memory_cells: usize,
    ) -> BoundedHaltingResult {
        if instruction_bound == 0 {
            return BoundedHaltingResult::Unsupported {
                reason: "instruction bound must be greater than zero".into(),
            };
        }
        if memory_cells == 0 {
            return BoundedHaltingResult::Unsupported {
                reason: "memory width must be greater than zero".into(),
            };
        }
        if let Err(error) = program.validate() {
            return BoundedHaltingResult::Unsupported {
                reason: format!("invalid bytecode: {error}"),
            };
        }
        if let Some(reason) = bytecode_halting_boundary(program) {
            return BoundedHaltingResult::Unsupported { reason };
        }
        let memory = match PagedMemory::with_size(memory_cells) {
            Ok(memory) => memory,
            Err(error) => {
                return BoundedHaltingResult::Unsupported {
                    reason: error.to_string(),
                };
            }
        };
        match BytecodeVm::with_config(BytecodeVmConfig {
            max_instructions: instruction_bound,
            ..BytecodeVmConfig::default()
        })
        .run(program, &memory)
        {
            Ok(result)
                if matches!(
                    result.status,
                    BytecodeVmStatus::Finished | BytecodeVmStatus::Halted
                ) =>
            {
                BoundedHaltingResult::Halted {
                    instructions: result.instructions_executed,
                    output: result.output,
                }
            }
            Ok(result) if result.status == BytecodeVmStatus::Paradox => {
                BoundedHaltingResult::Unsupported {
                    reason: "PARADOX is outside classical halting semantics".into(),
                }
            }
            Ok(result) => BoundedHaltingResult::RuntimeError {
                message: "bytecode VM returned a nonterminal state".into(),
                instructions: result.instructions_executed,
            },
            Err(BytecodeVmError::GasExhausted { .. }) => {
                BoundedHaltingResult::NotHaltedWithinBound { instruction_bound }
            }
            Err(error) => BoundedHaltingResult::RuntimeError {
                message: error.to_string(),
                // The public VM error is deliberately side-effect free but
                // does not expose a partial execution snapshot.
                instructions: 0,
            },
        }
    }
}

fn bytecode_halting_boundary(program: &BytecodeProgram) -> Option<String> {
    let mut pending = VecDeque::from([program.main]);
    let mut visited = HashSet::new();
    while let Some(range) = pending.pop_front() {
        if !visited.insert((range.start, range.end)) {
            continue;
        }
        let start = range.start as usize;
        let end = range.end as usize;
        let Some(instructions) = program.instructions.get(start..end) else {
            return Some("invalid executable code range".into());
        };
        for instruction in instructions {
            match instruction {
                Instruction::TemporalEnter { .. } | Instruction::TemporalExit { .. } => {
                    return Some("temporal scopes are outside classical halting semantics".into());
                }
                Instruction::CallForeign(_) => {
                    return Some(
                        "foreign calls are outside deterministic classical halting analysis".into(),
                    );
                }
                Instruction::CallProcedure(id) => {
                    let Some(entry) = program.procedures.get(id.index()) else {
                        return Some("invalid procedure call target".into());
                    };
                    pending.push_back(entry.range);
                }
                Instruction::Primitive(opcode)
                    if !bytecode_vm_supports(*opcode)
                        || opcode.effect_class() != EffectClass::Pure
                        || matches!(
                            opcode,
                            OpCode::Oracle
                                | OpCode::Paradox
                                | OpCode::Input
                                | OpCode::Clock
                                | OpCode::Random
                        ) =>
                {
                    return Some(format!(
                        "{} is temporal, nondeterministic, external, or unsupported; bounded halting analysis covers the deterministic classical bytecode core",
                        opcode.name()
                    ));
                }
                Instruction::Primitive(
                    OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec,
                ) => {
                    // Quotation identity is a runtime word for these
                    // combinators. Conservatively include every quotation;
                    // this can reject an unused quote but cannot admit a
                    // temporal or externally stateful execution.
                    pending.extend(program.quotations.iter().map(|entry| entry.range));
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn reports_a_finite_halt() {
        let program = parse("40 2 ADD OUTPUT").unwrap();
        match BoundedHaltingAnalyzer::analyze(&program, 100, 16) {
            BoundedHaltingResult::Halted { output, .. } => {
                assert!(matches!(&output[0], OutputItem::Val(value) if value.val == 42));
            }
            other => panic!("expected halt, got {:?}", other),
        }
    }

    #[test]
    fn exhaustion_is_not_reported_as_nonhalting() {
        let program = parse("WHILE { 1 } { NOP }").unwrap();
        assert!(matches!(
            BoundedHaltingAnalyzer::analyze(&program, 100, 16),
            BoundedHaltingResult::NotHaltedWithinBound {
                instruction_bound: 100
            }
        ));
    }

    #[test]
    fn temporal_programs_are_outside_the_classical_analyzer() {
        let program = parse("0 ORACLE POP").unwrap();
        assert!(matches!(
            BoundedHaltingAnalyzer::analyze(&program, 100, 16),
            BoundedHaltingResult::Unsupported { .. }
        ));
    }
}
