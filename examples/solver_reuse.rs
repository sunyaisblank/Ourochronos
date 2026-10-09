//! Bounded, reproducible solver-session/result-cache study; no production cache.
//!
//! Run `cargo run --locked --example solver_reuse --release -- [--cvc5-python PATH]`.
//! Twelve frozen fixtures, 32 exact queries each and five repeats are compared.
//! Every query explicitly restricts anamnesis to <=8 masked bits and zero outside
//! its whole-main scope. UNSAT concerns that declared finite domain only.
//! An independent small numeric interpreter enumerates the entire domain, and
//! every row (including gas, stack and outside frame) is compared with the VM.
//! SAT models from either backend must replay the ORIGINAL VM with frozen INPUT.
//! Solver agreement is backend evidence, not an independently checked proof.
//!
//! INPUT specialization is limited to unconditional linear main bodies. INPUT
//! and its replacement PushWord both cost one fetched record. Calls/branches,
//! unused tape entries, live input, host/heap/output/loops and recursion are
//! excluded from that specialization. Ordinary fixtures permit forward IF and
//! acyclic direct procedures. Source/HIR compilation remains a trusted boundary.
//! Cache keys bind exact original/specialized canonical artifacts, tape, query,
//! entire IR/domain text, semantics version and exact closed VM/solver settings.
//! The bounded process-local cache holds only validated SAT or exhaustive UNSAT;
//! it is private to this example and never used by production solving.

use ourochronos::ast::OpCode;
use ourochronos::bytecode::{CodeRange, Instruction};
use ourochronos::finite_proof::FiniteProofConfig;
use ourochronos::hir::HirProgram;
use ourochronos::parser::parse;
use ourochronos::stdlib::StdLib;
use ourochronos::temporal::global_solver::{GlobalFixedPointSolver, GlobalSolveConfig};
use ourochronos::temporal::ir::IrCompleteness;
use ourochronos::{
    BoundsPolicy, BytecodeProgram, BytecodeVm, BytecodeVmStatus, PagedMemory, Value,
};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use z3::ast::{Array, Ast, BV};
use z3::{Config, Context, SatResult, Solver, Sort};

type StudyResult<T> = Result<T, String>;
pub const STUDY_SEMANTICS: &str = "ourochronos-finite-solver-study-v1";
pub const MAX_CACHE_ENTRIES: usize = 1000;
pub const MAX_CACHE_BYTES: usize = 1024 * 1024;
const REPEATS: usize = 5;
const TIMEOUT_MS: u64 = 3000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    Exists,
    CellEquals { cell: usize, value: u64 },
    CellNotEquals { cell: usize, value: u64 },
}

impl Query {
    pub fn holds(&self, memory: &[u64]) -> bool {
        match *self {
            Self::Exists => true,
            Self::CellEquals { cell, value } => memory.get(cell) == Some(&value),
            Self::CellNotEquals { cell, value } => memory.get(cell).is_some_and(|&v| v != value),
        }
    }
    fn smt(&self) -> String {
        match *self {
            Self::Exists => "true".into(),
            Self::CellEquals { cell, value } => {
                format!("(= (select anamnesis (_ bv{cell} 64)) (_ bv{value} 64))")
            }
            Self::CellNotEquals { cell, value } => {
                format!("(not (= (select anamnesis (_ bv{cell} 64)) (_ bv{value} 64)))")
            }
        }
    }
    fn validate(&self, cells: usize) -> StudyResult<()> {
        match self {
            Self::Exists => Ok(()),
            Self::CellEquals { cell, .. } | Self::CellNotEquals { cell, .. } if *cell < cells => {
                Ok(())
            }
            _ => Err("query address outside declared scope".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Sat(Vec<u64>),
    Unsat,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scope {
    pub cells: usize,
    pub bits: u8,
}

impl Scope {
    fn mask(self) -> u64 {
        (1u64 << self.bits) - 1
    }
    pub fn states(self) -> usize {
        1usize << (self.cells * self.bits as usize)
    }
    fn input(self, row: usize, memory_cells: usize) -> Vec<u64> {
        (0..memory_cells)
            .map(|cell| {
                if cell < self.cells {
                    ((row as u64) >> (cell * self.bits as usize)) & self.mask()
                } else {
                    0
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Workload {
    pub name: String,
    original: BytecodeProgram,
    specialized: BytecodeProgram,
    tape: Vec<u64>,
    profile: FiniteProofConfig,
    scope: Scope,
    solver_config: GlobalSolveConfig,
    smt_base: String,
    pub queries: Vec<Query>,
    fixed: Vec<Vec<u64>>,
    // Immutable construction snapshot permits cheap exact-key lookups; it is
    // never persisted or shared with another executable revision.
    context_fingerprint: [u8; 32],
}

fn numeric(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Nop
            | OpCode::Pop
            | OpCode::Dup
            | OpCode::Swap
            | OpCode::Over
            | OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::And
            | OpCode::Or
            | OpCode::Xor
            | OpCode::Shl
            | OpCode::Shr
            | OpCode::Eq
            | OpCode::Neq
            | OpCode::Not
            | OpCode::Oracle
            | OpCode::PresentRead
            | OpCode::Prophecy
    )
}

/// Static finite profile, independent of solver/IR admission. Unused prelude
/// units are structurally checked but cannot be dynamically invoked.
pub fn admit(program: &BytecodeProgram, profile: &FiniteProofConfig) -> StudyResult<Scope> {
    if program.instructions.len() > 4096
        || program.procedures.len() > 64
        || profile.memory_cells == 0
        || profile.memory_cells > 16
        || profile.max_instructions == 0
        || profile.max_instructions > 4096
        || profile.max_stack_depth > 256
        || profile.max_call_depth > 64
        || profile.max_temporal_depth != 1
        || !program.quotations.is_empty()
        || !program.foreigns.is_empty()
    {
        return Err("outside bounded study profile".into());
    }
    program.validate().map_err(|e| e.to_string())?;
    let main = program.main;
    if main.end - main.start < 3 {
        return Err("main scope missing".into());
    }
    let exit = main.end - 2;
    let scope = match program.instructions[main.start as usize] {
        Instruction::TemporalEnter {
            base: 0,
            size,
            cell_bits,
            exit_target,
        } if exit_target == exit
            && size > 0
            && size <= profile.memory_cells as u64
            && cell_bits > 0
            && cell_bits <= 8
            && size * u64::from(cell_bits) <= 8 =>
        {
            Scope {
                cells: size as usize,
                bits: cell_bits,
            }
        }
        _ => return Err("requires whole-main base-zero scope of at most eight bits".into()),
    };
    if program.instructions[exit as usize]
        != (Instruction::TemporalExit {
            enter_target: main.start,
        })
        || program.instructions[(main.end - 1) as usize] != Instruction::Return
    {
        return Err("main must end EXIT/RETURN".into());
    }
    let mut colors = vec![0u8; program.procedures.len() + 1];
    fn visit(unit: usize, p: &BytecodeProgram, exit: u32, colors: &mut [u8]) -> StudyResult<()> {
        if colors[unit] == 1 {
            return Err("recursive call".into());
        }
        if colors[unit] == 2 {
            return Ok(());
        }
        colors[unit] = 1;
        let range = if unit == 0 {
            p.main
        } else {
            p.procedures[unit - 1].range
        };
        let limit = if unit == 0 { exit } else { range.end - 1 };
        for pc in range.start..range.end {
            match p.instructions[pc as usize] {
                Instruction::PushWord(_) => {}
                Instruction::Primitive(op) if numeric(op) => {}
                Instruction::CallProcedure(id) => visit(id.index() + 1, p, exit, colors)?,
                Instruction::IfFalse {
                    else_target,
                    end_target,
                    ..
                } if pc < else_target && else_target <= end_target && end_target <= limit => {}
                Instruction::Jump { target } if pc < target && target <= limit => {}
                Instruction::TemporalEnter { .. } if unit == 0 && pc == p.main.start => {}
                Instruction::TemporalExit { .. } if unit == 0 && pc == exit => {}
                Instruction::Return if pc == range.end - 1 => {}
                other => return Err(format!("unsupported reachable instruction {pc}: {other:?}")),
            }
        }
        colors[unit] = 2;
        Ok(())
    }
    visit(0, program, exit, &mut colors)?;
    Ok(scope)
}

pub fn specialize_input(program: &BytecodeProgram, tape: &[u64]) -> StudyResult<BytecodeProgram> {
    if program.instructions.len() > 4096
        || program.procedures.len() > 64
        || program.source_map.len() > 4096
    {
        return Err("input artifact exceeds study ceiling".into());
    }
    program.validate().map_err(|e| e.to_string())?;
    if tape.len() > 8 {
        return Err("input tape exceeds study ceiling".into());
    }
    let has_input = program.instructions[program.main.start as usize..program.main.end as usize]
        .contains(&Instruction::Primitive(OpCode::Input));
    if !has_input {
        if !tape.is_empty() {
            return Err("unused frozen input".into());
        }
        return Ok(program.clone());
    }
    let mut result = program.clone();
    let mut consumed = 0;
    for pc in program.main.start..program.main.end {
        match program.instructions[pc as usize] {
            Instruction::Primitive(OpCode::Input) => {
                let value = *tape.get(consumed).ok_or("frozen input exhausted")?;
                consumed += 1;
                result.instructions[pc as usize] = Instruction::PushWord(value);
            }
            Instruction::PushWord(_)
            | Instruction::TemporalEnter { .. }
            | Instruction::TemporalExit { .. }
            | Instruction::Return => {}
            Instruction::Primitive(op) if numeric(op) => {}
            _ => return Err("INPUT specialization requires unconditional linear main".into()),
        }
    }
    if consumed != tape.len() {
        return Err("unused frozen input".into());
    }
    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub present: Vec<u64>,
    pub stack: Vec<u64>,
    pub gas: u64,
}

/// Deliberately separate interpreter: no VM, IR, solver or arithmetic helpers.
pub fn interpret(
    p: &BytecodeProgram,
    cfg: &FiniteProofConfig,
    scope: Scope,
    input: &[u64],
) -> StudyResult<Row> {
    if input.len() != cfg.memory_cells {
        return Err("input memory width".into());
    }
    let mut present = vec![0u64; cfg.memory_cells];
    let mut stack = Vec::new();
    let mut frames: Vec<CodeRange> = Vec::new();
    let mut cursor = p.main;
    let mut active = false;
    let mut gas = 0u64;
    fn pop(stack: &mut Vec<u64>) -> StudyResult<u64> {
        stack.pop().ok_or("stack underflow".into())
    }
    loop {
        if gas >= cfg.max_instructions {
            return Err("gas exhausted".into());
        }
        if cursor.start >= cursor.end {
            return Err("invalid cursor".into());
        }
        let record = *p
            .instructions
            .get(cursor.start as usize)
            .ok_or("invalid cursor")?;
        cursor.start += 1;
        gas += 1;
        match record {
            Instruction::PushWord(value) => stack.push(value),
            Instruction::CallProcedure(id) => {
                if frames.len() >= cfg.max_call_depth {
                    return Err("call depth".into());
                }
                frames.push(cursor);
                cursor = p.procedures.get(id.index()).ok_or("call identity")?.range;
            }
            Instruction::IfFalse { else_target, .. } => {
                if pop(&mut stack)? == 0 {
                    cursor.start = else_target;
                }
            }
            Instruction::Jump { target } => cursor.start = target,
            Instruction::TemporalEnter {
                base: 0,
                size,
                cell_bits,
                ..
            } if !active && size == scope.cells as u64 && cell_bits == scope.bits => active = true,
            Instruction::TemporalExit { enter_target }
                if active && enter_target == p.main.start =>
            {
                active = false
            }
            Instruction::Return => {
                if let Some(frame) = frames.pop() {
                    cursor = frame;
                } else {
                    return Ok(Row {
                        present,
                        stack,
                        gas,
                    });
                }
            }
            Instruction::Primitive(op) => match op {
                OpCode::Nop => {}
                OpCode::Pop => {
                    pop(&mut stack)?;
                }
                OpCode::Dup | OpCode::Over => {
                    let back = if op == OpCode::Dup { 1 } else { 2 };
                    let index = stack.len().checked_sub(back).ok_or("stack underflow")?;
                    stack.push(stack[index]);
                }
                OpCode::Swap => {
                    let index = stack.len().checked_sub(2).ok_or("stack underflow")?;
                    stack.swap(index, index + 1);
                }
                OpCode::Not => {
                    let x = pop(&mut stack)?;
                    stack.push(u64::from(x == 0));
                }
                OpCode::Oracle | OpCode::PresentRead | OpCode::Prophecy => {
                    if !active {
                        return Err("memory access outside scope".into());
                    }
                    let raw = pop(&mut stack)?;
                    let address = if raw < scope.cells as u64 {
                        raw as usize
                    } else {
                        match cfg.memory_bounds {
                            BoundsPolicy::Error => return Err("scoped bounds".into()),
                            BoundsPolicy::Wrap => (raw % scope.cells as u64) as usize,
                            BoundsPolicy::Clamp => scope.cells - 1,
                        }
                    };
                    if op == OpCode::Prophecy {
                        present[address] = pop(&mut stack)? & scope.mask();
                    } else {
                        stack.push(if op == OpCode::Oracle {
                            input[address]
                        } else {
                            present[address]
                        });
                    }
                }
                OpCode::Add
                | OpCode::Sub
                | OpCode::Mul
                | OpCode::Div
                | OpCode::Mod
                | OpCode::And
                | OpCode::Or
                | OpCode::Xor
                | OpCode::Shl
                | OpCode::Shr
                | OpCode::Eq
                | OpCode::Neq => {
                    let right = pop(&mut stack)?;
                    let left = pop(&mut stack)?;
                    stack.push(match op {
                        OpCode::Add => left.wrapping_add(right),
                        OpCode::Sub => left.wrapping_sub(right),
                        OpCode::Mul => left.wrapping_mul(right),
                        OpCode::Div => {
                            if right == 0 {
                                0
                            } else {
                                left / right
                            }
                        }
                        OpCode::Mod => {
                            if right == 0 {
                                0
                            } else {
                                left % right
                            }
                        }
                        OpCode::And => left & right,
                        OpCode::Or => left | right,
                        OpCode::Xor => left ^ right,
                        OpCode::Shl => left << (right % 64),
                        OpCode::Shr => left >> (right % 64),
                        OpCode::Eq => u64::from(left == right),
                        OpCode::Neq => u64::from(left != right),
                        _ => unreachable!(),
                    });
                }
                _ => return Err("unsupported interpreter primitive".into()),
            },
            _ => return Err("unsupported interpreter control".into()),
        }
        if stack.len() > cfg.max_stack_depth {
            return Err("stack depth".into());
        }
    }
}

fn vm_row(w: &Workload, memory: &[u64]) -> StudyResult<Row> {
    let mut input = PagedMemory::with_size(w.profile.memory_cells).map_err(|e| e.to_string())?;
    for (cell, &value) in memory.iter().enumerate() {
        input
            .write(cell as u64, Value::new(value))
            .map_err(|e| e.to_string())?;
    }
    let mut config = w.profile.vm_config();
    config.input = w.tape.clone();
    let result = BytecodeVm::with_config(config)
        .run(&w.original, &input)
        .map_err(|e| e.to_string())?;
    if result.status != BytecodeVmStatus::Finished
        || !result.output.is_empty()
        || !result.effects.is_empty()
        || result.inputs_consumed != w.tape
    {
        return Err("non-normal or non-frozen VM observation".into());
    }
    Ok(Row {
        present: (0..w.profile.memory_cells)
            .map(|cell| result.present.get(cell as u64).unwrap().val)
            .collect(),
        stack: result.stack.iter().map(|v| v.val).collect(),
        gas: result.instructions_executed,
    })
}

impl Workload {
    pub fn original(&self) -> &BytecodeProgram {
        &self.original
    }
    pub fn specialized(&self) -> &BytecodeProgram {
        &self.specialized
    }
    pub fn profile(&self) -> &FiniteProofConfig {
        &self.profile
    }
    pub fn scope(&self) -> Scope {
        self.scope
    }
    pub fn fixed(&self) -> &[Vec<u64>] {
        &self.fixed
    }
    pub fn compile(
        name: &str,
        source: &str,
        tape: &[u64],
        profile: FiniteProofConfig,
    ) -> StudyResult<Self> {
        if source.len() > 256 * 1024 || name.len() > 128 {
            return Err("source/label exceeds study ceiling".into());
        }
        let mut parsed = parse(source).map_err(|e| e.to_string())?;
        parsed.procedures.extend(StdLib::procedures());
        let original =
            BytecodeProgram::compile(&HirProgram::resolve(&parsed).map_err(|e| format!("{e:?}"))?)
                .map_err(|e| e.to_string())?;
        Self::from_program(name, original, tape, profile)
    }
    pub fn from_program(
        name: &str,
        original: BytecodeProgram,
        tape: &[u64],
        profile: FiniteProofConfig,
    ) -> StudyResult<Self> {
        let specialized = specialize_input(&original, tape)?;
        let scope = admit(&specialized, &profile)?;
        let solver_config = GlobalSolveConfig {
            memory_cells: profile.memory_cells,
            loop_unroll_limit: 0,
            solver_timeout_ms: TIMEOUT_MS,
            max_instructions: profile.max_instructions,
            bounds_policy: profile.memory_bounds,
        };
        let ir = GlobalFixedPointSolver::compile_bytecode(&specialized, solver_config)
            .map_err(|e| e.to_string())?;
        if ir.completeness != IrCompleteness::Complete {
            return Err("incomplete IR".into());
        }
        let mut smt_base = ir.to_smt2(false);
        // Explicit domain matches the oracle; no inference from sampled words.
        for cell in 0..profile.memory_cells {
            if cell < scope.cells {
                smt_base.push_str(&format!(
                    "(assert (bvule (select anamnesis (_ bv{cell} 64)) (_ bv{} 64)))\n",
                    scope.mask()
                ));
            } else {
                smt_base.push_str(&format!(
                    "(assert (= (select anamnesis (_ bv{cell} 64)) (_ bv0 64)))\n"
                ));
            }
        }
        let mut queries = vec![Query::Exists];
        queries.extend((0..16).map(|value| Query::CellEquals {
            cell: value as usize % scope.cells,
            value,
        }));
        queries.extend((0..15).map(|value| Query::CellNotEquals {
            cell: value as usize % scope.cells,
            value,
        }));
        let mut workload = Self {
            name: name.into(),
            original,
            specialized,
            tape: tape.to_vec(),
            profile,
            scope,
            solver_config,
            smt_base,
            queries,
            fixed: Vec::new(),
            context_fingerprint: [0; 32],
        };
        for row in 0..scope.states() {
            let memory = scope.input(row, workload.profile.memory_cells);
            let independent = interpret(&workload.specialized, &workload.profile, scope, &memory)?;
            let actual = vm_row(&workload, &memory)?;
            if independent != actual {
                return Err(format!("{} oracle/VM row mismatch at {row}", workload.name));
            }
            if actual.present[scope.cells..].iter().any(|&v| v != 0)
                || actual.present[..scope.cells]
                    .iter()
                    .any(|&v| v > scope.mask())
            {
                return Err("finite closure/frame violation".into());
            }
            if actual.present == memory {
                workload.fixed.push(memory);
            }
        }
        workload.context_fingerprint = workload.compute_identity()?;
        Ok(workload)
    }
    pub fn oracle_sat(&self, query: &Query) -> StudyResult<bool> {
        query.validate(self.scope.cells)?;
        Ok(self.fixed.iter().any(|memory| query.holds(memory)))
    }
    pub fn validate_answer(&self, query: &Query, answer: &Answer) -> StudyResult<()> {
        let expected = self.oracle_sat(query)?;
        match answer {
            Answer::Sat(memory) => {
                if !expected
                    || memory.len() != self.profile.memory_cells
                    || memory[..self.scope.cells]
                        .iter()
                        .any(|&v| v > self.scope.mask())
                    || memory[self.scope.cells..].iter().any(|&v| v != 0)
                    || !query.holds(memory)
                    || vm_row(self, memory)?.present != *memory
                {
                    return Err("SAT witness failed finite oracle/domain/original VM replay".into());
                }
            }
            Answer::Unsat if expected => {
                return Err("UNSAT contradicts complete finite oracle".into())
            }
            Answer::Unknown => return Err("Unknown cannot be validated or cached".into()),
            Answer::Unsat => {}
        }
        Ok(())
    }
    pub fn key(&self, query: &Query, semantics: &str) -> StudyResult<[u8; 32]> {
        query.validate(self.scope.cells)?;
        let mut hash = Sha256::new();
        hash.update((semantics.len() as u64).to_le_bytes());
        hash.update(semantics.as_bytes());
        hash.update(self.context_fingerprint);
        let query = query.smt();
        hash.update((query.len() as u64).to_le_bytes());
        hash.update(query.as_bytes());
        Ok(hash.finalize().into())
    }
    fn compute_identity(&self) -> StudyResult<[u8; 32]> {
        let mut hash = Sha256::new();
        fn field(hash: &mut Sha256, bytes: &[u8]) {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        field(&mut hash, STUDY_SEMANTICS.as_bytes());
        field(
            &mut hash,
            &self.original.to_bytes().map_err(|e| e.to_string())?,
        );
        field(
            &mut hash,
            &self.specialized.to_bytes().map_err(|e| e.to_string())?,
        );
        field(
            &mut hash,
            &self
                .tape
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        field(&mut hash, self.smt_base.as_bytes());
        field(
            &mut hash,
            format!(
                "{:?}|{:?}|{:?}",
                self.profile.vm_config(),
                self.solver_config,
                self.scope
            )
            .as_bytes(),
        );
        // Process-local only: exact loaded semantics revision, including the
        // deliberately independent oracle, is frozen into this executable.
        field(&mut hash, include_bytes!("solver_reuse.rs"));
        field(&mut hash, include_bytes!("../src/bytecode_vm.rs"));
        field(&mut hash, include_bytes!("../src/bytecode_temporal.rs"));
        field(&mut hash, include_bytes!("../src/temporal/ir.rs"));
        field(&mut hash, include_bytes!("../src/core/value.rs"));
        field(&mut hash, include_bytes!("../src/finite_proof.rs"));
        Ok(hash.finalize().into())
    }
    fn cvc5_query(&self, query: &Query) -> String {
        format!("{}(assert {})\n", self.smt_base, query.smt())
    }
}

pub fn fixtures() -> StudyResult<Vec<Workload>> {
    let specs: [(&str, &str, &[u64]); 12] = [
        ("identity", "TEMPORAL 0 2 BITS 2 { 0 ORACLE 0 PROPHECY 1 ORACLE 1 PROPHECY }", &[]),
        ("constants", "TEMPORAL 0 2 BITS 2 { 2 0 PROPHECY 3 1 PROPHECY }", &[]),
        ("flip", "TEMPORAL 0 1 BITS 4 { 0 ORACLE 1 XOR 0 PROPHECY }", &[]),
        ("coupled_xor", "TEMPORAL 0 3 BITS 1 { 0 ORACLE 0 PROPHECY 1 ORACLE 1 PROPHECY 0 ORACLE 1 ORACLE XOR 2 PROPHECY }", &[]),
        ("multiply", "TEMPORAL 0 1 BITS 4 { 0 ORACLE 3 MUL 2 ADD 0 PROPHECY }", &[]),
        ("multiply_unsat", "TEMPORAL 0 1 BITS 4 { 0 ORACLE 3 MUL 1 ADD 0 PROPHECY }", &[]),
        ("shift", "TEMPORAL 0 1 BITS 4 { 0 ORACLE 65 SHL 0 PROPHECY }", &[]),
        ("zero_divisor", "TEMPORAL 0 1 BITS 4 { 0 ORACLE 0 DIV 0 PROPHECY }", &[]),
        ("branch", "TEMPORAL 0 1 BITS 2 { 0 ORACLE IF { 3 } ELSE { 1 } 0 PROPHECY }", &[]),
        ("procedure", "PROCEDURE double_read { 0 ORACLE 2 MUL } PROCEDURE write_double { double_read 0 PROPHECY } TEMPORAL 0 1 BITS 8 { write_double }", &[]),
        ("frozen_input_0", "TEMPORAL 0 2 BITS 2 { INPUT 0 PROPHECY 0 ORACLE 1 PROPHECY }", &[0]),
        ("frozen_input_3", "TEMPORAL 0 2 BITS 2 { INPUT 0 PROPHECY 0 ORACLE 1 PROPHECY }", &[3]),
    ];
    specs
        .iter()
        .map(|(name, source, tape)| {
            Workload::compile(name, source, tape, FiniteProofConfig::default())
        })
        .collect()
}

fn z3_config(workload: &Workload) -> Config {
    let mut config = Config::new();
    config.set_model_generation(true);
    config.set_proof_generation(true); // Match the production solver profile.
    config.set_timeout_msec(workload.solver_config.solver_timeout_ms);
    config
}
fn parse_base(solver: &Solver<'_>, workload: &Workload) {
    // Z3 4.8 C API rejects set-logic/set-option in from_string.
    solver.from_string(
        workload
            .smt_base
            .lines()
            .filter(|line| !line.starts_with("(set-"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
fn z3_query(
    context: &Context,
    solver: &Solver<'_>,
    workload: &Workload,
    query: &Query,
) -> StudyResult<Answer> {
    query.validate(workload.scope.cells)?;
    let memory = Array::new_const(
        context,
        "anamnesis",
        &Sort::bitvector(context, 64),
        &Sort::bitvector(context, 64),
    );
    let condition = match *query {
        Query::Exists => z3::ast::Bool::from_bool(context, true),
        Query::CellEquals { cell, value } | Query::CellNotEquals { cell, value } => {
            let selected = memory
                .select(&BV::from_u64(context, cell as u64, 64))
                .as_bv()
                .ok_or("non-word array")?;
            let equal = selected._eq(&BV::from_u64(context, value, 64));
            if matches!(query, Query::CellNotEquals { .. }) {
                equal.not()
            } else {
                equal
            }
        }
    };
    solver.push();
    solver.assert(&condition);
    let answer = match solver.check() {
        SatResult::Sat => {
            let model = solver.get_model().ok_or("SAT without model")?;
            let cells = (0..workload.profile.memory_cells)
                .map(|cell| {
                    model
                        .eval(
                            &memory.select(&BV::from_u64(context, cell as u64, 64)),
                            true,
                        )
                        .and_then(|v| v.as_bv())
                        .and_then(|v| v.as_u64())
                        .ok_or("non-numeral model word".to_string())
                })
                .collect::<StudyResult<Vec<_>>>()?;
            Answer::Sat(cells)
        }
        SatResult::Unsat => Answer::Unsat,
        SatResult::Unknown => Answer::Unknown,
    };
    solver.pop(1);
    Ok(answer)
}

/// Fresh context/parsing with the same push/check/pop protocol as reuse.
pub fn solve_fresh(workload: &Workload, query: &Query) -> StudyResult<Answer> {
    let context = Context::new(&z3_config(workload));
    let solver = Solver::new(&context);
    parse_base(&solver, workload);
    // Share query/decode logic; the fresh arm also enters incremental mode so
    // measured differences are context/parsing/session reuse, not tactic choice.
    z3_query(&context, &solver, workload, query)
}

/// Bounded reused-session conformance entry point for the focused harness.
pub fn solve_reused(workload: &Workload, queries: &[Query]) -> StudyResult<Vec<Answer>> {
    if queries.len() > 32 {
        return Err("reused session query ceiling".into());
    }
    let context = Context::new(&z3_config(workload));
    let solver = Solver::new(&context);
    parse_base(&solver, workload);
    queries
        .iter()
        .map(|query| z3_query(&context, &solver, workload, query))
        .collect()
}

struct Entry {
    key: [u8; 32],
    answer: Answer,
    bytes: usize,
}
pub struct StudyCache {
    entries: VecDeque<Entry>,
    bytes: usize,
    entry_limit: usize,
    byte_limit: usize,
}
impl StudyCache {
    pub fn new(entry_limit: usize, byte_limit: usize) -> StudyResult<Self> {
        if entry_limit == 0
            || entry_limit > MAX_CACHE_ENTRIES
            || byte_limit == 0
            || byte_limit > MAX_CACHE_BYTES
        {
            return Err("cache resource ceiling".into());
        }
        Ok(Self {
            entries: VecDeque::new(),
            bytes: 0,
            entry_limit,
            byte_limit,
        })
    }
    pub fn insert(
        &mut self,
        workload: &Workload,
        query: &Query,
        answer: Answer,
    ) -> StudyResult<()> {
        workload.validate_answer(query, &answer)?; // Validation is mandatory before positive insertion.
        let key = workload.key(query, STUDY_SEMANTICS)?;
        let bytes = 256
            + match &answer {
                Answer::Sat(memory) => memory.len() * 8,
                _ => 0,
            };
        if bytes > self.byte_limit {
            return Err("cache entry exceeds byte limit".into());
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.bytes -= self.entries.remove(index).unwrap().bytes;
        }
        while self.entries.len() >= self.entry_limit || self.bytes + bytes > self.byte_limit {
            self.bytes -= self.entries.pop_front().unwrap().bytes;
        }
        self.entries.push_back(Entry { key, answer, bytes });
        self.bytes += bytes;
        Ok(())
    }
    pub fn get(&self, workload: &Workload, query: &Query) -> StudyResult<Option<&Answer>> {
        let key = workload.key(query, STUDY_SEMANTICS)?;
        Ok(self
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.answer))
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}

// Official cvc5 base Python API; exact SMT text, fresh solver per query. Its
// numeral model is sent back to Rust for domain/original-VM validation.
const CVC5_WORKER: &str = r#"
import sys,time,cvc5
print('VERSION\t'+cvc5.__version__,flush=True)
def evaluate(text,cells):
    begin=time.perf_counter_ns()
    tm=solver=parser=command=result=array=None
    try:
        tm=cvc5.TermManager()
        solver=cvc5.Solver(tm)
        solver.setOption('produce-models','true')
        solver.setOption('tlimit-per','3000')
        # Required by this pinned backend for the IR's constant-zero arrays.
        solver.setOption('arrays-exp','true')
        parser=cvc5.InputParser(solver)
        parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6,bytes.fromhex(text).decode(),'frozen-study')
        while True:
            command=parser.nextCommand()
            if command.isNull(): break
            command.invoke(solver,parser.getSymbolManager())
        result=solver.checkSat()
        words=''
        if result.isSat():
            array=next(term for term in parser.getSymbolManager().getDeclaredTerms() if term.hasSymbol() and term.getSymbol()=='anamnesis')
            words=','.join(solver.getValue(tm.mkTerm(cvc5.Kind.SELECT,array,tm.mkBitVector(64,i))).getBitVectorValue(10) for i in range(int(cells)))
            status='sat'
        elif result.isUnsat(): status='unsat'
        else: status='unknown'
        return status,words,time.perf_counter_ns()-begin
    finally:
        # cvc5 1.3.2 requires manager ownership to outlive parser/terms/solver.
        # Clear dependent objects before creating the next manager, including
        # exception paths and interpreter shutdown; never bypass cleanup.
        array=command=result=None
        parser=None
        solver=None
        tm=None
for line in sys.stdin:
    key,cells,text=line.rstrip('\n').split('\t')
    try:
        status,words,elapsed=evaluate(text,cells)
        print(key+'\t'+status+'\t'+words+'\t'+str(elapsed),flush=True)
    except Exception as error:
        print(key+'\terror\t'+str(error).replace('\t',' ').replace('\n',' ')+'\t0',flush=True)
"#;

fn second_solver(python: &str, workloads: &[Workload]) -> StudyResult<(String, usize, Duration)> {
    let count: usize = workloads.iter().map(|w| w.queries.len()).sum();
    if count > 512 {
        return Err("second-backend query ceiling".into());
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ouro-solver-study-{}-{nonce}.input",
        std::process::id()
    ));
    let mut input = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    let mut bytes = 0usize;
    for workload in workloads {
        for query in &workload.queries {
            let text = workload.cvc5_query(query);
            let line = format!(
                "{}\t{}\t{}\n",
                hex(&workload.key(query, STUDY_SEMANTICS)?),
                workload.profile.memory_cells,
                hex(text.as_bytes())
            );
            bytes += line.len();
            if bytes > 16 * 1024 * 1024 {
                let _ = fs::remove_file(&path);
                return Err("second-backend input ceiling".into());
            }
            input
                .write_all(line.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    drop(input);
    let input = fs::File::open(&path).map_err(|e| e.to_string())?;
    let child = Command::new(python)
        .args(["-c", CVC5_WORKER])
        .stdin(Stdio::from(input))
        .output();
    let _ = fs::remove_file(path);
    let child = child.map_err(|e| e.to_string())?;
    if !child.status.success()
        || child.stdout.len() > 1024 * 1024
        || child.stderr.len() > 1024 * 1024
    {
        return Err(format!(
            "cvc5 worker failed: {}",
            String::from_utf8_lossy(&child.stderr)
        ));
    }
    let output = String::from_utf8(child.stdout).map_err(|e| e.to_string())?;
    let mut lines = output.lines();
    let version = lines
        .next()
        .and_then(|line| line.strip_prefix("VERSION\t"))
        .ok_or("cvc5 version missing")?
        .to_string();
    let mut total = Duration::ZERO;
    let mut checked = 0;
    for workload in workloads {
        for query in &workload.queries {
            let fields: Vec<_> = lines
                .next()
                .ok_or("cvc5 result missing")?
                .split('\t')
                .collect();
            if fields.len() != 4 || fields[0] != hex(&workload.key(query, STUDY_SEMANTICS)?) {
                return Err("cvc5 result binding mismatch".into());
            }
            let answer = match fields[1] {
                "sat" => Answer::Sat(
                    fields[2]
                        .split(',')
                        .map(|word| word.parse::<u64>().map_err(|e| e.to_string()))
                        .collect::<StudyResult<_>>()?,
                ),
                "unsat" => Answer::Unsat,
                "unknown" => Answer::Unknown,
                _ => return Err(format!("cvc5 worker error: {}", fields[2])),
            };
            workload.validate_answer(query, &answer)?;
            total += Duration::from_nanos(fields[3].parse::<u64>().map_err(|e| e.to_string())?);
            checked += 1;
        }
    }
    if lines.next().is_some() {
        return Err("unexpected cvc5 results".into());
    }
    Ok((version, checked, total))
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}
fn milliseconds(time: Duration) -> f64 {
    time.as_secs_f64() * 1000.0
}

fn run() -> StudyResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let python = match args.as_slice() {
        [] => None,
        [flag, path] if flag == "--cvc5-python" => Some(path.as_str()),
        _ => return Err("usage: solver_reuse [--cvc5-python PATH]".into()),
    };
    let start = Instant::now();
    let workloads = fixtures()?;
    let mut binding = Sha256::new();
    for workload in &workloads {
        for query in &workload.queries {
            binding.update(workload.key(query, STUDY_SEMANTICS)?);
        }
    }
    let binding: [u8; 32] = binding.finalize().into();
    println!("profile={STUDY_SEMANTICS} fixtures={} queries_per_fixture=32 repeats={REPEATS} oracle_validation_ms={:.3} z3_crate=0.12.1 timeout_ms={TIMEOUT_MS}", workloads.len(), milliseconds(start.elapsed()));
    println!("workload_sha256={} oracle_rows={} memory_cells=16 bounds=Error gas=1024 stack=256 calls=64 temporal_depth=1 input=exact-frozen-tape capabilities=closed z3_protocol=push-check-pop-both-arms proof_generation=true", hex(&binding), workloads.iter().map(|w| w.scope.states()).sum::<usize>());
    println!("domain=whole-main-mask-at-most-8-bits/fresh-outside-zero trust=solver+exhaustive-independent-numeric-oracle+original-VM-replay cache=study-only");
    let mut cache = StudyCache::new(MAX_CACHE_ENTRIES, MAX_CACHE_BYTES)?;
    let mut totals = vec![[Duration::ZERO; 3]; REPEATS];
    let mut validation = Duration::ZERO;
    let mut cold_cache = Duration::ZERO;
    for workload in &workloads {
        let mut fresh_times = vec![Duration::ZERO; REPEATS];
        let mut reuse_times = vec![Duration::ZERO; REPEATS];
        let mut cache_times = vec![Duration::ZERO; REPEATS];
        for repeat in 0..REPEATS {
            // Alternate order to reduce systematic first-arm warmup bias.
            for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                if arm == 0 {
                    for query in &workload.queries {
                        let start = Instant::now();
                        let answer = solve_fresh(workload, query)?;
                        fresh_times[repeat] += start.elapsed();
                        let check = Instant::now();
                        workload.validate_answer(query, &answer)?;
                        validation += check.elapsed();
                        if repeat == 0 {
                            let insert = Instant::now();
                            cache.insert(workload, query, answer)?;
                            cold_cache += insert.elapsed();
                        }
                    }
                } else {
                    let start = Instant::now();
                    let context = Context::new(&z3_config(workload));
                    let solver = Solver::new(&context);
                    parse_base(&solver, workload);
                    reuse_times[repeat] += start.elapsed();
                    for query in &workload.queries {
                        let start = Instant::now();
                        let answer = z3_query(&context, &solver, workload, query)?;
                        reuse_times[repeat] += start.elapsed();
                        let check = Instant::now();
                        workload.validate_answer(query, &answer)?;
                        validation += check.elapsed();
                    }
                    let cleanup = Instant::now();
                    drop(solver);
                    drop(context);
                    reuse_times[repeat] += cleanup.elapsed();
                }
            }
            for query in &workload.queries {
                let start = Instant::now();
                let answer = cache.get(workload, query)?.ok_or("validated cache miss")?;
                cache_times[repeat] += start.elapsed();
                if *answer == Answer::Unknown {
                    return Err("cached Unknown".into());
                }
            }
            totals[repeat][0] += fresh_times[repeat];
            totals[repeat][1] += reuse_times[repeat];
            totals[repeat][2] += cache_times[repeat];
        }
        println!("fixture={} states={} fixed={} fresh_median_ms={:.3} reused_median_ms={:.3} warm_cache_median_ms={:.3}", workload.name, workload.scope.states(), workload.fixed.len(), milliseconds(median(&mut fresh_times)), milliseconds(median(&mut reuse_times)), milliseconds(median(&mut cache_times)));
    }
    let fresh = median(&mut totals.iter().map(|row| row[0]).collect::<Vec<_>>());
    let reused = median(&mut totals.iter().map(|row| row[1]).collect::<Vec<_>>());
    let cached = median(&mut totals.iter().map(|row| row[2]).collect::<Vec<_>>());
    println!("aggregate fresh_median_ms={:.3} reused_median_ms={:.3} warm_cache_median_ms={:.3} reuse_speedup={:.3} validation_ms={:.3} cold_cache_validation_insert_ms={:.3} cache_entries={} cache_bytes={} unknown=0", milliseconds(fresh), milliseconds(reused), milliseconds(cached), fresh.as_secs_f64()/reused.as_secs_f64(), milliseconds(validation), milliseconds(cold_cache), cache.len(), cache.bytes());
    if let Some(python) = python {
        let (version, checked, elapsed) = second_solver(python, &workloads)?;
        println!("second_solver=cvc5 version={version} arrays-exp=true checked={checked} disagreements=0 unknown=0 fresh_total_ms={:.3} evidence=backend-agreement-plus-finite-oracle-and-original-VM-replay", milliseconds(elapsed));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("solver reuse study failed: {error}");
        std::process::exit(1);
    }
}
