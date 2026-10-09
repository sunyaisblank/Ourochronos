//! Independent numeric core semantics for the conformance target.
//!
//! This module imports no Ourochronos code. The grammar admits decimal words,
//! the operations below, structured IF/WHILE, named procedures, and quotations
//! used only as code by EXEC/DIP/KEEP. Code values cannot escape as numeric
//! results. It excludes source sugar, imports, explicit TEMPORAL regions,
//! provenance, heap/host operations, and proof/selection policies.
//!
//! A machine configuration is (pending continuations, operand stack, immutable
//! anamnesis A, initially zero present P, frozen INPUT and RANDOM tapes/cursors,
//! observations). Optional write masking models the separately declared whole
//! memory profile used by extraction tests, without parsing production scopes.
//! Each `step` consumes one continuation. A sequence schedules its next node;
//! IF schedules one branch after popping its condition; WHILE schedules its
//! condition, a decision, then its body and repetition. Calls schedule an
//! explicit return continuation. No guest call uses Rust recursion. Fuel counts
//! these source transitions, **not** fetched production bytecode instructions.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bounds {
    Error,
    Wrap,
    Clamp,
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub anamnesis: Vec<u64>,
    pub input: Vec<u64>,
    /// A complete frozen RANDOM tape; no live or iid source is implied.
    pub random: Vec<u64>,
    /// Optional finite-profile mask applied to every present-memory write.
    /// Tests serialize the corresponding whole-main scope separately.
    pub write_mask: Option<u64>,
    pub bounds: Bounds,
    pub steps: usize,
    pub stack: usize,
    pub calls: usize,
    pub output: usize,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            anamnesis: vec![0; 4],
            input: vec![],
            random: vec![],
            write_mask: None,
            bounds: Bounds::Error,
            steps: 50_000,
            stack: 512,
            calls: 128,
            output: 512,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation {
    Number(u64),
    Byte(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Finished,
    Halted,
    Paradox,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub stack: Vec<u64>,
    pub present: Vec<u64>,
    pub output: Vec<Observation>,
    pub consumed: Vec<u64>,
    pub stop: Stop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Epoch {
    pub snapshot: Snapshot,
    pub steps: usize,
    pub random_consumed: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    Underflow,
    Address { address: u64, width: usize },
    InputExhausted { consumed: usize },
    RandomExhausted { consumed: usize },
    InvalidCode,
    StackLimit,
    CallLimit,
    OutputLimit,
    StepLimit,
    OutsideCore(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    And,
    Or,
    Xor,
    Left,
    Right,
    Equal,
    Unequal,
    Less,
    Greater,
    AtMost,
    AtLeast,
    SignedLess,
    SignedGreater,
    SignedAtMost,
    SignedAtLeast,
    Min,
    Max,
}

impl Binary {
    fn apply(self, a: u64, b: u64) -> u64 {
        match self {
            Self::Add => a.wrapping_add(b),
            Self::Subtract => a.wrapping_sub(b),
            Self::Multiply => a.wrapping_mul(b),
            Self::Divide => {
                if b == 0 {
                    0
                } else {
                    a / b
                }
            }
            Self::Remainder => {
                if b == 0 {
                    0
                } else {
                    a % b
                }
            }
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            Self::Left => a << (b % 64),
            Self::Right => a >> (b % 64),
            Self::Equal => u64::from(a == b),
            Self::Unequal => u64::from(a != b),
            Self::Less => u64::from(a < b),
            Self::Greater => u64::from(a > b),
            Self::AtMost => u64::from(a <= b),
            Self::AtLeast => u64::from(a >= b),
            Self::SignedLess => u64::from((a as i64) < (b as i64)),
            Self::SignedGreater => u64::from((a as i64) > (b as i64)),
            Self::SignedAtMost => u64::from((a as i64) <= (b as i64)),
            Self::SignedAtLeast => u64::from((a as i64) >= (b as i64)),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unary {
    Negate,
    Abs,
    Sign,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackOp {
    Drop,
    Duplicate,
    Swap,
    Over,
    Rotate,
    Depth,
    Pick,
    Roll,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryOp {
    Oracle,
    Prophecy,
    Present,
    Index,
    Store,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeOp {
    Exec,
    Dip,
    Keep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Number(u64),
    Binary(Binary),
    Unary(Unary),
    Stack(StackOp),
    Memory(MemoryOp),
    Code(CodeOp),
    Quote(Vec<Node>),
    If(Vec<Node>, Vec<Node>),
    While(Vec<Node>, Vec<Node>),
    Call(String),
    Input,
    Random,
    Output,
    Emit,
    Nop,
    Halt,
    Paradox,
}

#[derive(Debug, Clone)]
pub struct Program {
    body: Vec<Node>,
    procedures: BTreeMap<String, Vec<Node>>,
}

/// This lexer/parser is deliberately separate from the production frontend.
/// Keywords are exact uppercase; names are ASCII identifiers; literals decimal.
pub fn parse(source: &str) -> Result<Program, String> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut comment = false;
    for character in source.chars() {
        if comment {
            if character == '\n' {
                comment = false;
            }
            continue;
        }
        if character == '#' || character.is_whitespace() || "{}[]".contains(character) {
            if !word.is_empty() {
                tokens.push(std::mem::take(&mut word));
            }
            if character == '#' {
                comment = true;
            } else if "{}[]".contains(character) {
                tokens.push(character.to_string());
            }
        } else if character.is_ascii_alphanumeric() || character == '_' {
            word.push(character);
        } else {
            return Err(format!(
                "character outside the declared core: {character:?}"
            ));
        }
    }
    if !word.is_empty() {
        tokens.push(word);
    }
    let mut parser = Parser { tokens, cursor: 0 };
    let mut program = Program {
        body: vec![],
        procedures: BTreeMap::new(),
    };
    while let Some(token) = parser.peek() {
        if token == "PROCEDURE" {
            parser.next()?;
            let name = parser.next()?;
            if parser.peek() == Some("PURE") || parser.peek() == Some("TEMPORAL") {
                parser.next()?;
            }
            let block = parser.block("{", "}")?;
            if program.procedures.insert(name.clone(), block).is_some() {
                return Err(format!("duplicate procedure {name}"));
            }
        } else {
            program.body.push(parser.node()?);
        }
    }
    Ok(program)
}

struct Parser {
    tokens: Vec<String>,
    cursor: usize,
}

impl Parser {
    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.cursor).map(String::as_str)
    }

    fn next(&mut self) -> Result<String, String> {
        let token = self
            .tokens
            .get(self.cursor)
            .cloned()
            .ok_or("unexpected end of core source")?;
        self.cursor += 1;
        Ok(token)
    }

    fn block(&mut self, open: &str, close: &str) -> Result<Vec<Node>, String> {
        if self.next()? != open {
            return Err(format!("expected {open}"));
        }
        let mut block = Vec::new();
        while self.peek() != Some(close) {
            if self.peek().is_none() {
                return Err(format!("missing {close}"));
            }
            block.push(self.node()?);
        }
        self.next()?;
        Ok(block)
    }

    fn node(&mut self) -> Result<Node, String> {
        use Binary as B;
        use CodeOp as C;
        use MemoryOp as M;
        use StackOp as S;
        use Unary as U;
        let token = self.next()?;
        if token.bytes().all(|byte| byte.is_ascii_digit()) {
            return token
                .parse()
                .map(Node::Number)
                .map_err(|_| "word exceeds u64".into());
        }
        Ok(match token.as_str() {
            "ADD" => Node::Binary(B::Add),
            "SUB" => Node::Binary(B::Subtract),
            "MUL" => Node::Binary(B::Multiply),
            "DIV" => Node::Binary(B::Divide),
            "MOD" => Node::Binary(B::Remainder),
            "AND" => Node::Binary(B::And),
            "OR" => Node::Binary(B::Or),
            "XOR" => Node::Binary(B::Xor),
            "SHL" => Node::Binary(B::Left),
            "SHR" => Node::Binary(B::Right),
            "EQ" => Node::Binary(B::Equal),
            "NEQ" => Node::Binary(B::Unequal),
            "LT" => Node::Binary(B::Less),
            "GT" => Node::Binary(B::Greater),
            "LTE" => Node::Binary(B::AtMost),
            "GTE" => Node::Binary(B::AtLeast),
            "SLT" => Node::Binary(B::SignedLess),
            "SGT" => Node::Binary(B::SignedGreater),
            "SLTE" => Node::Binary(B::SignedAtMost),
            "SGTE" => Node::Binary(B::SignedAtLeast),
            "MIN" => Node::Binary(B::Min),
            "MAX" => Node::Binary(B::Max),
            "NEG" => Node::Unary(U::Negate),
            "ABS" => Node::Unary(U::Abs),
            "SIGN" => Node::Unary(U::Sign),
            "NOT" => Node::Unary(U::Not),
            "POP" => Node::Stack(S::Drop),
            "DUP" => Node::Stack(S::Duplicate),
            "SWAP" => Node::Stack(S::Swap),
            "OVER" => Node::Stack(S::Over),
            "ROT" => Node::Stack(S::Rotate),
            "DEPTH" => Node::Stack(S::Depth),
            "PICK" => Node::Stack(S::Pick),
            "ROLL" => Node::Stack(S::Roll),
            "REVERSE" => Node::Stack(S::Reverse),
            "ORACLE" => Node::Memory(M::Oracle),
            "PROPHECY" => Node::Memory(M::Prophecy),
            "PRESENT" => Node::Memory(M::Present),
            "INDEX" => Node::Memory(M::Index),
            "STORE" => Node::Memory(M::Store),
            "EXEC" => Node::Code(C::Exec),
            "DIP" => Node::Code(C::Dip),
            "KEEP" => Node::Code(C::Keep),
            "INPUT" => Node::Input,
            "RANDOM" => Node::Random,
            "OUTPUT" => Node::Output,
            "EMIT" => Node::Emit,
            "NOP" => Node::Nop,
            "HALT" => Node::Halt,
            "PARADOX" => Node::Paradox,
            "[" => {
                self.cursor -= 1;
                Node::Quote(self.block("[", "]")?)
            }
            "IF" => {
                let yes = self.block("{", "}")?;
                let no = if self.peek() == Some("ELSE") {
                    self.next()?;
                    self.block("{", "}")?
                } else {
                    vec![]
                };
                Node::If(yes, no)
            }
            "WHILE" => Node::While(self.block("{", "}")?, self.block("{", "}")?),
            name if name.as_bytes()[0].is_ascii_lowercase() => Node::Call(name.into()),
            _ => return Err(format!("token outside the declared core: {token}")),
        })
    }
}

#[derive(Debug, Clone)]
enum Datum<'a> {
    Word(u64),
    Code(&'a [Node]),
}

enum Continuation<'a> {
    Sequence(std::slice::Iter<'a, Node>),
    LoopDecision {
        condition: &'a [Node],
        body: &'a [Node],
    },
    Repeat {
        condition: &'a [Node],
        body: &'a [Node],
    },
    Return {
        restore: Option<Datum<'a>>,
    },
}

struct Machine<'a> {
    program: &'a Program,
    environment: &'a Environment,
    pending: Vec<Continuation<'a>>,
    stack: Vec<Datum<'a>>,
    present: Vec<u64>,
    output: Vec<Observation>,
    consumed: Vec<u64>,
    random_consumed: Vec<u64>,
    call_depth: usize,
    steps: usize,
    stop: Option<Stop>,
}

pub fn evaluate(program: &Program, environment: &Environment) -> Result<Epoch, Fault> {
    if environment.anamnesis.is_empty() {
        return Err(Fault::OutsideCore("zero-width memory"));
    }
    let mut machine = Machine {
        program,
        environment,
        pending: vec![Continuation::Sequence(program.body.iter())],
        stack: vec![],
        present: vec![0; environment.anamnesis.len()],
        output: vec![],
        consumed: vec![],
        random_consumed: vec![],
        call_depth: 0,
        steps: 0,
        stop: None,
    };
    while !machine.pending.is_empty() && machine.stop.is_none() {
        if machine.steps == environment.steps {
            return Err(Fault::StepLimit);
        }
        machine.steps += 1;
        machine.step()?;
    }
    let stack = machine
        .stack
        .iter()
        .map(|datum| match datum {
            Datum::Word(word) => Ok(*word),
            Datum::Code(_) => Err(Fault::OutsideCore("code escaping as a numeric result")),
        })
        .collect::<Result<_, _>>()?;
    Ok(Epoch {
        snapshot: Snapshot {
            stack,
            present: machine.present,
            output: machine.output,
            consumed: machine.consumed,
            stop: machine.stop.unwrap_or(Stop::Finished),
        },
        steps: machine.steps,
        random_consumed: machine.random_consumed,
    })
}

impl<'a> Machine<'a> {
    fn push(&mut self, value: Datum<'a>) -> Result<(), Fault> {
        if self.stack.len() == self.environment.stack {
            return Err(Fault::StackLimit);
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<Datum<'a>, Fault> {
        self.stack.pop().ok_or(Fault::Underflow)
    }

    fn word(&mut self) -> Result<u64, Fault> {
        match self.pop()? {
            Datum::Word(word) => Ok(word),
            Datum::Code(_) => Err(Fault::OutsideCore("code used as a word")),
        }
    }

    fn address(&self, address: u64) -> Result<usize, Fault> {
        let width = self.present.len();
        if address < width as u64 {
            return Ok(address as usize);
        }
        match self.environment.bounds {
            Bounds::Wrap => Ok((address % width as u64) as usize),
            Bounds::Clamp => Ok(width - 1),
            Bounds::Error => Err(Fault::Address { address, width }),
        }
    }

    fn enter(&mut self, code: &'a [Node], restore: Option<Datum<'a>>) -> Result<(), Fault> {
        if self.call_depth == self.environment.calls {
            return Err(Fault::CallLimit);
        }
        self.call_depth += 1;
        self.pending.push(Continuation::Return { restore });
        self.pending.push(Continuation::Sequence(code.iter()));
        Ok(())
    }

    fn loop_condition(&mut self, condition: &'a [Node], body: &'a [Node]) {
        self.pending
            .push(Continuation::LoopDecision { condition, body });
        self.pending.push(Continuation::Sequence(condition.iter()));
    }

    fn step(&mut self) -> Result<(), Fault> {
        match self.pending.pop().expect("a scheduled transition") {
            Continuation::Sequence(mut sequence) => {
                if let Some(node) = sequence.next() {
                    self.pending.push(Continuation::Sequence(sequence));
                    self.node(node)?;
                }
            }
            Continuation::LoopDecision { condition, body } => {
                if self.word()? != 0 {
                    self.pending.push(Continuation::Repeat { condition, body });
                    self.pending.push(Continuation::Sequence(body.iter()));
                }
            }
            Continuation::Repeat { condition, body } => self.loop_condition(condition, body),
            Continuation::Return { restore } => {
                self.call_depth -= 1;
                if let Some(value) = restore {
                    self.push(value)?;
                }
            }
        }
        Ok(())
    }

    fn node(&mut self, node: &'a Node) -> Result<(), Fault> {
        match node {
            Node::Number(word) => self.push(Datum::Word(*word))?,
            Node::Binary(operation) => {
                let right = self.word()?;
                let left = self.word()?;
                self.push(Datum::Word(operation.apply(left, right)))?;
            }
            Node::Unary(operation) => {
                let word = self.word()?;
                let result = match operation {
                    Unary::Negate => word.wrapping_neg(),
                    Unary::Abs => (word as i64).wrapping_abs() as u64,
                    Unary::Sign => (word as i64).signum() as u64,
                    Unary::Not => u64::from(word == 0),
                };
                self.push(Datum::Word(result))?;
            }
            Node::Stack(operation) => self.stack_operation(*operation)?,
            Node::Memory(operation) => self.memory_operation(*operation)?,
            Node::Quote(block) => self.push(Datum::Code(block))?,
            Node::Code(operation) => {
                let Datum::Code(code) = self.pop()? else {
                    return Err(Fault::InvalidCode);
                };
                let restore = match operation {
                    CodeOp::Exec => None,
                    CodeOp::Dip => Some(self.pop()?),
                    CodeOp::Keep => {
                        let value = self.stack.last().cloned().ok_or(Fault::Underflow)?;
                        Some(value)
                    }
                };
                self.enter(code, restore)?;
            }
            Node::If(yes, no) => {
                let branch = if self.word()? == 0 { no } else { yes };
                self.pending.push(Continuation::Sequence(branch.iter()));
            }
            Node::While(condition, body) => self.loop_condition(condition, body),
            Node::Call(name) => {
                let body = self
                    .program
                    .procedures
                    .get(name)
                    .ok_or(Fault::OutsideCore("undefined procedure"))?;
                self.enter(body, None)?;
            }
            Node::Input => {
                let cursor = self.consumed.len();
                let word = *self
                    .environment
                    .input
                    .get(cursor)
                    .ok_or(Fault::InputExhausted { consumed: cursor })?;
                self.consumed.push(word);
                self.push(Datum::Word(word))?;
            }
            Node::Random => {
                let cursor = self.random_consumed.len();
                let word = *self
                    .environment
                    .random
                    .get(cursor)
                    .ok_or(Fault::RandomExhausted { consumed: cursor })?;
                self.random_consumed.push(word);
                self.push(Datum::Word(word))?;
            }
            Node::Output | Node::Emit => {
                let word = self.word()?;
                if self.output.len() == self.environment.output {
                    return Err(Fault::OutputLimit);
                }
                self.output.push(if matches!(node, Node::Output) {
                    Observation::Number(word)
                } else {
                    Observation::Byte((word % 256) as u8)
                });
            }
            Node::Nop => {}
            Node::Halt => self.stop = Some(Stop::Halted),
            Node::Paradox => self.stop = Some(Stop::Paradox),
        }
        Ok(())
    }

    fn stack_operation(&mut self, operation: StackOp) -> Result<(), Fault> {
        use StackOp as S;
        match operation {
            S::Drop => {
                self.pop()?;
            }
            S::Duplicate => {
                let value = self.stack.last().cloned().ok_or(Fault::Underflow)?;
                self.push(value)?;
            }
            S::Swap | S::Over => {
                let length = self.stack.len();
                if length < 2 {
                    return Err(Fault::Underflow);
                }
                if operation == S::Swap {
                    self.stack[length - 2..].reverse();
                } else {
                    self.push(self.stack[length - 2].clone())?;
                }
            }
            S::Rotate => {
                let length = self.stack.len();
                if length < 3 {
                    return Err(Fault::Underflow);
                }
                self.stack[length - 3..].rotate_left(1);
            }
            S::Depth => self.push(Datum::Word(self.stack.len() as u64))?,
            S::Pick | S::Roll => {
                let depth = self.word()?;
                if depth >= self.stack.len() as u64 {
                    return Err(Fault::Underflow);
                }
                let index = self.stack.len() - 1 - depth as usize;
                let value = if operation == S::Pick {
                    self.stack[index].clone()
                } else {
                    self.stack.remove(index)
                };
                self.push(value)?;
            }
            S::Reverse => {
                let count = self.word()?;
                if count > self.stack.len() as u64 {
                    return Err(Fault::Underflow);
                }
                let start = self.stack.len() - count as usize;
                self.stack[start..].reverse();
            }
        }
        Ok(())
    }

    fn memory_operation(&mut self, operation: MemoryOp) -> Result<(), Fault> {
        use MemoryOp as M;
        match operation {
            M::Oracle | M::Present => {
                let raw = self.word()?;
                let address = self.address(raw)?;
                let value = if operation == M::Oracle {
                    self.environment.anamnesis[address]
                } else {
                    self.present[address]
                };
                self.push(Datum::Word(value))?;
            }
            M::Prophecy => {
                let raw = self.word()?;
                let value = self.word()?;
                let address = self.address(raw)?;
                self.present[address] = value & self.environment.write_mask.unwrap_or(u64::MAX);
            }
            M::Index | M::Store => {
                let offset = self.word()?;
                let base = self.word()?;
                let address = self.address(base.wrapping_add(offset))?;
                if operation == M::Index {
                    self.push(Datum::Word(self.present[address]))?;
                } else {
                    self.present[address] =
                        self.word()? & self.environment.write_mask.unwrap_or(u64::MAX);
                }
            }
        }
        Ok(())
    }
}
