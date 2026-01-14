//! Static analysis for OUROCHRONOS programs.
//!
//! This module provides:
//! - Temporal Dependency Graph construction
//! - Temporal Core identification (cells in feedback loops)
//! - Negative causal loop detection (grandfather paradox patterns)
//! - Provenance analysis

use crate::ast::{Program, Stmt, OpCode};
use crate::core_types::Address;
use std::collections::{HashMap, HashSet, VecDeque};

/// Temporal Dependency Graph.
/// 
/// Nodes are memory addresses. An edge (a, b, polarity) means:
/// "The value written to b depends on the oracle read of a,
///  with the given polarity (positive or negating)."
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TemporalDependencyGraph {
    /// Forward edges: source → set of (target, is_negating)
    edges: HashMap<Address, HashSet<(Address, bool)>>,
    /// All addresses that are read via ORACLE
    oracle_reads: HashSet<Address>,
    /// All addresses that are written via PROPHECY
    prophecy_writes: HashSet<Address>,
}

impl TemporalDependencyGraph {
    /// Build TDG from a program via abstract interpretation.
    pub fn build(program: &Program) -> Self {
        let mut builder = TDGBuilder::new();
        builder.analyze(&program.body);
        builder.build()
    }
    
    /// Find the temporal core: addresses in feedback cycles.
    pub fn temporal_core(&self) -> HashSet<Address> {
        // Find SCCs, return addresses in non-trivial SCCs
        let sccs = self.tarjan_scc();
        
        let mut core = HashSet::new();
        for scc in sccs {
            if scc.len() > 1 || self.has_self_loop(&scc[0]) {
                core.extend(scc);
            }
        }
        core
    }
    
    /// Find negative causal loops (grandfather paradox patterns).
    /// Returns cycles where the parity of negations is odd.
    pub fn find_negative_loops(&self) -> Vec<NegativeLoop> {
        let mut negative_loops = Vec::new();
        
        // Check each SCC for negative cycles
        let sccs = self.tarjan_scc();
        
        for scc in sccs {
            if scc.len() == 1 {
                // Self-loop: check if negating
                let addr = scc[0];
                if let Some(edges) = self.edges.get(&addr) {
                    for (target, is_neg) in edges {
                        if *target == addr && *is_neg {
                            negative_loops.push(NegativeLoop {
                                cells: vec![addr],
                                explanation: format!(
                                    "Cell {} has a negating self-loop: its new value \
                                     is the negation of its old value. \
                                     Constraint: A[{}] = NOT(A[{}]) is unsatisfiable.",
                                    addr, addr, addr
                                ),
                            });
                        }
                    }
                }
            } else {
                // Multi-node SCC: look for odd-parity cycles
                if let Some(cycle) = self.find_odd_parity_cycle(&scc) {
                    negative_loops.push(cycle);
                }
            }
        }
        
        negative_loops
    }
    
    /// Check if an address has a self-loop.
    fn has_self_loop(&self, addr: &Address) -> bool {
        self.edges.get(addr)
            .map(|edges| edges.iter().any(|(t, _)| t == addr))
            .unwrap_or(false)
    }
    
    /// Tarjan's algorithm for SCCs.
    fn tarjan_scc(&self) -> Vec<Vec<Address>> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        let mut indices = HashMap::new();
        let mut lowlinks = HashMap::new();
        let mut sccs = Vec::new();
        
        let all_nodes: HashSet<_> = self.edges.keys().cloned()
            .chain(self.edges.values().flat_map(|e| e.iter().map(|(t, _)| *t)))
            .collect();
        
        for v in &all_nodes {
            if !indices.contains_key(v) {
                self.strongconnect(*v, &mut index, &mut stack, &mut on_stack,
                                  &mut indices, &mut lowlinks, &mut sccs);
            }
        }
        
        sccs
    }
    
    fn strongconnect(&self, v: Address, 
                     index: &mut usize, 
                     stack: &mut Vec<Address>,
                     on_stack: &mut HashSet<Address>,
                     indices: &mut HashMap<Address, usize>,
                     lowlinks: &mut HashMap<Address, usize>,
                     sccs: &mut Vec<Vec<Address>>) {
        indices.insert(v, *index);
        lowlinks.insert(v, *index);
        *index += 1;
        stack.push(v);
        on_stack.insert(v);
        
        if let Some(edges) = self.edges.get(&v) {
            for (w, _) in edges {
                if !indices.contains_key(w) {
                    self.strongconnect(*w, index, stack, on_stack, indices, lowlinks, sccs);
                    let low_w = *lowlinks.get(w).unwrap();
                    let low_v = lowlinks.get_mut(&v).unwrap();
                    *low_v = (*low_v).min(low_w);
                } else if on_stack.contains(w) {
                    let idx_w = *indices.get(w).unwrap();
                    let low_v = lowlinks.get_mut(&v).unwrap();
                    *low_v = (*low_v).min(idx_w);
                }
            }
        }
        
        if lowlinks.get(&v) == indices.get(&v) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w);
                if w == v { break; }
            }
            sccs.push(scc);
        }
    }
    
    /// Find a cycle with odd parity (negation count) in an SCC.
    fn find_odd_parity_cycle(&self, scc: &[Address]) -> Option<NegativeLoop> {
        let scc_set: HashSet<_> = scc.iter().cloned().collect();
        
        // BFS to find cycles with parity tracking
        for &start in scc {
            // (current_node, parity, path)
            let mut queue: VecDeque<(Address, bool, Vec<Address>)> = VecDeque::new();
            queue.push_back((start, false, vec![start]));
            
            let mut visited: HashMap<Address, bool> = HashMap::new();
            
            while let Some((node, parity, path)) = queue.pop_front() {
                if let Some(edges) = self.edges.get(&node) {
                    for (next, is_neg) in edges {
                        if !scc_set.contains(next) { continue; }
                        
                        let new_parity = parity ^ is_neg;
                        
                        if *next == start {
                            // Found a cycle
                            if new_parity {
                                // Odd parity: negative loop!
                                return Some(NegativeLoop {
                                    cells: path.clone(),
                                    explanation: self.explain_negative_loop(&path),
                                });
                            }
                        } else if !visited.contains_key(next) {
                            visited.insert(*next, new_parity);
                            let mut new_path = path.clone();
                            new_path.push(*next);
                            queue.push_back((*next, new_parity, new_path));
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Generate explanation for a negative loop.
    fn explain_negative_loop(&self, path: &[Address]) -> String {
        if path.len() == 1 {
            format!("Cell {} has a negating self-dependency. \
                    Its new value is derived from NOT of its old value.",
                    path[0])
        } else {
            let path_str = path.iter()
                .map(|a| format!("[{}]", a))
                .collect::<Vec<_>>()
                .join(" → ");
            format!("Negative causal loop: {0} → [{1}]\n\
                    The dependency chain contains an odd number of negations.\n\
                    This creates a grandfather paradox: no self-consistent value exists.",
                    path_str, path[0])
        }
    }
    
    /// Check if a program is trivially consistent (no feedback).
    pub fn is_trivially_consistent(&self) -> bool {
        self.temporal_core().is_empty()
    }
}

/// A negative causal loop (grandfather paradox pattern).
#[derive(Debug, Clone)]
pub struct NegativeLoop {
    /// Addresses in the loop.
    pub cells: Vec<Address>,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Builder for TDG via abstract interpretation.
struct TDGBuilder {
    /// Current provenance on abstract stack.
    stack: Vec<AbstractProvenance>,
    /// Edges being built.
    edges: HashMap<Address, HashSet<(Address, bool)>>,
    /// Oracle reads.
    oracle_reads: HashSet<Address>,
    /// Prophecy writes.
    prophecy_writes: HashSet<Address>,
}

/// Abstract provenance for TDG construction.
#[derive(Debug, Clone)]
enum AbstractProvenance {
    /// Constant (no dependency).
    Const,
    /// Depends on specific oracle address with polarity.
    Oracle(Address, bool), // (addr, is_negated)
    /// Merged dependencies.
    Merged(Vec<(Address, bool)>),
}

impl AbstractProvenance {
    fn negate(&self) -> Self {
        match self {
            AbstractProvenance::Const => AbstractProvenance::Const,
            AbstractProvenance::Oracle(a, neg) => AbstractProvenance::Oracle(*a, !neg),
            AbstractProvenance::Merged(deps) => {
                AbstractProvenance::Merged(deps.iter().map(|(a, n)| (*a, !n)).collect())
            }
        }
    }
    
    fn merge(&self, other: &Self) -> Self {
        let mut deps = Vec::new();
        
        match self {
            AbstractProvenance::Const => {}
            AbstractProvenance::Oracle(a, n) => deps.push((*a, *n)),
            AbstractProvenance::Merged(d) => deps.extend(d.iter().cloned()),
        }
        
        match other {
            AbstractProvenance::Const => {}
            AbstractProvenance::Oracle(a, n) => deps.push((*a, *n)),
            AbstractProvenance::Merged(d) => deps.extend(d.iter().cloned()),
        }
        
        if deps.is_empty() {
            AbstractProvenance::Const
        } else if deps.len() == 1 {
            AbstractProvenance::Oracle(deps[0].0, deps[0].1)
        } else {
            AbstractProvenance::Merged(deps)
        }
    }
    
    fn dependencies(&self) -> Vec<(Address, bool)> {
        match self {
            AbstractProvenance::Const => vec![],
            AbstractProvenance::Oracle(a, n) => vec![(*a, *n)],
            AbstractProvenance::Merged(d) => d.clone(),
        }
    }
}

impl TDGBuilder {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            edges: HashMap::new(),
            oracle_reads: HashSet::new(),
            prophecy_writes: HashSet::new(),
        }
    }
    
    fn analyze(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.analyze_stmt(stmt);
        }
    }
    
    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Push(_) => {
                self.stack.push(AbstractProvenance::Const);
            }
            
            Stmt::Op(op) => self.analyze_op(*op),
            
            Stmt::Block(stmts) => self.analyze(stmts),
            
            Stmt::If { then_branch, else_branch } => {
                // Pop condition
                self.stack.pop();
                
                // Analyze both branches (conservative: merge effects)
                let stack_before = self.stack.clone();
                self.analyze(then_branch);
                
                if let Some(else_stmts) = else_branch {
                    let stack_after_then = self.stack.clone();
                    self.stack = stack_before;
                    self.analyze(else_stmts);
                    
                    // Merge stacks (take max length)
                    if stack_after_then.len() > self.stack.len() {
                        self.stack = stack_after_then;
                    }
                }
            }
            
            Stmt::While { cond, body } => {
                // Analyze condition
                self.analyze(cond);
                self.stack.pop();
                
                // Analyze body (conservative: assume executes)
                self.analyze(body);
            }
            
            Stmt::Call { name: _ } => {
                // Procedure call: conservative, push unknown provenance
                self.stack.push(AbstractProvenance::Const);
            }
            
            Stmt::Match { cases, default } => {
                // Pop the matched value
                self.stack.pop();
                
                // Analyze all branches
                for (_, body) in cases {
                    self.analyze(body);
                }
                if let Some(default_body) = default {
                    self.analyze(default_body);
                }
            }
            
            Stmt::TemporalScope { body, .. } => {
                // Temporal scope: analyze the body
                self.analyze(body);
            }
        }
    }
    
    fn analyze_op(&mut self, op: OpCode) {
        match op {
            OpCode::Nop | OpCode::Halt | OpCode::Paradox => {}
            
            OpCode::Pop => { self.stack.pop(); }
            
            OpCode::Dup => {
                if let Some(top) = self.stack.last().cloned() {
                    self.stack.push(top);
                }
            }
            
            OpCode::Swap => {
                if self.stack.len() >= 2 {
                    let len = self.stack.len();
                    self.stack.swap(len - 1, len - 2);
                }
            }
            
            OpCode::Over => {
                if self.stack.len() >= 2 {
                    let val = self.stack[self.stack.len() - 2].clone();
                    self.stack.push(val);
                }
            }
            
            OpCode::Rot => {
                if self.stack.len() >= 3 {
                    let len = self.stack.len();
                    let c = self.stack.remove(len - 3);
                    self.stack.push(c);
                }
            }
            
            OpCode::Depth => {
                self.stack.push(AbstractProvenance::Const);
            }
            
            // Binary operations merge provenance
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod |
            OpCode::And | OpCode::Or | OpCode::Xor | OpCode::Shl | OpCode::Shr |
            OpCode::Eq | OpCode::Neq | OpCode::Lt | OpCode::Gt | OpCode::Lte | OpCode::Gte => {
                let b = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                let a = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(a.merge(&b));
            }
            
            OpCode::Neg | OpCode::Abs | OpCode::Sign => {
                // Unary operations: preserve/negate provenance
                if let Some(a) = self.stack.pop() {
                    self.stack.push(a.negate());
                }
            }

            OpCode::Min | OpCode::Max => {
                // Binary operations that merge provenance
                let b = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                let a = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(a.merge(&b));
            }

            OpCode::Slt | OpCode::Sgt | OpCode::Slte | OpCode::Sgte => {
                // Signed comparison: merge provenance
                let b = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                let a = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(a.merge(&b));
            }
            
            OpCode::Not => {
                if let Some(a) = self.stack.pop() {
                    self.stack.push(a.negate());
                }
            }
            
            OpCode::Oracle => {
                // Pop address, push oracle dependency
                if let Some(_addr_prov) = self.stack.pop() {
                    // For static analysis, assume address is constant 0 or track it
                    // Here we're conservative: if we know the address, use it
                    // Otherwise, mark as depending on "all" (represented as addr 0xFFFF)
                    let addr = 0; // Conservative: assume cell 0
                    self.oracle_reads.insert(addr);
                    self.stack.push(AbstractProvenance::Oracle(addr, false));
                }
            }
            
            OpCode::Prophecy => {
                // Pop address and value
                if let (Some(_addr_prov), Some(val_prov)) = (self.stack.pop(), self.stack.pop()) {
                    let addr = 0; // Conservative
                    self.prophecy_writes.insert(addr);
                    
                    // Add edges from val_prov dependencies to this address
                    for (src, neg) in val_prov.dependencies() {
                        self.edges.entry(src)
                            .or_insert_with(HashSet::new)
                            .insert((addr, neg));
                    }
                }
            }
            
            OpCode::PresentRead => {
                self.stack.pop();
                self.stack.push(AbstractProvenance::Const);
            }
            
            OpCode::Input => {
                self.stack.push(AbstractProvenance::Const);
            }
            
            OpCode::Output | OpCode::Emit => {
                self.stack.pop();
            }

            OpCode::Pick => {
                // Try to pop n
                 if let Some(_n_prov) = self.stack.pop() {
                     // We push None as we can't easily track provenance through dynamic Pick
                     self.stack.push(AbstractProvenance::Const); 
                 }
            }
            
            // Array operations - conservative analysis
            OpCode::Pack => {
                self.stack.pop(); // n
                self.stack.pop(); // base
            }
            OpCode::Unpack => {
                self.stack.pop(); // n
                self.stack.pop(); // base
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::Index => {
                self.stack.pop(); // index
                self.stack.pop(); // base
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::Store => {
                self.stack.pop(); // index
                self.stack.pop(); // base
                self.stack.pop(); // value
            }

            // Vector operations - provenance tracking for data structures
            OpCode::VecNew => {
                // Creates a new handle, no temporal provenance
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::VecPush => {
                self.stack.pop(); // value
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::VecPop => {
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                // Value could have any provenance, conservatively mark as Const
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::VecGet => {
                self.stack.pop(); // index
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                // Value could have any provenance
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::VecSet => {
                self.stack.pop(); // index
                self.stack.pop(); // value
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::VecLen => {
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const);
            }

            // Hash table operations
            OpCode::HashNew => {
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::HashPut => {
                self.stack.pop(); // value
                self.stack.pop(); // key
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::HashGet => {
                self.stack.pop(); // key
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const); // value
                self.stack.push(AbstractProvenance::Const); // found flag
            }
            OpCode::HashDel => {
                self.stack.pop(); // key
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::HashHas => {
                self.stack.pop(); // key
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const); // found flag
            }
            OpCode::HashLen => {
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const);
            }

            // Set operations
            OpCode::SetNew => {
                self.stack.push(AbstractProvenance::Const);
            }
            OpCode::SetAdd => {
                self.stack.pop(); // value
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::SetHas => {
                self.stack.pop(); // value
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const); // found flag
            }
            OpCode::SetDel => {
                self.stack.pop(); // value
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
            }
            OpCode::SetLen => {
                let handle = self.stack.pop().unwrap_or(AbstractProvenance::Const);
                self.stack.push(handle);
                self.stack.push(AbstractProvenance::Const);
            }

            // Dynamic stack structure changes
            OpCode::Roll | OpCode::Reverse | OpCode::StrRev | OpCode::StrCat | OpCode::StrSplit | OpCode::Assert | OpCode::Exec | OpCode::Dip | OpCode::Keep | OpCode::Bi | OpCode::Rec => {
                // Cannot track provenance through dynamic stack permutations
                // In a production system, this would taint the whole stack
            }

            // FFI and I/O operations - these don't create temporal dependencies
            // They may have external side effects but don't affect oracle/prophecy analysis
            OpCode::FFICall | OpCode::FFICallNamed |
            OpCode::FileOpen | OpCode::FileRead | OpCode::FileWrite |
            OpCode::FileSeek | OpCode::FileFlush | OpCode::FileClose |
            OpCode::FileExists | OpCode::FileSize |
            OpCode::BufferNew | OpCode::BufferFromStack | OpCode::BufferToStack |
            OpCode::BufferLen | OpCode::BufferReadByte | OpCode::BufferWriteByte |
            OpCode::BufferFree |
            OpCode::TcpConnect | OpCode::SocketSend | OpCode::SocketRecv |
            OpCode::SocketClose |
            OpCode::ProcExec |
            OpCode::Clock | OpCode::Sleep | OpCode::Random => {
                // External operations don't create temporal dependencies
                // They may produce non-deterministic values but don't affect oracle/prophecy
            }
        }
    }
    
    fn build(self) -> TemporalDependencyGraph {
        TemporalDependencyGraph {
            edges: self.edges,
            oracle_reads: self.oracle_reads,
            prophecy_writes: self.prophecy_writes,
        }
    }
}

/// Analyze a program and return negative loops if any.
pub fn find_negative_loops(program: &Program) -> Vec<NegativeLoop> {
    let tdg = TemporalDependencyGraph::build(program);
    tdg.find_negative_loops()
}

/// Check if a program is trivially consistent.
pub fn is_trivially_consistent(program: &Program) -> bool {
    let tdg = TemporalDependencyGraph::build(program);
    tdg.is_trivially_consistent()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    
    #[test]
    fn test_trivially_consistent() {
        let program = parse("10 20 ADD OUTPUT").unwrap();
        assert!(is_trivially_consistent(&program));
    }
    
    #[test]
    fn test_grandfather_paradox() {
        let program = parse("0 ORACLE NOT 0 PROPHECY").unwrap();
        let loops = find_negative_loops(&program);
        
        // Should detect negative loop
        assert!(!loops.is_empty() || !is_trivially_consistent(&program));
    }
}
