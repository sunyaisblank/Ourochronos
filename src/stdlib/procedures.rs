//! Standard Library for OUROCHRONOS.
//!
//! Provides common operations and utilities for temporal programming.

use crate::ast::{Stmt, OpCode, Procedure, Effect};
use crate::core::Value;

/// Standard library module.
pub struct StdLib;

impl StdLib {
    /// Get all standard library procedures.
    pub fn procedures() -> Vec<Procedure> {
        vec![
            Self::math_procedures(),
            Self::stack_procedures(),
            Self::memory_procedures(),
            Self::io_procedures(),
            Self::string_procedures(),
        ].into_iter().flatten().collect()
    }
    
    /// Mathematical procedures.
    fn math_procedures() -> Vec<Procedure> {
        vec![
            // MIN(a b -- min)
            Procedure {
                name: "MIN".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    // ( a b -- min )
                    Stmt::Op(OpCode::Over),   // a b a
                    Stmt::Op(OpCode::Over),   // a b a b
                    Stmt::Op(OpCode::Gt),     // a b (a>b)
                    Stmt::If {
                        then_branch: vec![
                            Stmt::Op(OpCode::Swap),
                            Stmt::Op(OpCode::Pop),
                        ],
                        else_branch: Some(vec![Stmt::Op(OpCode::Pop)]),
                    },
                ],
            },
            // MAX(a b -- max)
            Procedure {
                name: "MAX".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Over),
                    Stmt::Op(OpCode::Over),
                    Stmt::Op(OpCode::Lt),
                    Stmt::If {
                        then_branch: vec![
                            Stmt::Op(OpCode::Swap),
                            Stmt::Op(OpCode::Pop),
                        ],
                        else_branch: Some(vec![Stmt::Op(OpCode::Pop)]),
                    },
                ],
            },
            // SQUARE(n -- n^2)
            Procedure {
                name: "SQUARE".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Mul),
                ],
            },
            // CUBE(n -- n^3)
            Procedure {
                name: "CUBE".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Mul),
                    Stmt::Op(OpCode::Mul),
                ],
            },
            // DOUBLE(n -- 2n)
            Procedure {
                name: "DOUBLE".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Add),
                ],
            },
            // HALVE(n -- n/2)
            Procedure {
                name: "HALVE".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Push(Value::new(2)),
                    Stmt::Op(OpCode::Div),
                ],
            },
            // INC(n -- n+1)
            Procedure {
                name: "INC".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Push(Value::new(1)),
                    Stmt::Op(OpCode::Add),
                ],
            },
            // DEC(n -- n-1)
            Procedure {
                name: "DEC".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Push(Value::new(1)),
                    Stmt::Op(OpCode::Sub),
                ],
            },
            // EVEN?(n -- bool) Returns 1 if even, 0 if odd
            Procedure {
                name: "EVEN?".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Push(Value::new(2)),
                    Stmt::Op(OpCode::Mod),
                    Stmt::Push(Value::new(0)),
                    Stmt::Op(OpCode::Eq),
                ],
            },
            // ODD?(n -- bool) Returns 1 if odd, 0 if even
            Procedure {
                name: "ODD?".to_string(),
                params: vec!["n".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Push(Value::new(2)),
                    Stmt::Op(OpCode::Mod),
                ],
            },
            // CLAMP(val lo hi -- clamped)
            Procedure {
                name: "CLAMP".to_string(),
                params: vec!["val".to_string(), "lo".to_string(), "hi".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    // Stack: val lo hi
                    Stmt::Op(OpCode::Rot),    // lo hi val
                    Stmt::Op(OpCode::Rot),    // hi val lo
                    Stmt::Op(OpCode::Over),   // hi val lo val
                    Stmt::Op(OpCode::Over),   // hi val lo val lo
                    Stmt::Op(OpCode::Lt),     // hi val lo (val<lo)
                    Stmt::If {
                        // val < lo: return lo
                        then_branch: vec![
                            Stmt::Op(OpCode::Swap),
                            Stmt::Op(OpCode::Pop),
                            Stmt::Op(OpCode::Swap),
                            Stmt::Op(OpCode::Pop),
                        ],
                        // val >= lo: check hi
                        else_branch: Some(vec![
                            Stmt::Op(OpCode::Pop),   // hi val
                            Stmt::Op(OpCode::Over),  // hi val hi
                            Stmt::Op(OpCode::Over),  // hi val hi val
                            Stmt::Op(OpCode::Lt),    // hi val (hi<val)
                            Stmt::If {
                                // hi < val: return hi
                                then_branch: vec![
                                    Stmt::Op(OpCode::Pop),
                                ],
                                // val <= hi: return val
                                else_branch: Some(vec![
                                    Stmt::Op(OpCode::Swap),
                                    Stmt::Op(OpCode::Pop),
                                ]),
                            },
                        ]),
                    },
                ],
            },
        ]
    }
    
    /// Stack manipulation procedures.
    fn stack_procedures() -> Vec<Procedure> {
        vec![
            // NIP(a b -- b)
            Procedure {
                name: "NIP".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Swap),
                    Stmt::Op(OpCode::Pop),
                ],
            },
            // TUCK(a b -- b a b)
            Procedure {
                name: "TUCK".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 3,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Swap),
                    Stmt::Op(OpCode::Over),
                ],
            },
            // 2DUP(a b -- a b a b)
            Procedure {
                name: "2DUP".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 4,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Over),
                    Stmt::Op(OpCode::Over),
                ],
            },
            // 2DROP(a b -- )
            Procedure {
                name: "2DROP".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 0,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Pop),
                    Stmt::Op(OpCode::Pop),
                ],
            },
        ]
    }
    
    /// Memory utility procedures.
    fn memory_procedures() -> Vec<Procedure> {
        vec![
            // ZERO(addr -- ) Clear a memory cell
            Procedure {
                name: "ZERO".to_string(),
                params: vec!["addr".to_string()],
                returns: 0,
                effects: vec![],
                body: vec![
                    Stmt::Push(Value::new(0)),
                    Stmt::Op(OpCode::Prophecy),
                ],
            },
            // INC_MEM(addr -- ) Increment memory cell
            Procedure {
                name: "INC_MEM".to_string(),
                params: vec!["addr".to_string()],
                returns: 0,
                effects: vec![],
                body: vec![
                    Stmt::Op(OpCode::Dup),
                    Stmt::Op(OpCode::Oracle),
                    Stmt::Push(Value::new(1)),
                    Stmt::Op(OpCode::Add),
                    Stmt::Op(OpCode::Prophecy),
                ],
            },
        ]
    }
    
    /// I/O utility procedures.
    fn io_procedures() -> Vec<Procedure> {
        vec![
            // NEWLINE(-- ) Output a newline
            Procedure {
                name: "NEWLINE".to_string(),
                params: vec![],
                returns: 0,
                effects: vec![],
                body: vec![
                    Stmt::Push(Value::new(10)), // ASCII newline
                    Stmt::Op(OpCode::Emit),
                ],
            },
            // SPACE(-- ) Output a space
            Procedure {
                name: "SPACE".to_string(),
                params: vec![],
                returns: 0,
                effects: vec![],
                body: vec![
                    Stmt::Push(Value::new(32)), // ASCII space
                    Stmt::Op(OpCode::Emit),
                ],
            },
        ]
    }
    
    /// String manipulation procedures.
    fn string_procedures() -> Vec<Procedure> {
        vec![
            // STR_LEN(str -- str len)
            Procedure {
                name: "STR_LEN".to_string(),
                params: vec!["str".to_string()],
                returns: 2,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::Dup),
                ],
            },
            // REV(str -- str) - Alias for REVERSE/STR_REV
            Procedure {
                name: "REV".to_string(),
                params: vec!["str".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::StrRev),
                ],
            },
            // CAT(str1 str2 -- str3) - Alias for STR_CAT
            Procedure {
                name: "CAT".to_string(),
                params: vec!["a".to_string(), "b".to_string()],
                returns: 1,
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::StrCat),
                ],
            },
            // SPLIT(str delim -- ... count) - Alias for STR_SPLIT
            Procedure {
                name: "SPLIT".to_string(),
                params: vec!["s".to_string(), "d".to_string()],
                returns: 1, // Variable returns, technically
                effects: vec![Effect::Pure],
                body: vec![
                    Stmt::Op(OpCode::StrSplit),
                ],
            },
        ]
    }
    
    /// Get documentation for all procedures.
    pub fn documentation() -> Vec<(&'static str, &'static str)> {
        vec![
            // Math
            ("MIN", "( a b -- min ) Returns minimum"),
            ("MAX", "( a b -- max ) Returns maximum"),
            ("SQUARE", "( n -- n^2 ) Returns square"),
            ("CUBE", "( n -- n^3 ) Returns cube"),
            ("DOUBLE", "( n -- 2n ) Doubles value"),
            ("HALVE", "( n -- n/2 ) Halves value"),
            ("INC", "( n -- n+1 ) Increments by 1"),
            ("DEC", "( n -- n-1 ) Decrements by 1"),
            ("EVEN?", "( n -- bool ) True if even"),
            ("ODD?", "( n -- bool ) True if odd"),
            ("CLAMP", "( val lo hi -- clamped ) Clamps value to range"),
            // Stack
            ("NIP", "( a b -- b ) Removes second"),
            ("TUCK", "( a b -- b a b ) Tucks top under second"),
            ("2DUP", "( a b -- a b a b ) Duplicates pair"),
            ("2DROP", "( a b -- ) Drops pair"),
            // Memory
            ("ZERO", "( addr -- ) Sets cell to 0"),
            ("INC_MEM", "( addr -- ) Increments cell"),
            // I/O
            ("NEWLINE", "( -- ) Outputs newline"),
            ("SPACE", "( -- ) Outputs space"),
            // String
            ("STR_LEN", "( str -- str len ) Returns string length"),
            ("REV", "( str -- str' ) Reverses string"),
            ("CAT", "( s1 s2 -- s3 ) Concatenates strings"),
            ("SPLIT", "( s d -- ... n ) Splits string by delimiter"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdlib_procedures() {
        let procs = StdLib::procedures();
        assert!(!procs.is_empty());
        assert!(procs.iter().any(|p| p.name == "MIN"));
    }
    
    #[test]
    fn test_stdlib_documentation() {
        let docs = StdLib::documentation();
        assert!(docs.len() >= 10);
    }
}
