#[cfg(test)]
mod suite {
    use crate::*;

    fn parse(code: &str) -> Program {
        let tokens = tokenize(code);
        let mut parser = Parser::new(&tokens);
        parser.parse_program().expect("Failed to parse program")
    }

    fn default_config() -> Config {
        Config {
            max_epochs: 100,
            mode: ExecutionMode::Standard,
            seed: 0,
            verbose: false,
            frozen_inputs: Vec::new(),
            max_instructions: 10_000_000,
            ..Default::default()
        }
    }

    fn action_config() -> Config {
        Config {
            max_epochs: 100,
            mode: ExecutionMode::ActionGuided {
                config: ActionConfig::anti_trivial(),
                num_seeds: 4,
            },
            seed: 0,
            verbose: false,
            frozen_inputs: Vec::new(),
            max_instructions: 10_000_000,
            ..Default::default()
        }
    }

    #[test]
    fn test_trivial_consistency() {
        let program = parse("10 20 ADD OUTPUT");
        let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
        assert!(matches!(result, ConvergenceStatus::Consistent { epochs: 1, .. }));
    }
    
    #[test]
    fn test_self_fulfilling_prophecy() {
        let program = parse("0 ORACLE 0 PROPHECY");
        let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
        assert!(matches!(result, ConvergenceStatus::Consistent { .. }));
    }
    
    #[test]
    fn test_grandfather_paradox() {
        let program = parse("0 ORACLE NOT 0 PROPHECY");
        let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
        if let ConvergenceStatus::Oscillation { period, .. } = result {
            assert_eq!(period, 2);
        } else {
            panic!("Expected Oscillation, got {:?}", result);
        }
    }
    
    #[test]
    fn test_divergence() {
        let program = parse("0 ORACLE 1 ADD 0 PROPHECY");
        let result = TimeLoop::new(Config { max_epochs: 50, ..default_config() }).expect("valid configuration").run(&program);
        assert!(matches!(result, ConvergenceStatus::Timeout { .. } | ConvergenceStatus::Divergence { .. }));
    }
    
    #[test]
    fn test_witness_pattern_primality() {
        let program = parse("0 ORACLE DUP 3 EQ IF { DUP 0 PROPHECY } ELSE { POP 3 0 PROPHECY }");
        let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
        
        match result {
            ConvergenceStatus::Consistent { memory, epochs, .. } => {
                assert_eq!(memory.read(0).val, 3);
                assert!(epochs <= 2);
            }
            _ => panic!("Expected consistent execution, got {:?}", result),
        }
    }

    #[test]
    fn test_smt_generation_smoke() {
        let program = parse("0 ORACLE DUP 1 ADD 0 PROPHECY");
        let mut encoder = SmtEncoder::new();
        let smt = encoder.encode(&program).expect("within fragment");
        assert!(smt.contains("(declare-const anamnesis"));
        assert!(smt.contains("(check-sat)"));
    }
    
    #[test]
    fn test_smt_control_flow() {
        let program = parse("1 IF { 2 } ELSE { 3 } 0 PROPHECY");
        let mut encoder = SmtEncoder::new();
        let smt = encoder.encode(&program).expect("within fragment");
        
        assert!(smt.contains("ite"), "SMT output missing 'ite': {}", smt);
    }
     
    #[test]
    fn test_div_by_zero() {
        let program = parse("10 0 DIV 0 PROPHECY");
         let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
         if let ConvergenceStatus::Consistent { memory, .. } = result {
             assert_eq!(memory.read(0).val, 0);
         } else {
             panic!("Expected consistency");
         }
    }

    #[test]
    fn test_new_opcodes() {
        let program = parse("1 2 3 ROT DEPTH 0 PROPHECY");
        let result = TimeLoop::new(default_config()).expect("valid configuration").run(&program);
         if let ConvergenceStatus::Consistent { memory, .. } = result {
             assert_eq!(memory.read(0).val, 3);
         } else {
             panic!("Expected consistency");
         }
    }

    #[test]
    fn test_grandfather_paradox_diagnosis() {
        let program = parse("0 ORACLE NOT 0 PROPHECY");
        let config = Config {
            mode: ExecutionMode::Diagnostic,
            max_epochs: 100,
            seed: 0,
            verbose: false,
            frozen_inputs: Vec::new(),
            max_instructions: 10_000_000,
            ..Default::default()
        };
        let result = TimeLoop::new(config).expect("valid configuration").run(&program);
        
        // Diagnostic mode in new timeloop returns Oscillation with Diagnosis
        if let ConvergenceStatus::Oscillation { diagnosis, .. } = result {
             match diagnosis {
                 crate::temporal::timeloop::ParadoxDiagnosis::NegativeLoop { .. } => {
                     // Pass
                 },
                 _ => panic!("Expected NegativeLoop diagnosis"),
             }
        } else {
             panic!("Expected Oscillation with diagnosis, got {:?}", result);
        }
    }
    
    // ========================================================================
    // Action-Guided Execution Tests
    // ========================================================================
    
    #[test]
    fn test_action_guided_finds_fixed_point() {
        // Simple self-fulfilling prophecy should work with action-guided mode
        let program = parse("0 ORACLE 0 PROPHECY");
        let result = TimeLoop::new(action_config()).expect("valid configuration").run(&program);
        assert!(matches!(result, ConvergenceStatus::Consistent { .. }));
    }
    
    #[test]
    fn test_action_guided_prefers_output() {
        // Action-guided mode should prefer the fixed point that produces output
        // Program: reads oracle, if non-zero outputs it and stabilizes
        let program = parse("0 ORACLE DUP 0 EQ NOT IF { DUP OUTPUT DUP 0 PROPHECY } ELSE { 1 0 PROPHECY }");
        let result = TimeLoop::new(action_config()).expect("valid configuration").run(&program);
        
        match result {
            ConvergenceStatus::Consistent { output, .. } => {
                // Should have produced output
                assert!(!output.is_empty(), "Action-guided should prefer output-producing fixed point");
            }
            _ => panic!("Expected consistent execution, got {:?}", result),
        }
    }
    
    #[test]
    fn test_action_guided_factor_witness() {
        // Test that action-guided mode finds factors
        // Program: ask oracle for factor of 15, verify, output if correct
        let program = parse(r#"
            0 ORACLE
            DUP 1 GT IF {
                DUP 15 LT IF {
                    DUP 15 SWAP MOD 0 EQ IF {
                        DUP 0 PROPHECY
                        OUTPUT
                    } ELSE {
                        1 ADD 0 PROPHECY
                    }
                } ELSE {
                    1 ADD 0 PROPHECY
                }
            } ELSE {
                1 ADD 0 PROPHECY
            }
        "#);
        
        let result = TimeLoop::new(action_config()).expect("valid configuration").run(&program);
        
        match result {
            ConvergenceStatus::Consistent { memory, output, .. } => {
                let factor = memory.read(0).val;
                // Factor should divide 15
                assert!(factor > 1 && factor < 15, "Factor {} should be > 1 and < 15", factor);
                assert_eq!(15 % factor, 0, "Factor {} should divide 15", factor);
                // Should have produced output
                assert!(!output.is_empty(), "Should output the factor");
            }
            _ => panic!("Expected consistent execution, got {:?}", result),
        }
    }
    
    #[test]
    fn test_action_mode_enum_equality() {
        // Test that ActionGuided mode can be compared
        let mode1 = ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds: 4,
        };
        let mode2 = ExecutionMode::ActionGuided {
            config: ActionConfig::anti_trivial(),
            num_seeds: 4,
        };
        assert_eq!(mode1, mode2);
    }
}

#[cfg(test)]
mod temporal_op_count {
    use crate::parser::parse;

    #[test]
    fn counts_oracle_and_prophecy_across_body_quotes_and_branches() {
        let program = parse("0 ORACLE 0 PROPHECY [ 1 ORACLE POP ] EXEC 1 IF { 2 ORACLE POP }").unwrap();
        assert_eq!(program.temporal_op_count(), 4);
    }

    #[test]
    fn pure_programme_has_zero_temporal_core() {
        let program = parse("1 2 ADD OUTPUT").unwrap();
        assert_eq!(program.temporal_op_count(), 0);
    }
}

#[cfg(test)]
mod opcode_registration {
    use crate::ast::{OpCode, Stmt};
    use crate::parser::{tokenize, Parser};

    fn contains_op(stmts: &[Stmt], expect: OpCode) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::Op(op) => *op == expect,
            Stmt::If { then_branch, else_branch } => {
                contains_op(then_branch, expect)
                    || else_branch.as_deref().is_some_and(|e| contains_op(e, expect))
            }
            Stmt::While { cond, body } => contains_op(cond, expect) || contains_op(body, expect),
            Stmt::Block(inner) => contains_op(inner, expect),
            Stmt::TemporalScope { body, .. } => contains_op(body, expect),
            Stmt::Push(_) | Stmt::Call { .. } => false,
        })
    }

    /// Every opcode's keyword must tokenise and parse back to that opcode.
    ///
    /// This is the registration-closure gate: INDEX/STORE/PACK/UNPACK once
    /// existed in the enum and the VM for two releases while no keyword
    /// reached them. The exhaustive OpCode::name() match forces a display
    /// arm for a new variant; this test forces the parser registration.
    #[test]
    fn every_opcode_keyword_parses_back_to_its_opcode() {
        let mut failures: Vec<String> = Vec::new();

        for &op in OpCode::ALL.iter() {
            // Enough literal operands to satisfy any static arity check;
            // opcodes with special surface syntax get their own template.
            let source = match op {
                OpCode::Assert => "1 ASSERT \"invariant\"".to_string(),
                _ => format!("0 0 0 0 0 0 0 0 {}", op.name()),
            };
            let tokens = tokenize(&source);
            let mut parser = Parser::new(&tokens);
            match parser.parse_program() {
                Ok(program) => {
                    if !contains_op(&program.body, op) {
                        failures.push(format!(
                            "{} parsed but produced no Op({:?})",
                            op.name(),
                            op
                        ));
                    }
                }
                Err(e) => failures.push(format!("{} failed to parse: {}", op.name(), e)),
            }
        }

        assert!(
            failures.is_empty(),
            "opcodes without a working keyword path:\n{}",
            failures.join("\n")
        );
    }

    /// OpCode::ALL must enumerate every variant exactly once. Duplicate
    /// entries would mask a missing one behind a correct length.
    #[test]
    fn opcode_list_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for &op in OpCode::ALL.iter() {
            assert!(seen.insert(op), "duplicate in OpCode::ALL: {:?}", op);
        }
    }
}

#[cfg(test)]
mod specification_drift {
    use crate::ast::OpCode;

    /// The specification's opcode reference must cover every opcode. The
    /// table is derived from the enum; this gate keeps it that way when a
    /// variant is added.
    #[test]
    fn specification_lists_every_opcode() {
        let spec = include_str!("../docs/specification.md");
        let missing: Vec<&str> = OpCode::ALL
            .iter()
            .map(|op| op.name())
            .filter(|kw| !spec.contains(&format!("`{}`", kw)))
            .collect();
        assert!(
            missing.is_empty(),
            "opcodes absent from docs/specification.md: {:?}",
            missing
        );
    }
}
