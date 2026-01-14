//! Determinism tests for OUROCHRONOS.
//!
//! These tests verify that the runtime produces identical outputs for identical
//! inputs across multiple runs. This is essential for correctness of the
//! Deutschian CTC model.

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::timeloop::TimeLoopConfig;
    use crate::core_types::OutputItem;

    /// Extract numeric values from output items.
    fn extract_output_values(output: &[OutputItem]) -> Vec<u64> {
        output.iter().filter_map(|item| {
            match item {
                OutputItem::Val(v) => Some(v.val),
                OutputItem::Char(c) => Some(*c as u64),
            }
        }).collect()
    }
    
    /// Run a program multiple times and verify identical results.
    fn verify_determinism(source: &str, runs: usize) {
        let tokens = tokenize(source);
        let mut parser = Parser::new(&tokens);
        let program = parser.parse_program().expect("Parse failed");
        
        let mut results: Vec<(Memory, Vec<u64>, usize)> = Vec::new();
        
        for _ in 0..runs {
            let config = TimeLoopConfig {
                max_epochs: 100,
                mode: ExecutionMode::Standard,
                seed: 0,
                verbose: false,
                frozen_inputs: Vec::new(),
                max_instructions: 10_000_000,
            };
            
            let mut driver = TimeLoop::new(config);
            let result = driver.run(&program);
            
            if let ConvergenceStatus::Consistent { memory, output, epochs } = result {
                results.push((
                    memory,
                    extract_output_values(&output),
                    epochs,
                ));
            }
        }

        // Verify all runs produced consistent results
        assert!(!results.is_empty(), "No consistent results produced");
        
        let (first_mem, first_out, first_epochs) = &results[0];
        for (i, (mem, out, epochs)) in results.iter().enumerate().skip(1) {
            assert!(
                first_mem.values_equal(mem),
                "Memory differs between run 0 and run {}", i
            );
            assert_eq!(
                first_out, out,
                "Output differs between run 0 and run {}", i
            );
            assert_eq!(
                first_epochs, epochs,
                "Epoch count differs between run 0 and run {}", i
            );
        }
    }
    
    /// Verify determinism for action-guided mode.
    fn verify_action_guided_determinism(source: &str, runs: usize, num_seeds: usize) {
        let tokens = tokenize(source);
        let mut parser = Parser::new(&tokens);
        let program = parser.parse_program().expect("Parse failed");
        
        let mut results: Vec<(Memory, Vec<u64>, usize)> = Vec::new();
        
        for _ in 0..runs {
            let config = TimeLoopConfig {
                max_epochs: 100,
                mode: ExecutionMode::ActionGuided {
                    config: ActionConfig::anti_trivial(),
                    num_seeds,
                },
                seed: 0,
                verbose: false,
                frozen_inputs: Vec::new(),
                max_instructions: 10_000_000,
            };
            
            let mut driver = TimeLoop::new(config);
            let result = driver.run(&program);
            
            if let ConvergenceStatus::Consistent { memory, output, epochs } = result {
                results.push((
                    memory,
                    extract_output_values(&output),
                    epochs,
                ));
            }
        }

        assert!(!results.is_empty(), "No consistent results produced");
        
        let (first_mem, first_out, _) = &results[0];
        for (i, (mem, out, _)) in results.iter().enumerate().skip(1) {
            assert!(
                first_mem.values_equal(mem),
                "Action-guided: Memory differs between run 0 and run {}", i
            );
            assert_eq!(
                first_out, out,
                "Action-guided: Output differs between run 0 and run {}", i
            );
        }
    }
    
    // ========================================================================
    // Determinism Tests - Standard Mode
    // ========================================================================
    
    #[test]
    fn test_determinism_trivial() {
        verify_determinism("10 20 ADD OUTPUT", 5);
    }
    
    #[test]
    fn test_determinism_self_fulfilling() {
        verify_determinism("0 ORACLE 0 PROPHECY", 5);
    }
    
    #[test]
    fn test_determinism_witness_pattern() {
        verify_determinism(
            "0 ORACLE DUP 3 EQ IF { DUP 0 PROPHECY } ELSE { POP 3 0 PROPHECY }",
            5
        );
    }
    
    #[test]
    fn test_determinism_arithmetic_chain() {
        verify_determinism(
            "0 ORACLE DUP 2 MUL 1 ADD 0 PROPHECY",
            5
        );
    }
    
    #[test]
    fn test_determinism_conditional() {
        // Program that converges based on condition
        verify_determinism(
            "0 ORACLE DUP 5 GT IF { 5 0 PROPHECY } ELSE { DUP 0 PROPHECY }",
            5
        );
    }
    
    // ========================================================================
    // Determinism Tests - Action-Guided Mode (Canonical Chronology)
    // ========================================================================
    
    #[test]
    fn test_action_guided_determinism() {
        verify_action_guided_determinism("0 ORACLE 0 PROPHECY", 3, 4);
    }
    
    #[test]
    fn test_action_guided_determinism_with_output() {
        verify_action_guided_determinism(
            "0 ORACLE DUP 0 EQ NOT IF { DUP OUTPUT DUP 0 PROPHECY } ELSE { 1 0 PROPHECY }",
            3,
            4
        );
    }
    
    #[test]
    fn test_canonical_selection_determinism() {
        // Test that canonical selection produces deterministic results
        // even when multiple fixed points have equal action
        verify_action_guided_determinism(
            r#"
            0 ORACLE
            DUP 1 GT IF {
                DUP 15 LT IF {
                    DUP 0 PROPHECY OUTPUT
                } ELSE {
                    2 0 PROPHECY
                }
            } ELSE {
                2 0 PROPHECY
            }
            "#,
            3,
            8
        );
    }
    
    // ========================================================================
    // Memory Ordering Tests
    // ========================================================================
    
    #[test]
    fn test_memory_ordering_consistency() {
        use std::cmp::Ordering;
        
        // Verify Memory::cmp produces consistent ordering
        let mut mem_a = Memory::new();
        let mut mem_b = Memory::new();
        let mut mem_c = Memory::new();
        
        mem_a.write(0, Value::new(1));
        mem_b.write(0, Value::new(2));
        mem_c.write(0, Value::new(1));
        mem_c.write(1, Value::new(1));
        
        // a < b (1 < 2 at address 0)
        assert_eq!(mem_a.cmp(&mem_b), Ordering::Less);
        
        // a < c (equal at 0, but c has non-zero at 1)
        assert_eq!(mem_a.cmp(&mem_c), Ordering::Less);
        
        // Reflexivity
        assert_eq!(mem_a.cmp(&mem_a), Ordering::Equal);
        
        // Antisymmetry
        assert_eq!(mem_b.cmp(&mem_a), Ordering::Greater);
    }
    
    #[test] 
    fn test_selection_rule_determinism() {
        use crate::action::{FixedPointSelector, ActionPrinciple, ActionConfig};
        
        let principle = ActionPrinciple::new(ActionConfig::default());
        let seed = Memory::new();
        
        // Create two candidates with equal action
        let mut mem1 = Memory::new();
        mem1.write(0, Value::new(5));
        
        let mut mem2 = Memory::new();
        mem2.write(0, Value::new(10));
        
        // Run selection multiple times - should always pick same one
        let mut selected_values: Vec<u64> = Vec::new();
        
        for _ in 0..5 {
            let mut selector = FixedPointSelector::new(principle.clone());
            
            // Add in different orders
            if selected_values.len() % 2 == 0 {
                selector.add_candidate(mem1.clone(), 1, vec![], seed.clone());
                selector.add_candidate(mem2.clone(), 1, vec![], seed.clone());
            } else {
                selector.add_candidate(mem2.clone(), 1, vec![], seed.clone());
                selector.add_candidate(mem1.clone(), 1, vec![], seed.clone());
            }
            
            if let Some(best) = selector.select_best() {
                selected_values.push(best.memory.read(0).val);
            }
        }
        
        // All selections should be identical (canonical ordering picks the same one)
        let first = selected_values[0];
        for val in &selected_values {
            assert_eq!(*val, first, "Selection varied across runs: {:?}", selected_values);
        }
    }
}
