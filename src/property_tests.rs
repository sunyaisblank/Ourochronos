//! Property-based tests for OUROCHRONOS.
//!
//! Uses proptest to verify invariants across randomly generated inputs.

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::core_types::{Memory, Value, OutputItem};
    use proptest::prelude::*;

    /// Extract the numeric value from an OutputItem.
    fn output_value(item: &OutputItem) -> u64 {
        match item {
            OutputItem::Val(v) => v.val,
            OutputItem::Char(c) => *c as u64,
        }
    }
    
    // ========================================================================
    // Memory Property Tests
    // ========================================================================
    
    proptest! {
        /// Memory hash is consistent: same writes produce same hash.
        #[test]
        fn prop_memory_hash_deterministic(
            addr1 in 0u16..1000,
            val1 in any::<u64>(),
            addr2 in 0u16..1000,
            val2 in any::<u64>(),
        ) {
            let mut mem1 = Memory::new();
            let mut mem2 = Memory::new();
            
            mem1.write(addr1, Value::new(val1));
            mem1.write(addr2, Value::new(val2));
            
            mem2.write(addr1, Value::new(val1));
            mem2.write(addr2, Value::new(val2));
            
            prop_assert_eq!(mem1.state_hash(), mem2.state_hash());
        }
        
        /// Incremental hash matches full recompute.
        #[test]
        fn prop_incremental_hash_correct(
            writes in prop::collection::vec((0u16..500, any::<u64>()), 1..20),
        ) {
            let mut mem = Memory::new();
            
            for (addr, val) in writes {
                mem.write(addr, Value::new(val));
            }
            
            prop_assert_eq!(mem.state_hash(), mem.recompute_hash());
        }
        
        /// Memory ordering is antisymmetric.
        #[test]
        fn prop_memory_ordering_antisymmetric(
            addr in 0u16..100,
            val1 in any::<u64>(),
            val2 in any::<u64>(),
        ) {
            let mut mem1 = Memory::new();
            let mut mem2 = Memory::new();
            
            mem1.write(addr, Value::new(val1));
            mem2.write(addr, Value::new(val2));
            
            use std::cmp::Ordering;
            let cmp1 = mem1.cmp(&mem2);
            let cmp2 = mem2.cmp(&mem1);
            
            match cmp1 {
                Ordering::Less => prop_assert_eq!(cmp2, Ordering::Greater),
                Ordering::Greater => prop_assert_eq!(cmp2, Ordering::Less),
                Ordering::Equal => prop_assert_eq!(cmp2, Ordering::Equal),
            }
        }
    }
    
    // ========================================================================
    // Value Property Tests
    // ========================================================================
    
    proptest! {
        /// Value arithmetic is commutative for addition.
        #[test]
        fn prop_value_add_commutative(a in any::<u64>(), b in any::<u64>()) {
            let va = Value::new(a);
            let vb = Value::new(b);
            
            prop_assert_eq!((va.clone() + vb.clone()).val, (vb + va).val);
        }
        
        /// Value arithmetic is commutative for multiplication.
        #[test]
        fn prop_value_mul_commutative(a in any::<u64>(), b in any::<u64>()) {
            let va = Value::new(a);
            let vb = Value::new(b);
            
            prop_assert_eq!((va.clone() * vb.clone()).val, (vb * va).val);
        }
        
        /// Division by zero returns zero.
        #[test]
        fn prop_div_by_zero_is_zero(a in any::<u64>()) {
            let va = Value::new(a);
            let zero = Value::new(0);
            
            prop_assert_eq!((va / zero).val, 0);
        }
        
        /// Modulo by zero returns zero.
        #[test]
        fn prop_mod_by_zero_is_zero(a in any::<u64>()) {
            let va = Value::new(a);
            let zero = Value::new(0);
            
            prop_assert_eq!((va % zero).val, 0);
        }
    }
    
    // ========================================================================
    // Fixed-Point Selection Property Tests
    // ========================================================================
    
    proptest! {
        /// Selection is idempotent: selecting from same candidates gives same result.
        #[test]
        fn prop_selection_idempotent(
            vals in prop::collection::vec(0u64..1000, 2..10),
        ) {
            use crate::action::{ActionPrinciple, ActionConfig, FixedPointSelector};
            
            let principle = ActionPrinciple::new(ActionConfig::default());
            let seed = Memory::new();
            
            let mut results = Vec::new();
            
            for _ in 0..3 {
                let mut selector = FixedPointSelector::new(principle.clone());
                
                for (i, &val) in vals.iter().enumerate() {
                    let mut mem = Memory::new();
                    mem.write(0, Value::new(val));
                    mem.write(1, Value::new(i as u64));
                    selector.add_candidate(mem, 1, vec![], seed.clone());
                }
                
                if let Some(best) = selector.select_best() {
                    results.push(best.memory.read(0).val);
                }
            }
            
            // All selections should be identical
            if let Some(&first) = results.first() {
                for &val in &results {
                    prop_assert_eq!(val, first);
                }
            }
        }
    }
    
    // ========================================================================
    // Execution Property Tests  
    // ========================================================================
    
    proptest! {
        /// Pure programs (no ORACLE) always converge in 1 epoch.
        #[test]
        fn prop_pure_program_single_epoch(
            a in 1u64..1000,
            b in 1u64..1000,
        ) {
            let source = format!("{} {} ADD OUTPUT", a, b);
            let tokens = tokenize(&source);
            let mut parser = Parser::new(&tokens);
            
            if let Ok(program) = parser.parse_program() {
                let config = timeloop::TimeLoopConfig::default();
                let mut driver = TimeLoop::new(config);
                let result = driver.run(&program);
                
                match result {
                    ConvergenceStatus::Consistent { epochs, output, .. } => {
                        prop_assert_eq!(epochs, 1);
                        prop_assert!(!output.is_empty());
                        prop_assert_eq!(output_value(&output[0]), a.wrapping_add(b));
                    }
                    _ => prop_assert!(false, "Expected consistent"),
                }
            }
        }
    }
}
