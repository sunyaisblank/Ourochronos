//! Property-based tests for OUROCHRONOS.
//!
//! Uses proptest to verify invariants across randomly generated inputs.

#[cfg(test)]
mod tests {
    use crate::core::{Memory, OutputItem, Value};
    use crate::*;
    use proptest::prelude::*;

    // ========================================================================
    // Memory Property Tests
    // ========================================================================

    proptest! {
        /// Memory hash is consistent: same writes produce same hash.
        #[test]
        fn prop_memory_hash_deterministic(
            addr1 in 0u64..1000,
            val1 in any::<u64>(),
            addr2 in 0u64..1000,
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
            writes in prop::collection::vec((0u64..500, any::<u64>()), 1..20),
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
            addr in 0u64..100,
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
            use crate::temporal::action::{ActionPrinciple, ActionConfig, FixedPointSelector};

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

                let best = selector.select_best();
                prop_assert!(best.is_some(), "nonempty candidates must produce a selection");
                results.push(best.unwrap().memory.read(0).val);
            }

            // All selections should be identical
            prop_assert_eq!(results.len(), 3);
            for &val in &results {
                prop_assert_eq!(val, results[0]);
            }
        }
    }

    // ========================================================================
    // Execution Property Tests
    // ========================================================================

    proptest! {
        /// Generated pure additions converge in one epoch with exact output.
        #[test]
        fn prop_pure_program_single_epoch(
            a in any::<u64>(),
            b in any::<u64>(),
        ) {
            let source = format!("{} {} ADD OUTPUT", a, b);
            let tokens = tokenize(&source);
            let mut parser = Parser::new(&tokens);

            let parsed = parser.parse_program();
            prop_assert!(parsed.is_ok(), "generated valid source failed to parse: {:?}", parsed.as_ref().err());
            let program = parsed.unwrap();
            let config = crate::temporal::timeloop::TimeLoopConfig::default();
            let mut driver = TimeLoop::new(config).expect("valid configuration");
            let result = driver.run(&program);

            match result {
                ConvergenceStatus::Consistent { epochs, output, .. } => {
                    prop_assert_eq!(epochs, 1);
                    prop_assert_eq!(output, vec![OutputItem::Val(Value::new(a.wrapping_add(b)))]);
                }
                status => prop_assert!(false, "Expected consistent, got {:?}", status),
            }
        }
    }
}
