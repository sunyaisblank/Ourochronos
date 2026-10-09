//! Analytic exact expectations for the bounded sparse stochastic backend.
//! Expected distributions follow flow balance, symmetry or absorbing-state
//! identities, rather than calling the backend's elimination implementation.

use num_bigint::BigInt;
use num_traits::{One, Zero};
use ourochronos::temporal::sparse_markov::{
    ExactRational as R, SparseMarkovChain, SparseMarkovError as Error,
    SparseMarkovLimits as Limits, SparseMarkovResource as Resource,
    SparseStationaryDecision as Decision, MAX_SPARSE_MARKOV_STATES,
};

fn r(numerator: i64, denominator: i64) -> R {
    R::new(numerator.into(), denominator.into())
}

fn chain(rows: Vec<Vec<(usize, R)>>) -> SparseMarkovChain {
    SparseMarkovChain::new(rows, Limits::default()).unwrap()
}

fn cycle(states: usize) -> Vec<Vec<(usize, R)>> {
    (0..states)
        .map(|source| vec![((source + 1) % states, R::one())])
        .collect()
}

#[test]
fn duplicate_zero_and_raw_fraction_inputs_are_canonicalized_exactly() {
    let model = chain(vec![
        vec![(1, r(1, 4)), (0, R::zero()), (1, r(1, 4)), (0, r(1, 2))],
        vec![
            (1, R::new_raw(3.into(), 4.into())),
            (0, R::new_raw(2.into(), 8.into())),
        ],
    ]);
    assert_eq!(model.edges(), 4);
    assert_eq!(
        model.rows(),
        &[
            vec![(0, r(1, 2)), (1, r(1, 2))],
            vec![(0, r(1, 4)), (1, r(3, 4))]
        ]
    );
    let family = model.stationary_family().unwrap();
    // Flow across the cut: pi0*(1/2)=pi1*(1/4), pi0+pi1=1.
    assert_eq!(family.extremal[0].weights, vec![(0, r(1, 3)), (1, r(2, 3))]);
    assert!(family.extremal[0].certificate.is_stationary());
    assert_eq!(
        model.residual(&family.extremal[0].weights).unwrap(),
        family.extremal[0].certificate
    );
    assert_eq!(
        model
            .analyze(&[false, true], r(2, 3), r(1, 3))
            .unwrap()
            .decision,
        Decision::Accept
    );
    assert_eq!(
        model
            .analyze(&[true, false], r(2, 3), r(1, 3))
            .unwrap()
            .decision,
        Decision::Reject
    );
}

#[test]
fn reducible_periodic_and_unreachable_classes_all_have_readout_witnesses() {
    // From initial state 0 only absorbing state 1 is reachable. This API
    // quantifies over all stationary classes, including {2,3} and {4}.
    let model = chain(vec![
        vec![(1, R::one())],
        vec![(1, R::one())],
        vec![(3, R::one())],
        vec![(2, R::one())],
        vec![(4, R::one())],
        vec![(0, r(1, 2)), (2, r(1, 2))],
    ]);
    let analysis = model
        .analyze(&[false, true, true, false, false, false], r(2, 3), r(1, 3))
        .unwrap();
    assert_eq!(
        analysis
            .family
            .extremal
            .iter()
            .map(|distribution| distribution.recurrent_class.clone())
            .collect::<Vec<_>>(),
        vec![vec![1], vec![2, 3], vec![4]]
    );
    assert_eq!(
        analysis
            .family
            .extremal
            .iter()
            .map(|distribution| distribution.weights.clone())
            .collect::<Vec<_>>(),
        vec![
            vec![(1, R::one())],
            vec![(2, r(1, 2)), (3, r(1, 2))],
            vec![(4, R::one())]
        ]
    );
    assert_eq!(
        analysis.acceptance_probabilities,
        vec![R::one(), r(1, 2), R::zero()]
    );
    assert_eq!(analysis.decision, Decision::Ambiguous);
    assert_eq!(
        (analysis.minimum.class_index, analysis.minimum.probability),
        (2, R::zero())
    );
    assert_eq!(
        (analysis.maximum.class_index, analysis.maximum.probability),
        (0, R::one())
    );
    let gap = analysis.gap_witness.unwrap();
    assert_eq!((gap.class_index, gap.probability), (1, r(1, 2)));
    assert!(analysis
        .family
        .extremal
        .iter()
        .all(|distribution| distribution.certificate.is_stationary()));
    assert_eq!(
        model
            .analyze(&[false, true, true, true, true, false], r(2, 3), r(1, 3))
            .unwrap()
            .decision,
        Decision::Accept
    );
    assert_eq!(
        model
            .analyze(&[false; 6], r(2, 3), r(1, 3))
            .unwrap()
            .decision,
        Decision::Reject
    );
    // Disagreement can exist without any extremal inside the promise gap.
    let disagree = chain(vec![vec![(0, R::one())], vec![(1, R::one())]])
        .analyze(&[true, false], r(2, 3), r(1, 3))
        .unwrap();
    assert_eq!(disagree.decision, Decision::Ambiguous);
    assert!(disagree.gap_witness.is_none());
}

#[test]
fn directed_holding_cycle_has_inverse_rate_invariant_distribution() {
    let reciprocals = [2_i64, 3, 5, 7];
    let model = chain(
        reciprocals
            .iter()
            .enumerate()
            .map(|(source, denominator)| {
                vec![
                    (source, r(denominator - 1, *denominator)),
                    ((source + 1) % 4, r(1, *denominator)),
                ]
            })
            .collect(),
    );
    // Stationary flux around every directed edge is the same; pi_i*q_i=c.
    let wanted: Vec<_> = reciprocals
        .iter()
        .enumerate()
        .map(|(state, denominator)| (state, r(*denominator, 17)))
        .collect();
    let family = model.stationary_family().unwrap();
    assert_eq!(family.extremal[0].weights, wanted);
    assert!(family.extremal[0].certificate.is_stationary());
    let periodic = chain(cycle(7)).stationary_family().unwrap();
    assert_eq!(
        periodic.extremal[0].weights,
        (0..7).map(|state| (state, r(1, 7))).collect::<Vec<_>>()
    );
}

#[test]
fn reversible_path_campaign_matches_detailed_balance_under_state_permutation() {
    for states in 2..=12 {
        // Arbitrary positive integer weights w_i=i+1. Neighbor rates
        // 1/(4*w_i) make w_i P_ij = w_j P_ji = 1/4 on every edge.
        let mut rows = Vec::new();
        for state in 0..states {
            let weight = state as i64 + 1;
            let mut row = Vec::new();
            if state > 0 {
                row.push((state - 1, r(1, 4 * weight)));
            }
            if state + 1 < states {
                row.push((state + 1, r(1, 4 * weight)));
            }
            row.push((state, r(4 * weight - row.len() as i64, 4 * weight)));
            rows.push(row);
        }
        let total = (states * (states + 1) / 2) as i64;
        let wanted: Vec<_> = (0..states)
            .map(|state| (state, r(state as i64 + 1, total)))
            .collect();
        let family = chain(rows.clone()).stationary_family().unwrap();
        assert_eq!(family.extremal[0].weights, wanted);
        assert!(family.extremal[0].certificate.is_stationary());
        let mut permuted = vec![Vec::new(); states];
        for (source, row) in rows.into_iter().enumerate() {
            permuted[states - 1 - source] = row
                .into_iter()
                .map(|(target, probability)| (states - 1 - target, probability))
                .collect();
        }
        let expected: Vec<_> = (0..states)
            .map(|state| (state, r((states - state) as i64, total)))
            .collect();
        assert_eq!(
            chain(permuted).stationary_family().unwrap().extremal[0].weights,
            expected
        );
    }
}

#[test]
fn denominators_beyond_fixed_width_produce_exact_flow_balance() {
    let denominator = (BigInt::one() << 200_usize) + BigInt::from(39);
    let next: BigInt = &denominator + 1;
    let model = chain(vec![
        vec![
            (0, R::new(&denominator - 1, denominator.clone())),
            (1, R::new(1.into(), denominator.clone())),
        ],
        vec![
            (0, R::new(1.into(), next.clone())),
            (1, R::new(&next - 1, next.clone())),
        ],
    ]);
    let normalizer: BigInt = &denominator * 2 + 1;
    // pi0 / D = pi1 / (D+1), hence pi0=D/(2D+1).
    let wanted = vec![
        (0, R::new(denominator, normalizer.clone())),
        (1, R::new(next, normalizer)),
    ];
    let family = model.stationary_family().unwrap();
    assert_eq!(family.extremal[0].weights, wanted);
    assert!(family.extremal[0]
        .weights
        .iter()
        .all(|(_, probability)| probability.denom().bits() > 128));
    assert!(family.extremal[0].certificate.is_stationary());
    let readout = model.analyze(&[false, true], r(2, 3), r(1, 3)).unwrap();
    assert_eq!(readout.decision, Decision::Ambiguous);
    assert_eq!(
        readout.gap_witness.unwrap().probability,
        family.extremal[0].weights[1].1
    );
}

#[test]
fn independent_residual_identities_detect_wrong_weights_and_normalization() {
    let model = chain(vec![
        vec![(0, r(1, 2)), (1, r(1, 2))],
        vec![(0, r(1, 4)), (1, r(3, 4))],
    ]);
    let incorrect = model.residual(&[(0, r(1, 2)), (1, r(1, 2))]).unwrap();
    assert_eq!(incorrect.normalization, R::one());
    assert_eq!(incorrect.residual, vec![(0, r(-1, 8)), (1, r(1, 8))]);
    assert!(!incorrect.is_stationary());
    let doubled = model.residual(&[(0, r(2, 3)), (1, r(4, 3))]).unwrap();
    assert_eq!(doubled.normalization, r(2, 1));
    assert!(doubled.residual.is_empty());
    assert!(!doubled.is_stationary());
    for malformed in [
        vec![(0, r(-1, 1))],
        vec![(0, R::one()), (0, R::one())],
        vec![(2, R::one())],
    ] {
        assert!(matches!(
            model.residual(&malformed),
            Err(Error::InvalidDistribution { .. })
        ));
    }
}

#[test]
fn malformed_probability_and_readout_inputs_fail_predictably() {
    assert!(matches!(
        SparseMarkovChain::new(vec![], Limits::default()),
        Err(Error::EmptyChain)
    ));
    assert!(matches!(
        SparseMarkovChain::new(vec![vec![(0, r(-1, 1))]], Limits::default()),
        Err(Error::NegativeProbability { row: 0, target: 0 })
    ));
    assert!(matches!(
        SparseMarkovChain::new(vec![vec![(0, r(1, 2))]], Limits::default()),
        Err(Error::InvalidRowSum { row: 0, .. })
    ));
    assert!(matches!(
        SparseMarkovChain::new(vec![vec![(1, R::zero()), (0, R::one())]], Limits::default()),
        Err(Error::InvalidTarget {
            row: 0,
            target: 1,
            states: 1
        })
    ));
    assert!(matches!(
        SparseMarkovChain::new(
            vec![vec![(0, R::new_raw(1.into(), 0.into()))]],
            Limits::default()
        ),
        Err(Error::ZeroDenominator {
            context: "row",
            index: 0
        })
    ));
    // Noncanonical signs are normalized before probability sign validation.
    assert_eq!(
        chain(vec![vec![(0, R::new_raw((-2).into(), (-2).into()))]]).rows()[0],
        vec![(0, R::one())]
    );
    let model = chain(vec![vec![(0, R::one())]]);
    assert!(matches!(
        model.analyze(&[], r(2, 3), r(1, 3)),
        Err(Error::InvalidReadoutLength {
            expected: 1,
            actual: 0
        })
    ));
    for (accept, reject) in [(r(1, 3), r(2, 3)), (r(2, 1), r(1, 3)), (r(2, 3), r(-1, 3))] {
        assert_eq!(
            model.analyze(&[true], accept, reject).unwrap_err(),
            Error::InvalidThresholds
        );
    }
    assert!(matches!(
        model.analyze(&[true], R::new_raw(1.into(), 0.into()), r(1, 3)),
        Err(Error::ZeroDenominator {
            context: "accept threshold",
            ..
        })
    ));
    // Inclusive equal thresholds preserve the existing Accept precedence.
    assert_eq!(
        model.analyze(&[true], R::one(), R::one()).unwrap().decision,
        Decision::Accept
    );
}

#[test]
fn state_raw_edge_and_input_integer_caps_apply_before_canonicalization() {
    let limits = Limits {
        max_states: 1,
        ..Limits::default()
    };
    assert!(matches!(
        SparseMarkovChain::new(cycle(2), limits),
        Err(Error::ResourceLimit {
            resource: Resource::States,
            limit: 1,
            required: 2
        })
    ));
    let rows = vec![vec![(0, R::one()), (0, R::zero()), (0, R::zero())]];
    assert!(matches!(
        SparseMarkovChain::new(
            rows,
            Limits {
                max_raw_edges: 2,
                ..Limits::default()
            }
        ),
        Err(Error::ResourceLimit {
            resource: Resource::RawEdges,
            limit: 2,
            required: 3
        })
    ));
    let integer: BigInt = (BigInt::one() << 10_usize) - 1;
    let raw = R::new_raw(integer.clone(), integer);
    // Raw integers have ten bits even though the rational reduces to one.
    assert_eq!(
        SparseMarkovChain::new(
            vec![vec![(0, raw.clone())]],
            Limits {
                max_integer_bits: 10,
                ..Limits::default()
            }
        )
        .unwrap()
        .rows()[0],
        vec![(0, R::one())]
    );
    assert!(matches!(
        SparseMarkovChain::new(
            vec![vec![(0, raw)]],
            Limits {
                max_integer_bits: 9,
                ..Limits::default()
            }
        ),
        Err(Error::ResourceLimit {
            resource: Resource::IntegerBits,
            limit: 9,
            required: 10
        })
    ));
    assert!(matches!(
        SparseMarkovChain::new(
            cycle(1),
            Limits {
                max_states: MAX_SPARSE_MARKOV_STATES + 1,
                ..Limits::default()
            }
        ),
        Err(Error::InvalidLimit {
            resource: Resource::States,
            ..
        })
    ));
    assert!(matches!(
        SparseMarkovChain::new(
            cycle(1),
            Limits {
                max_operations: 0,
                ..Limits::default()
            }
        ),
        Err(Error::InvalidLimit {
            resource: Resource::Operations,
            requested: 0,
            ..
        })
    ));
}

#[test]
fn elimination_work_fill_and_intermediate_growth_have_adjacent_caps() {
    let rows = cycle(3);
    let model = chain(rows.clone());
    let family = model.stationary_family().unwrap();
    assert_eq!(
        family.extremal[0].weights,
        vec![(0, r(1, 3)), (1, r(1, 3)), (2, r(1, 3))]
    );
    let stats = family.stats;
    assert!(stats.charged_operations > model.admission_stats().charged_operations);
    assert!(stats.peak_integer_bits > model.admission_stats().peak_integer_bits);
    for resource in [
        Resource::Operations,
        Resource::MatrixEntries,
        Resource::IntegerBits,
    ] {
        let limit = match resource {
            Resource::Operations => stats.charged_operations,
            Resource::MatrixEntries => stats.peak_matrix_entries as u64,
            Resource::IntegerBits => stats.peak_integer_bits,
            _ => unreachable!(),
        };
        let configure = |cap| match resource {
            Resource::Operations => Limits {
                max_operations: cap,
                ..Limits::default()
            },
            Resource::MatrixEntries => Limits {
                max_matrix_entries: cap as usize,
                ..Limits::default()
            },
            Resource::IntegerBits => Limits {
                max_integer_bits: cap,
                ..Limits::default()
            },
            _ => unreachable!(),
        };
        assert_eq!(
            SparseMarkovChain::new(rows.clone(), configure(limit))
                .unwrap()
                .stationary_family()
                .unwrap()
                .extremal,
            family.extremal
        );
        if resource == Resource::Operations {
            // Solving and classifying share one budget; readout does not get
            // a silently reset allowance after the stationary solve.
            let limited = SparseMarkovChain::new(rows.clone(), configure(limit)).unwrap();
            assert!(matches!(
                limited.analyze(&[true, false, false], r(2, 3), r(1, 3)),
                Err(Error::ResourceLimit {
                    resource: Resource::Operations,
                    ..
                })
            ));
        }
        let limited = SparseMarkovChain::new(rows.clone(), configure(limit - 1)).unwrap();
        assert!(
            matches!(limited.stationary_family(), Err(Error::ResourceLimit { resource: actual, limit: actual_limit, required }) if actual == resource && actual_limit == limit - 1 && required > actual_limit)
        );
    }
}

#[test]
fn many_absorbing_classes_use_sparse_output_and_linear_graph_work() {
    let states = 513;
    let model = chain((0..states).map(|state| vec![(state, R::one())]).collect());
    let family = model.stationary_family().unwrap();
    assert_eq!(family.extremal.len(), states);
    assert_eq!(
        family
            .extremal
            .iter()
            .map(|distribution| distribution.weights.len())
            .sum::<usize>(),
        states
    );
    assert_eq!(family.stats.peak_matrix_entries, 0);
    assert!(family.stats.charged_operations < 40 * states as u64);
    for (state, distribution) in family.extremal.iter().enumerate() {
        assert_eq!(distribution.recurrent_class, vec![state]);
        assert_eq!(distribution.weights, vec![(state, R::one())]);
        assert!(distribution.certificate.is_stationary());
    }
}
