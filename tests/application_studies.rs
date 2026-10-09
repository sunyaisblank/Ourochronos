//! Finite application acceptance through real source/admission/solver/VM and
//! package APIs. Linux production proof rendering runs in supervised workers.

#[path = "../examples/studies/mod.rs"]
mod studies;

use ourochronos::{GlobalUniquenessResult, PortablePackage, PropertyVerificationResult};
use studies::{Study, MAX_GAS};

fn fixture(case: usize) -> (Study, Vec<Vec<u64>>) {
    match case {
        0 => (
            Study::Exclusion {
                contenders: 2,
                eligible: 3,
            },
            vec![vec![1], vec![2]],
        ),
        1 => (
            Study::Exclusion {
                contenders: 8,
                eligible: 255,
            },
            (0..8).map(|bit| vec![1 << bit]).collect(),
        ),
        2 => (
            Study::Exclusion {
                contenders: 8,
                eligible: 128,
            },
            vec![vec![128]],
        ),
        3 => (
            Study::Exclusion {
                contenders: 3,
                eligible: 0,
            },
            vec![],
        ),
        4 => (
            Study::Dataflow {
                nodes: 2,
                bits: 2,
                gains: vec![1, 2],
                biases: vec![1, 0],
            },
            vec![vec![3, 2]],
        ),
        5 => (
            Study::Dataflow {
                nodes: 1,
                bits: 4,
                gains: vec![1],
                biases: vec![1],
            },
            vec![],
        ),
        6 => (
            Study::Dataflow {
                nodes: 8,
                bits: 4,
                gains: vec![1; 8],
                biases: vec![0; 8],
            },
            (0..16).map(|value| vec![value; 8]).collect(),
        ),
        7 => (
            Study::Dataflow {
                nodes: 8,
                bits: 4,
                gains: vec![0; 8],
                biases: (0..8).collect(),
            },
            vec![(0..8).collect()],
        ),
        8 => (
            Study::Game {
                actions: 4,
                successors: vec![1, 0, 2, 3],
            },
            vec![vec![2], vec![3]],
        ),
        9 => (
            Study::Game {
                actions: 16,
                successors: (0..16).map(|action| (action + 1) % 16).collect(),
            },
            vec![],
        ),
        10 => (
            Study::Game {
                actions: 7,
                successors: vec![0; 7],
            },
            vec![vec![0]],
        ),
        11 => (
            Study::Game {
                actions: 1,
                successors: vec![0],
            },
            vec![vec![0]],
        ),
        12 => (
            Study::Dataflow {
                nodes: 3,
                bits: 2,
                gains: vec![2, 0, 3],
                biases: vec![1, 2, 3],
            },
            vec![vec![1, 2, 2]],
        ),
        _ => unreachable!("bounded fixture index"),
    }
}

fn check_workflow(case: usize) {
    let (study, expected) = fixture(case);
    let report = studies::run(study, MAX_GAS).unwrap();
    assert!(
        !report.is_unknown(),
        "case {case}: {}",
        report.text().unwrap()
    );
    assert_eq!(report.conventional.fixed_states, expected);
    assert_eq!(report.replays.len(), expected.len());
    assert_eq!(
        report.solver_status(),
        match expected.len() {
            0 => "no-fixed-point",
            1 => "unique",
            _ => "multiple",
        }
    );
    assert!(matches!(
        report.property,
        PropertyVerificationResult::Proven { .. } | PropertyVerificationResult::Vacuous { .. }
    ));
    for replay in &report.replays {
        assert_eq!(replay.output, replay.package_output);
        assert!(replay.fetched_records > 0 && replay.fetched_records <= MAX_GAS);
        let decoded = PortablePackage::from_bytes(&replay.package).unwrap();
        assert_eq!(decoded.program, report.code);
        assert_eq!(
            decoded.manifest.memory_cells as usize,
            report.study.memory_cells()
        );
        assert_eq!(decoded.manifest.max_instructions, MAX_GAS);
        assert_eq!(decoded.to_bytes().unwrap(), replay.package);
    }
    match case {
        1 => {
            assert_eq!(report.conventional.candidate_trials, 256);
            assert_eq!(report.conventional.operations, 256);
        }
        3 => assert!(report.orbit_outcome.as_ref().unwrap().contains("paradox")),
        5 => assert!(report.orbit_outcome.as_ref().unwrap().contains("period 16")),
        6 => {
            assert_eq!(report.conventional.candidate_trials, 16);
            assert_eq!(report.conventional.operations, 128);
        }
        8 => {
            assert_eq!(report.graph_fetched_records, 4 * 15);
            assert_eq!(report.conventional.transitions, vec![1, 0, 2, 3]);
            assert_eq!(
                report
                    .conventional
                    .recurrent
                    .iter()
                    .map(|class| (&class.cycle, class.basin_size))
                    .collect::<Vec<_>>(),
                vec![(&vec![0, 1], 2), (&vec![2], 1), (&vec![3], 1)]
            );
            assert!(report.orbit_outcome.as_ref().unwrap().contains("period 2"));
        }
        9 => {
            assert_eq!(report.graph_fetched_records, 16 * 15);
            assert_eq!(report.conventional.recurrent.len(), 1);
            assert_eq!(report.conventional.recurrent[0].cycle.len(), 16);
            assert!(report.orbit_outcome.as_ref().unwrap().contains("period 16"));
        }
        10 => assert_eq!(report.conventional.recurrent[0].basin_size, 8),
        11 => assert_eq!(report.conventional.transitions, vec![0, 0]),
        _ => {}
    }
    let record = report.text().unwrap();
    assert!(record.len() < 1024 * 1024);
    assert!(record.contains(&report.code_sha256));
    assert!(record.contains(&report.source));
    assert!(record.contains("different work models"));
    assert!(record.contains("UNSAT evidence trusts Z3 backend"));
    println!(
        "qualified application case {case}: {}",
        report.solver_status()
    );
}

#[test]
fn configured_workflows_match_conventional_and_package_results() {
    // The child runs exactly this substantive test with one fixture, avoiding
    // an extra no-op probe test and bounding query plus proof serialization.
    if let Ok(case) = std::env::var("OUROCHRONOS_APPLICATION_FIXTURE") {
        check_workflow(case.parse().unwrap());
        return;
    }
    for case in 0..13 {
        #[cfg(target_os = "linux")]
        {
            use ourochronos::runtime::isolation::{run_bounded, ProcessLimits};
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "configured_workflows_match_conventional_and_package_results",
                    "--nocapture",
                    "--test-threads",
                    "1",
                ])
                .env("OUROCHRONOS_APPLICATION_FIXTURE", case.to_string());
            let output = run_bounded(
                command,
                ProcessLimits {
                    wall_time: std::time::Duration::from_secs(15),
                    address_space_bytes: 512 * 1024 * 1024,
                    output_bytes: 1024 * 1024,
                },
                None,
            )
            .unwrap();
            assert_eq!(output.stopped, None, "case {case}: {output:?}");
            assert!(
                output.status.success(),
                "case {case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout)
                .contains(&format!("qualified application case {case}:")));
        }
        #[cfg(not(target_os = "linux"))]
        check_workflow(case);
    }
}

#[test]
fn ring_reconstruction_matches_separate_exhaustive_small_graphs() {
    // Independent check of the conventional method itself: enumerate whole
    // vectors and directly evaluate every simultaneous equation. Never use
    // the reverse-ring reconstruction here. Noninvertible gains included.
    for nodes in 1..=3 {
        for gain in 0..4 {
            for bias in 0..4 {
                let gains: Vec<u64> = (0..nodes).map(|node| (gain + node as u64) % 4).collect();
                let biases: Vec<u64> = (0..nodes).map(|node| (bias + node as u64) % 4).collect();
                let study = Study::Dataflow {
                    nodes,
                    bits: 2,
                    gains: gains.clone(),
                    biases: biases.clone(),
                };
                let mut expected = Vec::new();
                for encoded in 0..1usize << (2 * nodes) {
                    let words: Vec<u64> = (0..nodes)
                        .map(|node| ((encoded >> (2 * node)) & 3) as u64)
                        .collect();
                    if (0..nodes).all(|node| {
                        words[node] == (gains[node] * words[(node + 1) % nodes] + biases[node]) % 4
                    }) {
                        expected.push(words);
                    }
                }
                expected.sort();
                let mut actual = study.conventional().unwrap().fixed_states;
                actual.sort();
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
fn invalid_parameters_and_parser_caps_are_rejected_before_work() {
    for study in [
        Study::Exclusion {
            contenders: 0,
            eligible: 0,
        },
        Study::Exclusion {
            contenders: 9,
            eligible: 0,
        },
        Study::Exclusion {
            contenders: 8,
            eligible: 256,
        },
        Study::Dataflow {
            nodes: 9,
            bits: 1,
            gains: vec![0; 9],
            biases: vec![0; 9],
        },
        Study::Dataflow {
            nodes: 1,
            bits: 5,
            gains: vec![0],
            biases: vec![0],
        },
        Study::Dataflow {
            nodes: 2,
            bits: 2,
            gains: vec![1],
            biases: vec![0, 0],
        },
        Study::Dataflow {
            nodes: 1,
            bits: 2,
            gains: vec![4],
            biases: vec![0],
        },
        Study::Game {
            actions: 17,
            successors: vec![0; 17],
        },
        Study::Game {
            actions: 2,
            successors: vec![2, 0],
        },
        Study::Game {
            actions: 2,
            successors: vec![1],
        },
    ] {
        assert!(studies::run(study, MAX_GAS).is_err());
    }
    for args in [
        vec!["exclusion", "8", "255", "--gas", "0"],
        vec!["game", "2", "1,0", "--gas", "1025"],
        vec!["dataflow", "999999999", "1", "0", "0"],
        vec!["game", "1", "0", "--save", "x", "--save", "y"],
    ] {
        assert!(
            studies::parse_cli(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
        );
    }
    let oversized = format!("0,{}", "0,".repeat(16));
    assert!(studies::parse_cli(&["game".into(), "1".into(), oversized]).is_err());
}

#[test]
fn gas_unknown_retains_conventional_answers_without_package_promotion() {
    for case in [0, 4, 8] {
        let (study, expected) = fixture(case);
        let report = studies::run(study, 1).unwrap();
        assert_eq!(report.conventional.fixed_states, expected);
        assert!(report.is_unknown());
        assert!(matches!(
            report.solver,
            GlobalUniquenessResult::Unknown { .. }
        ));
        assert!(report.replays.is_empty());
        assert!(report.orbit_package.is_none());
        let text = report.text().unwrap();
        assert!(text.contains("solver-outcome: unknown"));
        assert!(!text.contains("point-replay:"));
    }
}

#[test]
fn frozen_parameters_bind_code_and_package_evidence_and_save_preserves_existing_file() {
    use sha2::{Digest, Sha256};
    let first = studies::run(
        Study::Exclusion {
            contenders: 2,
            eligible: 1,
        },
        MAX_GAS,
    )
    .unwrap();
    let second = studies::run(
        Study::Exclusion {
            contenders: 2,
            eligible: 2,
        },
        MAX_GAS,
    )
    .unwrap();
    assert_ne!(first.source, second.source);
    assert_ne!(first.code_sha256, second.code_sha256);
    assert_ne!(first.replays[0].output, second.replays[0].output);
    let own_hash = format!("{:x}", Sha256::digest(first.code.to_bytes().unwrap()));
    assert_eq!(first.code_sha256, own_hash);
    let parameter_hash = format!(
        "{:x}",
        Sha256::digest(format!("{:?}", first.study).as_bytes())
    );
    assert!(PortablePackage::from_bytes(&first.replays[0].package)
        .unwrap()
        .manifest
        .name
        .ends_with(&parameter_hash));
    let mut substituted = PortablePackage::from_bytes(&first.replays[0].package).unwrap();
    substituted.program = second.code.clone();
    assert!(substituted.to_bytes().is_err());
    let path = std::env::temp_dir().join(format!(
        "ouro-application-result-{}.txt",
        std::process::id()
    ));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    assert!(!path.exists());
    let _cleanup = Cleanup(path.clone());
    let text = first.text().unwrap();
    studies::save_new(&path, &text).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    assert!(studies::save_new(&path, &second.text().unwrap()).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    let args: Vec<_> = [
        "dataflow",
        "2",
        "2",
        "1,2",
        "1,0",
        "--save",
        "fresh-result.txt",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let (parsed, gas, save) = studies::parse_cli(&args).unwrap();
    assert_eq!(parsed, fixture(4).0);
    assert_eq!(gas, MAX_GAS);
    assert_eq!(save.unwrap().to_str(), Some("fresh-result.txt"));
    let broadcast: Vec<_> = ["dataflow", "2", "2", "1", "0,0"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        studies::parse_cli(&broadcast).unwrap().0,
        Study::Dataflow {
            nodes: 2,
            bits: 2,
            gains: vec![1, 1],
            biases: vec![0, 0]
        }
    );
}
