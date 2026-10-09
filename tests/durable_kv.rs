#![cfg(target_os = "linux")]
//! Controlled local fixtures only. Process exits/signals qualify recovery;
//! these tests do not emulate a physical power loss or an external service.
use ourochronos::temporal::durable_kv::{
    DurableCommitPhase, DurableCommitStore, DurableKvConfig, DurableKvError, DurableKvIntent,
    DurableRecovery, MAX_DURABLE_KV_BYTES,
};
use ourochronos::temporal::transaction::{
    CommitOutcome, CommitToken, EffectIntent, FrozenEndpointTape, FrozenFileSnapshot,
    FrozenProcessResult, ObservationTranscript, TemporalTransaction, TimelineCandidate,
    TransactionError, TransactionLimits,
};
use ourochronos::{OutputItem, Provenance, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::os::unix::fs::{symlink, DirBuilderExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// A fork for another crash fixture can retain a flock until exec closes its
// inherited CLOEXEC descriptor. Keep this single-owner campaign deterministic;
// explicit competing-owner checks still run within their own fixture.
static TEST_SESSION: std::sync::Mutex<()> = std::sync::Mutex::new(());
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/durable-kv-fixtures");
        std::fs::create_dir_all(&base).unwrap();
        let path = base.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .unwrap();
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn snapshot(&self) -> Vec<u8> {
        std::fs::read(self.0.join("store.bin")).unwrap()
    }
    fn replace_snapshot(&self, bytes: &[u8]) {
        std::fs::write(self.0.join("store.bin"), bytes).unwrap();
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn policy() -> DurableKvConfig {
    DurableKvConfig::new([7; 16], ["answer".into(), "branch".into()])
}
fn set(key: &str, value: &[u8]) -> EffectIntent {
    DurableKvIntent::Set {
        key: key.into(),
        value: value.into(),
    }
    .into_effect()
    .unwrap()
}
fn selected(effects: Vec<EffectIntent>) -> TemporalTransaction {
    let mut transaction = TemporalTransaction::new(vec![9], TransactionLimits::default()).unwrap();
    let candidate = TimelineCandidate {
        state: vec![
            (
                0,
                Value {
                    val: 0,
                    prov: Provenance {
                        saturated: false,
                        deps: Some(Arc::new(BTreeSet::new())),
                    },
                },
            ),
            (
                3,
                Value {
                    val: 7,
                    prov: Provenance {
                        saturated: true,
                        deps: Some(Arc::new(BTreeSet::from([0, 3]))),
                    },
                },
            ),
        ],
        output: vec![
            OutputItem::Val(Value::with_provenance(7, Provenance::single(3))),
            OutputItem::Char(b'!'),
        ],
        effects,
        inputs_consumed: vec![9],
        observations: ObservationTranscript {
            input: vec![9],
            clock: vec![10],
            random: vec![11],
            files: vec![FrozenFileSnapshot {
                path: "synthetic-file".into(),
                contents: Some(b"observation".to_vec()),
            }],
            endpoints: vec![FrozenEndpointTape {
                endpoint: "synthetic-endpoint".into(),
                recv_bytes: vec![1, 2],
            }],
            processes: vec![FrozenProcessResult {
                command: "synthetic-command".into(),
                output: vec![3],
                exit_code: -7,
            }],
        },
    };
    let id = transaction.stage_candidate(candidate).unwrap();
    transaction.select(id).unwrap();
    transaction
}
fn effects() -> Vec<EffectIntent> {
    vec![
        set("answer", b"answer-value"),
        set("branch", b"branch-value"),
    ]
}
fn assert_values(store: &DurableCommitStore, applied: bool) {
    assert_eq!(
        store.read("answer").unwrap(),
        applied.then_some(b"answer-value".as_slice())
    );
    assert_eq!(
        store.read("branch").unwrap(),
        applied.then_some(b"branch-value".as_slice())
    );
}
fn phases() -> [DurableCommitPhase; 6] {
    [
        DurableCommitPhase::RecordTempSynced,
        DurableCommitPhase::RecordRenamed,
        DurableCommitPhase::RecordDurable,
        DurableCommitPhase::ApplyTempSynced,
        DurableCommitPhase::ApplyRenamed,
        DurableCommitPhase::ApplyDurable,
    ]
}

#[test]
fn full_selected_identity_survives_restart_and_replay_without_republication() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    let token = CommitToken(123);
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    let mut transaction = selected(effects());
    let CommitOutcome::Committed(receipt) = store.commit_selected(&mut transaction, token).unwrap()
    else {
        panic!("new commit expected")
    };
    assert_eq!(receipt.sequence, 0);
    assert_values(&store, true);
    let bytes = scratch.snapshot();
    drop(store);
    let mut store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    assert_eq!(
        store.recover(token).unwrap(),
        DurableRecovery::KnownApplied(receipt.clone())
    );
    let mut replay = selected(effects());
    assert_eq!(
        store.commit_selected(&mut replay, token).unwrap(),
        CommitOutcome::AlreadyCommitted(receipt)
    );
    assert_eq!(scratch.snapshot(), bytes);
    assert_values(&store, true);
    let mut conflict = selected(effects());
    let mut changed = conflict.candidates().values().next().unwrap().clone();
    changed.output[0] = OutputItem::Val(Value::new(7));
    conflict = TemporalTransaction::new(vec![9], TransactionLimits::default()).unwrap();
    let id = conflict.stage_candidate(changed).unwrap();
    conflict.select(id).unwrap();
    assert!(matches!(
        store.commit_selected(&mut conflict, token),
        Err(DurableKvError::Transaction(
            TransactionError::CommitTokenConflict { .. }
        ))
    ));
    assert_eq!(scratch.snapshot(), bytes);
}

#[test]
fn only_explicit_selection_reaches_the_host_and_rollback_remains_terminal() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let mut store = DurableCommitStore::create(scratch.path(), policy()).unwrap();
    let baseline = scratch.snapshot();
    let mut transaction = TemporalTransaction::new(vec![], TransactionLimits::default()).unwrap();
    assert!(matches!(
        store.commit_selected(&mut transaction, CommitToken(1)),
        Err(DurableKvError::Transaction(
            TransactionError::NoSelectedTimeline
        ))
    ));
    let mut transaction = selected(effects());
    transaction.rollback().unwrap();
    assert!(matches!(
        store.commit_selected(&mut transaction, CommitToken(1)),
        Err(DurableKvError::Transaction(TransactionError::RolledBack))
    ));
    assert_eq!(scratch.snapshot(), baseline);
    assert_values(&store, false);
}

#[test]
fn denied_batches_have_no_prefix_and_retain_terminal_failure_across_restart() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    let token = CommitToken(77);
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    let denied = vec![
        set("answer", b"prefix-must-not-appear"),
        EffectIntent::NetworkSend {
            endpoint: "synthetic-never-connected".into(),
            bytes: vec![1],
        },
    ];
    let mut transaction = selected(denied.clone());
    let Err(DurableKvError::PreflightRejected { message, .. }) =
        store.commit_selected(&mut transaction, token)
    else {
        panic!("expected bounded preflight failure")
    };
    assert_values(&store, false);
    let baseline = scratch.snapshot();
    drop(store);
    let mut store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    assert!(
        matches!(store.recover(token).unwrap(),DurableRecovery::KnownNotApplied { receipt:Some(_),terminal_failure:Some(ref stored) } if *stored==message)
    );
    let mut replay = selected(denied);
    assert!(
        matches!(store.commit_selected(&mut replay,token),Err(DurableKvError::PreflightRejected { message:stored, .. }) if stored==message)
    );
    assert_eq!(scratch.snapshot(), baseline);
    // A terminal known-not-applied record does not block a later selected token.
    store
        .commit_selected(&mut selected(effects()), CommitToken(78))
        .unwrap();
    assert_values(&store, true);
    drop(store);
    let store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    assert_values(&store, true);
}

#[test]
fn native_and_other_custom_operations_never_escape_the_managed_namespace() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let mut store = DurableCommitStore::create(scratch.path(), policy()).unwrap();
    let sentinel = scratch.path().join("sentinel");
    std::fs::write(&sentinel, b"unchanged").unwrap();
    for (index, intent) in [
        EffectIntent::FileWrite {
            path: sentinel.to_string_lossy().into(),
            offset: 0,
            bytes: b"changed".to_vec(),
            initial: FrozenFileSnapshot {
                path: sentinel.to_string_lossy().into(),
                contents: Some(b"unchanged".to_vec()),
            },
        },
        EffectIntent::FileSetLength {
            path: sentinel.to_string_lossy().into(),
            length: 0,
            initial: FrozenFileSnapshot {
                path: sentinel.to_string_lossy().into(),
                contents: Some(b"unchanged".to_vec()),
            },
        },
        EffectIntent::ProcessSpawn {
            program: "synthetic-never-spawned".into(),
            arguments: vec![],
        },
        EffectIntent::Sleep {
            milliseconds: u64::MAX,
        },
        EffectIntent::Custom {
            namespace: "other".into(),
            payload: vec![],
        },
        set("../outside", b"logical-key-does-not-authorize-path"),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(matches!(
            store.commit_selected(
                &mut selected(vec![intent]),
                CommitToken(index as u128 + 100)
            ),
            Err(DurableKvError::PreflightRejected { .. })
        ));
    }
    assert_eq!(std::fs::read(sentinel).unwrap(), b"unchanged");
    assert_values(&store, false);
}

#[test]
#[ignore]
fn crash_worker() {
    let path = std::env::var("OURO_KV_CRASH_DIRECTORY").unwrap();
    let index: usize = std::env::var("OURO_KV_CRASH_PHASE")
        .unwrap()
        .parse()
        .unwrap();
    let target = phases()[index];
    let kill = std::env::var("OURO_KV_CRASH_KIND").unwrap() == "kill";
    let mut store = DurableCommitStore::open(path, &policy()).unwrap();
    store
        .commit_selected_with_observer(&mut selected(effects()), CommitToken(555), |phase| {
            if phase == target {
                // SAFETY: only this fixture subprocess terminates, without cleanup.
                unsafe {
                    if kill {
                        libc::kill(libc::getpid(), libc::SIGKILL);
                    }
                    libc::_exit(86)
                }
            }
        })
        .unwrap();
    panic!("crash phase was not reached");
}

#[test]
fn subprocess_crashes_recover_whole_images_at_every_publication_boundary() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for (index, phase) in phases().into_iter().enumerate() {
        let scratch = Scratch::new();
        let config = policy();
        drop(DurableCommitStore::create(scratch.path(), config.clone()).unwrap());
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_worker", "--ignored", "--nocapture"])
            .env("OURO_KV_CRASH_DIRECTORY", scratch.path())
            .env("OURO_KV_CRASH_PHASE", index.to_string())
            .env(
                "OURO_KV_CRASH_KIND",
                if index % 2 == 0 { "exit" } else { "kill" },
            )
            .status()
            .unwrap();
        assert!(
            status.code() == Some(86) || status.signal() == Some(libc::SIGKILL),
            "phase {phase:?}: {status:?}"
        );
        let mut store = DurableCommitStore::open(scratch.path(), &config).unwrap();
        let applied = index >= 4;
        assert_values(&store, applied);
        let recovery = store.recover(CommitToken(555)).unwrap();
        assert!(matches!(recovery, DurableRecovery::KnownApplied(_)) == applied);
        if (1..=3).contains(&index) {
            assert!(matches!(
                store.commit_selected(&mut selected(effects()), CommitToken(556)),
                Err(DurableKvError::PendingToken(CommitToken(555)))
            ));
        }
        assert!(!scratch.path().join("store.next").exists());
        store
            .commit_selected(&mut selected(effects()), CommitToken(555))
            .unwrap();
        assert_values(&store, true);
        let bytes = scratch.snapshot();
        store
            .commit_selected(&mut selected(effects()), CommitToken(555))
            .unwrap();
        assert_eq!(scratch.snapshot(), bytes);
        drop(store);
        assert_values(
            &DurableCommitStore::open(scratch.path(), &config).unwrap(),
            true,
        );
    }
}

#[test]
fn interrupted_session_is_unresolved_until_explicit_reopen() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for phase in phases() {
        let scratch = Scratch::new();
        let config = policy();
        let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store
                .commit_selected_with_observer(&mut selected(effects()), CommitToken(31), |seen| {
                    if seen == phase {
                        panic!("controlled interruption")
                    }
                })
                .unwrap();
        }));
        assert!(result.is_err());
        assert!(matches!(
            store.recover(CommitToken(31)),
            Err(DurableKvError::Unresolved { .. })
        ));
        assert!(matches!(
            store.read("answer"),
            Err(DurableKvError::Unresolved { .. })
        ));
        assert!(matches!(
            store.commit_selected(&mut selected(effects()), CommitToken(31)),
            Err(DurableKvError::Unresolved { .. })
        ));
        drop(store);
        let mut store = DurableCommitStore::open(scratch.path(), &config).unwrap();
        store
            .commit_selected(&mut selected(effects()), CommitToken(31))
            .unwrap();
        assert_values(&store, true);
    }
}

#[test]
fn policy_identity_and_every_configurable_limit_are_retained_across_reopen() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    drop(DurableCommitStore::create(scratch.path(), config.clone()).unwrap());
    for field in 0..6 {
        let mut changed = config.clone();
        match field {
            0 => changed.store_id[0] ^= 1,
            1 => {
                changed.allowed_keys.insert("new-capability".into());
            }
            2 => changed.max_snapshot_bytes -= 1,
            3 => changed.max_records -= 1,
            4 => changed.max_effects_per_batch -= 1,
            _ => changed.max_value_bytes -= 1,
        }
        let result = DurableCommitStore::open(scratch.path(), &changed);
        assert!(
            matches!(result, Err(DurableKvError::PolicyMismatch)),
            "policy field {field}: {:?}",
            result.err()
        );
    }
    let store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    assert!(matches!(
        DurableCommitStore::open(scratch.path(), &config),
        Err(DurableKvError::Busy)
    ));
    drop(store);
    std::fs::remove_file(scratch.path().join("store.bin")).unwrap();
    assert!(matches!(
        DurableCommitStore::open(scratch.path(), &config),
        Err(DurableKvError::Unresolved { .. })
    ));
    assert!(matches!(
        DurableCommitStore::create(scratch.path(), config),
        Err(DurableKvError::AlreadyExists)
    ));
}

fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.durable-managed-kv/v1\0");
    hash.update(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash.finalize());
}
#[test]
fn corruption_truncation_and_resealed_semantic_inconsistency_cannot_report_applied() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    store
        .commit_selected(&mut selected(effects()), CommitToken(44))
        .unwrap();
    drop(store);
    let baseline = scratch.snapshot();
    for index in 0..baseline.len() {
        let mut changed = baseline.clone();
        changed[index] ^= 1;
        scratch.replace_snapshot(&changed);
        assert!(
            DurableCommitStore::open(scratch.path(), &config).is_err(),
            "corrupt byte {index}"
        );
        scratch.replace_snapshot(&baseline[..index]);
        assert!(
            DurableCommitStore::open(scratch.path(), &config).is_err(),
            "truncation {index}"
        );
    }
    let mut changed = baseline.clone();
    let value = changed
        .windows(b"answer-value".len())
        .position(|bytes| bytes == b"answer-value")
        .unwrap();
    changed[value] = b'X';
    reseal(&mut changed);
    scratch.replace_snapshot(&changed);
    assert!(matches!(
        DurableCommitStore::open(scratch.path(), &config),
        Err(DurableKvError::InvalidSnapshot(
            "managed values disagree with acknowledged history"
        ))
    ));
    // Valid checksum plus an impossible count must fail before allocation.
    let mut changed = baseline.clone();
    let policy_key_count = 16 + 16 + 4 * 8;
    changed[policy_key_count..policy_key_count + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut changed);
    scratch.replace_snapshot(&changed);
    assert!(DurableCommitStore::open(scratch.path(), &config).is_err());
    let mut trailing = baseline.clone();
    trailing.push(0);
    scratch.replace_snapshot(&trailing);
    assert!(DurableCommitStore::open(scratch.path(), &config).is_err());
    scratch.replace_snapshot(&baseline);
    assert_values(
        &DurableCommitStore::open(scratch.path(), &config).unwrap(),
        true,
    );
}

#[test]
fn size_is_measured_before_commit_copy_and_tombstones_are_never_evicted() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let mut config = policy();
    config.max_snapshot_bytes = 1024;
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    let baseline = scratch.snapshot();
    let mut transaction = selected(vec![set("answer", &vec![7; 500])]);
    assert!(matches!(
        store.commit_selected(&mut transaction, CommitToken(1)),
        Err(DurableKvError::ResourceLimit(_))
    ));
    assert!(
        transaction.begin_candidate().is_ok(),
        "oversized selection was prematurely marked committed"
    );
    assert_eq!(scratch.snapshot(), baseline);
    assert_values(&store, false);
    drop(store);
    let other = Scratch::new();
    let mut config = policy();
    config.max_records = 1;
    let mut store = DurableCommitStore::create(other.path(), config).unwrap();
    store
        .commit_selected(&mut selected(effects()), CommitToken(2))
        .unwrap();
    let baseline = other.snapshot();
    assert!(matches!(
        store.commit_selected(&mut selected(effects()), CommitToken(3)),
        Err(DurableKvError::ResourceLimit(_))
    ));
    store
        .commit_selected(&mut selected(effects()), CommitToken(2))
        .unwrap();
    assert_eq!(other.snapshot(), baseline);
    let mut invalid = policy();
    invalid.max_snapshot_bytes = MAX_DURABLE_KV_BYTES + 1;
    assert!(matches!(
        DurableCommitStore::create(Scratch::new().path(), invalid),
        Err(DurableKvError::InvalidPolicy)
    ));
}

#[test]
fn symlink_staging_and_replaced_lock_fail_closed_without_touching_targets() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    let sentinel = scratch.path().join("sentinel");
    std::fs::write(&sentinel, b"unchanged").unwrap();
    std::fs::set_permissions(&sentinel, std::fs::Permissions::from_mode(0o600)).unwrap();
    symlink(&sentinel, scratch.path().join("store.next")).unwrap();
    assert!(matches!(
        store.commit_selected(&mut selected(effects()), CommitToken(1)),
        Err(DurableKvError::Unresolved { .. })
    ));
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"unchanged");
    drop(store);
    assert!(matches!(
        DurableCommitStore::open(scratch.path(), &config),
        Err(DurableKvError::Unresolved { .. })
    ));
    std::fs::remove_file(scratch.path().join("store.next")).unwrap();
    let store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    std::fs::rename(
        scratch.path().join("store.lock"),
        scratch.path().join("old.lock"),
    )
    .unwrap();
    std::fs::write(scratch.path().join("store.lock"), b"").unwrap();
    std::fs::set_permissions(
        scratch.path().join("store.lock"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(matches!(
        store.recover(CommitToken(1)),
        Err(DurableKvError::Unresolved { .. })
    ));
}

#[test]
fn pending_record_order_and_delete_commands_are_checked_on_restart() {
    let _session = TEST_SESSION
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let scratch = Scratch::new();
    let config = policy();
    let first = CommitToken(0x0123456789abcdef_1122334455667788);
    let second = CommitToken(0xfedcba9876543210_8877665544332211);
    let mut store = DurableCommitStore::create(scratch.path(), config.clone()).unwrap();
    store
        .commit_selected(&mut selected(effects()), first)
        .unwrap();
    let delete = DurableKvIntent::Delete {
        key: "branch".into(),
    }
    .into_effect()
    .unwrap();
    store
        .commit_selected(
            &mut selected(vec![delete, set("answer", b"answer-value")]),
            second,
        )
        .unwrap();
    assert_eq!(store.read("branch").unwrap(), None);
    drop(store);
    let store = DurableCommitStore::open(scratch.path(), &config).unwrap();
    assert_eq!(store.read("branch").unwrap(), None);
    drop(store);
    let mut changed = scratch.snapshot();
    let token = first.0.to_le_bytes();
    let status = changed
        .windows(16)
        .position(|bytes| bytes == token)
        .unwrap()
        + 40;
    assert_eq!(changed[status], 1);
    changed[status] = 0;
    reseal(&mut changed);
    scratch.replace_snapshot(&changed);
    assert!(matches!(
        DurableCommitStore::open(scratch.path(), &config),
        Err(DurableKvError::InvalidSnapshot("pending batch is not last"))
    ));
}
