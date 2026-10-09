//! One Linux transactional effect profile in a controlled private directory.
//!
//! Logical key/value effects and their exact selected-batch ledger share one
//! versioned bounded snapshot. A pending record is synced before dispatch;
//! replacement values and Applied acknowledgment are published together by
//! file-sync/renameat/directory-sync. Reopen reconstructs values from Applied
//! commands. No native file/network/process/sleep effect enters this profile.
//!
//! Qualification concerns process crashes on a cooperating local filesystem.
//! Power-loss durability depends on filesystem/device flush guarantees. Locks
//! are advisory; other writers, network filesystems and authenticity are outside
//! this profile. SHA256 is an integrity check, not an authenticity assertion.

use super::transaction::{
    CommitLog, CommitOutcome, CommitReceipt, CommitToken, CommittedBatch, EffectApplicationStatus,
    EffectIntent, FrozenEndpointTape, FrozenFileSnapshot, FrozenProcessResult,
    ObservationTranscript, TemporalTransaction, TimelineCandidate, TimelineId, TransactionError,
};
use crate::core::{OutputItem, Provenance, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

pub const DURABLE_KV_VERSION: u16 = 1;
pub const DURABLE_KV_NAMESPACE: &str = "ourochronos.durable-kv/v1";
pub const MAX_DURABLE_KV_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 10_000;
const MAX_EFFECTS: usize = 1_000;
const MAX_KEY: usize = 256;
const MAX_VALUE: usize = 1024 * 1024;
const MAX_ITEMS: usize = 100_000;
const MAX_TEXT: usize = 16 * 1024;
const MAX_FAILURE: usize = 4096;
const MAGIC: &[u8; 8] = b"OUROKV\0\0";
const HEADER: usize = 16;
const CHECKSUM: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableKvConfig {
    pub store_id: [u8; 16],
    pub allowed_keys: BTreeSet<String>,
    pub max_snapshot_bytes: usize,
    pub max_records: usize,
    pub max_effects_per_batch: usize,
    pub max_value_bytes: usize,
}

impl DurableKvConfig {
    pub fn new(store_id: [u8; 16], keys: impl IntoIterator<Item = String>) -> Self {
        Self {
            store_id,
            allowed_keys: keys.into_iter().collect(),
            max_snapshot_bytes: MAX_DURABLE_KV_BYTES,
            max_records: MAX_RECORDS,
            max_effects_per_batch: MAX_EFFECTS,
            max_value_bytes: MAX_VALUE,
        }
    }
    fn validate(&self) -> Result<(), DurableKvError> {
        if self.store_id == [0; 16]
            || self.allowed_keys.len() > MAX_RECORDS
            || self.allowed_keys.iter().any(|key| !valid_key(key))
            || self.max_snapshot_bytes > MAX_DURABLE_KV_BYTES
            || self.max_snapshot_bytes < HEADER + CHECKSUM
            || self.max_records == 0
            || self.max_records > MAX_RECORDS
            || self.max_effects_per_batch > MAX_EFFECTS
            || self.max_value_bytes > MAX_VALUE
        {
            return Err(DurableKvError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableKvIntent {
    Set { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl DurableKvIntent {
    pub fn into_effect(self) -> Result<EffectIntent, DurableKvError> {
        let (tag, key, value) = match self {
            Self::Set { key, value } => (1, key, Some(value)),
            Self::Delete { key } => (2, key, None),
        };
        if !valid_key(&key) || value.as_ref().is_some_and(|value| value.len() > MAX_VALUE) {
            return Err(DurableKvError::ResourceLimit("managed command"));
        }
        let mut payload = vec![tag];
        payload.extend_from_slice(&(key.len() as u32).to_le_bytes());
        payload.extend_from_slice(key.as_bytes());
        if let Some(value) = value {
            payload.extend_from_slice(&(value.len() as u32).to_le_bytes());
            payload.extend_from_slice(&value);
        }
        Ok(EffectIntent::Custom {
            namespace: DURABLE_KV_NAMESPACE.into(),
            payload,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableRecovery {
    KnownApplied(CommitReceipt),
    KnownNotApplied {
        receipt: Option<CommitReceipt>,
        terminal_failure: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurableCommitPhase {
    RecordTempSynced,
    RecordRenamed,
    RecordDurable,
    ApplyTempSynced,
    ApplyRenamed,
    ApplyDurable,
}

#[derive(Debug)]
pub enum DurableKvError {
    UnsupportedPlatform,
    InvalidPolicy,
    PolicyMismatch,
    AlreadyExists,
    Busy,
    ResourceLimit(&'static str),
    InvalidSnapshot(&'static str),
    IntegrityMismatch,
    Transaction(TransactionError),
    PendingToken(CommitToken),
    PreflightRejected {
        receipt: CommitReceipt,
        message: String,
    },
    KnownNotApplied {
        message: String,
    },
    Unresolved {
        message: String,
    },
}

impl fmt::Display for DurableKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("durable managed KV profile requires Linux"),
            Self::InvalidPolicy => f.write_str("invalid durable managed KV policy"),
            Self::PolicyMismatch => f.write_str("durable store identity, keys or limits changed"),
            Self::AlreadyExists => {
                f.write_str("durable store creation requires a fresh controlled directory")
            }
            Self::Busy => f.write_str("durable store already has an exclusive owner"),
            Self::ResourceLimit(what) => write!(f, "durable {what} exceeds its bounded profile"),
            Self::InvalidSnapshot(what) => {
                write!(f, "invalid durable snapshot: {what}; outcome unresolved")
            }
            Self::IntegrityMismatch => {
                f.write_str("durable snapshot SHA256 mismatch; outcome unresolved")
            }
            Self::Transaction(error) => error.fmt(f),
            Self::PendingToken(token) => {
                write!(f, "durable pending token {token:?} must be resolved first")
            }
            Self::PreflightRejected { message, .. } => write!(
                f,
                "durable preflight rejected without managed effects: {message}"
            ),
            Self::KnownNotApplied { message } => {
                write!(f, "durable managed effects known not applied: {message}")
            }
            Self::Unresolved { message } => {
                write!(f, "durable managed outcome unresolved: {message}")
            }
        }
    }
}
impl Error for DurableKvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Transaction(error) => Some(error),
            _ => None,
        }
    }
}
impl From<TransactionError> for DurableKvError {
    fn from(error: TransactionError) -> Self {
        Self::Transaction(error)
    }
}

#[derive(Clone)]
struct Snapshot {
    config: DurableKvConfig,
    generation: u64,
    log: CommitLog,
    values: BTreeMap<String, Vec<u8>>,
}

pub struct DurableCommitStore {
    host: Host,
    snapshot: Snapshot,
    poisoned: bool,
}

impl DurableCommitStore {
    pub fn create(
        directory: impl AsRef<Path>,
        config: DurableKvConfig,
    ) -> Result<Self, DurableKvError> {
        config.validate()?;
        let snapshot = Snapshot {
            config,
            generation: 0,
            log: CommitLog::default(),
            values: BTreeMap::new(),
        };
        // Even initialization is measured before creating host objects.
        let bytes = encode_snapshot(&snapshot)?;
        let host = Host::acquire(directory.as_ref())?;
        if !host.new_lock
            || host.load(snapshot.config.max_snapshot_bytes)?.is_some()
            || host.temp_exists()?
        {
            return Err(DurableKvError::AlreadyExists);
        }
        let mut store = Self {
            host,
            snapshot,
            poisoned: false,
        };
        store.poisoned = true;
        store
            .host
            .publish(&bytes, false, &mut |_| {})
            .map_err(|error| DurableKvError::Unresolved {
                message: error.message,
            })?;
        store.poisoned = false;
        Ok(store)
    }

    pub fn open(
        directory: impl AsRef<Path>,
        expected: &DurableKvConfig,
    ) -> Result<Self, DurableKvError> {
        expected.validate()?;
        let host = Host::acquire(directory.as_ref())?;
        if host.new_lock {
            return Err(DurableKvError::Unresolved {
                message: "missing retained lock identity".into(),
            });
        }
        let bytes =
            host.load(expected.max_snapshot_bytes)?
                .ok_or_else(|| DurableKvError::Unresolved {
                    message: "initialized snapshot is missing".into(),
                })?;
        let snapshot = decode_snapshot(&bytes, expected)?;
        validate_history(&snapshot)?;
        host.clean_temp()?;
        // Reconcile a formerly unacknowledged directory replacement before
        // returning any known application status from the recovered image.
        host.sync_directory()?;
        Ok(Self {
            host,
            snapshot,
            poisoned: false,
        })
    }

    pub fn config(&self) -> &DurableKvConfig {
        &self.snapshot.config
    }
    pub fn read(&self, key: &str) -> Result<Option<&[u8]>, DurableKvError> {
        self.check_session()?;
        if !self.snapshot.config.allowed_keys.contains(key) {
            return Err(DurableKvError::InvalidPolicy);
        }
        Ok(self.snapshot.values.get(key).map(Vec::as_slice))
    }
    pub fn recover(&self, token: CommitToken) -> Result<DurableRecovery, DurableKvError> {
        self.check_session()?;
        let Some(batch) = self
            .snapshot
            .log
            .batches()
            .iter()
            .find(|batch| batch.receipt.token == token)
        else {
            return Ok(DurableRecovery::KnownNotApplied {
                receipt: None,
                terminal_failure: None,
            });
        };
        match self.snapshot.log.application_status(token) {
            Some(EffectApplicationStatus::Applied) => {
                Ok(DurableRecovery::KnownApplied(batch.receipt.clone()))
            }
            Some(EffectApplicationStatus::NotAttempted) => Ok(DurableRecovery::KnownNotApplied {
                receipt: Some(batch.receipt.clone()),
                terminal_failure: None,
            }),
            Some(EffectApplicationStatus::Failed(message)) => {
                Ok(DurableRecovery::KnownNotApplied {
                    receipt: Some(batch.receipt.clone()),
                    terminal_failure: Some(message.clone()),
                })
            }
            _ => Err(DurableKvError::Unresolved {
                message: "application acknowledgment is unavailable".into(),
            }),
        }
    }
    fn check_session(&self) -> Result<(), DurableKvError> {
        if self.poisoned {
            Err(DurableKvError::Unresolved {
                message: "publication interrupted; reopen and reconcile before retry".into(),
            })
        } else {
            self.host.verify_lock()
        }
    }
    pub fn commit_selected(
        &mut self,
        transaction: &mut TemporalTransaction,
        token: CommitToken,
    ) -> Result<CommitOutcome, DurableKvError> {
        self.commit_selected_with_observer(transaction, token, |_| {})
    }

    /// An observer supports controlled crash injection and monitoring, never
    /// acts as an effect adapter. A panic poisons the session until reopen.
    pub fn commit_selected_with_observer(
        &mut self,
        transaction: &mut TemporalTransaction,
        token: CommitToken,
        mut observer: impl FnMut(DurableCommitPhase),
    ) -> Result<CommitOutcome, DurableKvError> {
        self.check_session()?;
        let (timeline, candidate) = transaction.selected_candidate_for_commit(token)?;
        let config = &self.snapshot.config;
        // Validate and measure borrowed data before core commit clones anything.
        measure_candidate(candidate, config)?;
        if let Some(pending) = self
            .snapshot
            .log
            .durable_entries()
            .find_map(|(batch, status)| {
                (status == &EffectApplicationStatus::NotAttempted).then_some(batch.receipt.token)
            })
        {
            if pending != token {
                return Err(DurableKvError::PendingToken(pending));
            }
        }
        let existing = self
            .snapshot
            .log
            .batches()
            .iter()
            .find(|batch| batch.receipt.token == token);
        if existing.is_some() {
            let outcome = transaction.commit_selected(token, &mut self.snapshot.log)?;
            match self.snapshot.log.application_status(token) {
                Some(EffectApplicationStatus::Applied) => return Ok(outcome),
                Some(EffectApplicationStatus::Failed(message)) => {
                    let receipt = receipt(&outcome).clone();
                    return Err(DurableKvError::PreflightRejected {
                        receipt,
                        message: message.clone(),
                    });
                }
                Some(EffectApplicationStatus::NotAttempted) => {}
                _ => {
                    return Err(DurableKvError::Unresolved {
                        message: "unacknowledged application".into(),
                    })
                }
            }
            self.apply_pending(token, &mut observer)?;
            return Ok(outcome);
        }
        if self.snapshot.log.batches().len() >= config.max_records {
            return Err(DurableKvError::ResourceLimit("retained token tombstones"));
        }
        let provisional = CommitReceipt {
            token,
            timeline,
            batch_digest: timeline.0,
            sequence: self.snapshot.log.batches().len(),
        };
        let commands = commands(&candidate.effects, config);
        let status = match &commands {
            Ok(_) => EffectApplicationStatus::NotAttempted,
            Err(message) => EffectApplicationStatus::Failed(message.clone()),
        };
        let values = borrowed_values(&self.snapshot.values);
        measure_view(
            &self.snapshot,
            Some((&provisional, candidate, &status)),
            None,
            &values,
        )?;
        if let Ok(commands) = &commands {
            let mut changed = values;
            apply_borrowed(&mut changed, commands);
            measure_view(
                &self.snapshot,
                Some((&provisional, candidate, &EffectApplicationStatus::Applied)),
                None,
                &changed,
            )?;
        }
        let mut recorded = self.snapshot.clone();
        let outcome = transaction.commit_selected(token, &mut recorded.log)?;
        recorded
            .log
            .set_durable_application(token, status.clone())?;
        recorded.generation += 1;
        self.publish(recorded, false, &mut observer)?;
        if let EffectApplicationStatus::Failed(message) = status {
            return Err(DurableKvError::PreflightRejected {
                receipt: receipt(&outcome).clone(),
                message,
            });
        }
        self.apply_pending(token, &mut observer)?;
        Ok(outcome)
    }

    fn apply_pending(
        &mut self,
        token: CommitToken,
        observer: &mut impl FnMut(DurableCommitPhase),
    ) -> Result<(), DurableKvError> {
        let batch = self
            .snapshot
            .log
            .batches()
            .iter()
            .find(|batch| batch.receipt.token == token)
            .ok_or(DurableKvError::InvalidSnapshot("pending batch absent"))?;
        let parsed = commands(&batch.effects, &self.snapshot.config)
            .map_err(|_| DurableKvError::InvalidSnapshot("pending commands invalid"))?;
        let mut values = borrowed_values(&self.snapshot.values);
        apply_borrowed(&mut values, &parsed);
        measure_view(
            &self.snapshot,
            None,
            Some((token, &EffectApplicationStatus::Applied)),
            &values,
        )?;
        let mut applied = self.snapshot.clone();
        for command in parsed {
            match command {
                Command::Set(key, value) => {
                    applied.values.insert(key.into(), value.to_vec());
                }
                Command::Delete(key) => {
                    applied.values.remove(key);
                }
            }
        }
        applied
            .log
            .set_durable_application(token, EffectApplicationStatus::Applied)?;
        applied.generation += 1;
        self.publish(applied, true, observer)
    }
    fn publish(
        &mut self,
        snapshot: Snapshot,
        applying: bool,
        observer: &mut impl FnMut(DurableCommitPhase),
    ) -> Result<(), DurableKvError> {
        let bytes = encode_snapshot(&snapshot)?;
        self.poisoned = true;
        match self.host.publish(&bytes, applying, observer) {
            Ok(()) => {
                self.snapshot = snapshot;
                self.poisoned = false;
                Ok(())
            }
            Err(error) if !error.renamed => {
                self.poisoned = false;
                Err(DurableKvError::KnownNotApplied {
                    message: error.message,
                })
            }
            Err(error) => Err(DurableKvError::Unresolved {
                message: error.message,
            }),
        }
    }
}

fn receipt(outcome: &CommitOutcome) -> &CommitReceipt {
    match outcome {
        CommitOutcome::Committed(receipt) | CommitOutcome::AlreadyCommitted(receipt) => receipt,
    }
}
fn valid_key(key: &str) -> bool {
    !key.is_empty() && key.len() <= MAX_KEY && !key.contains('\0')
}
#[derive(Clone, Copy)]
enum Command<'a> {
    Set(&'a str, &'a [u8]),
    Delete(&'a str),
}
fn commands<'a>(
    effects: &'a [EffectIntent],
    config: &DurableKvConfig,
) -> Result<Vec<Command<'a>>, String> {
    if effects.len() > config.max_effects_per_batch {
        return Err("managed effect count exceeds policy".into());
    }
    let mut result = Vec::new();
    for effect in effects {
        let EffectIntent::Custom { namespace, payload } = effect else {
            return Err("only managed KV custom intents are supported".into());
        };
        if namespace != DURABLE_KV_NAMESPACE {
            return Err("custom namespace is outside the managed profile".into());
        }
        let mut reader = Reader::new(payload);
        let command = (|| {
            let tag = reader.u8()?;
            let key = reader.text(MAX_KEY)?;
            if !valid_key(key) || !config.allowed_keys.contains(key) {
                return Err(DurableKvError::InvalidPolicy);
            }
            let command = match tag {
                1 => Command::Set(key, reader.blob(config.max_value_bytes)?),
                2 => Command::Delete(key),
                _ => return Err(DurableKvError::InvalidSnapshot("managed command tag")),
            };
            if !reader.remaining().is_empty() {
                return Err(DurableKvError::InvalidSnapshot("managed command suffix"));
            }
            Ok(command)
        })()
        .map_err(|_| "managed command, key capability, or value bound rejected".to_string())?;
        result.push(command);
    }
    Ok(result)
}
fn borrowed_values(values: &BTreeMap<String, Vec<u8>>) -> BTreeMap<&str, &[u8]> {
    values
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_slice()))
        .collect()
}
fn apply_borrowed<'a>(values: &mut BTreeMap<&'a str, &'a [u8]>, commands: &[Command<'a>]) {
    for command in commands {
        match command {
            Command::Set(key, value) => {
                values.insert(key, value);
            }
            Command::Delete(key) => {
                values.remove(key);
            }
        }
    }
}

fn measure_candidate(
    candidate: &TimelineCandidate,
    config: &DurableKvConfig,
) -> Result<(), DurableKvError> {
    let receipt = CommitReceipt {
        token: CommitToken(0),
        timeline: TimelineId(0),
        batch_digest: 0,
        sequence: 0,
    };
    let mut writer = Writer::measure(config.max_snapshot_bytes);
    encode_batch(
        &mut writer,
        BatchView {
            receipt: &receipt,
            state: &candidate.state,
            output: &candidate.output,
            effects: &candidate.effects,
            inputs: &candidate.inputs_consumed,
            observations: &candidate.observations,
            status: &EffectApplicationStatus::NotAttempted,
        },
    )
}
fn measure_view(
    snapshot: &Snapshot,
    extra: Option<(&CommitReceipt, &TimelineCandidate, &EffectApplicationStatus)>,
    status: Option<(CommitToken, &EffectApplicationStatus)>,
    values: &BTreeMap<&str, &[u8]>,
) -> Result<usize, DurableKvError> {
    let mut writer = Writer::measure(snapshot.config.max_snapshot_bytes);
    encode_view(&mut writer, snapshot, extra, status, values)?;
    Ok(writer.length)
}
fn encode_snapshot(snapshot: &Snapshot) -> Result<Vec<u8>, DurableKvError> {
    let values = borrowed_values(&snapshot.values);
    let size = measure_view(snapshot, None, None, &values)?;
    let mut writer = Writer::buffer(snapshot.config.max_snapshot_bytes, size)?;
    encode_view(&mut writer, snapshot, None, None, &values)?;
    let mut bytes = writer
        .bytes
        .ok_or(DurableKvError::InvalidSnapshot("missing encoding buffer"))?;
    let end = bytes.len() - CHECKSUM;
    let body = u32::try_from(end - HEADER)
        .map_err(|_| DurableKvError::ResourceLimit("snapshot length"))?;
    bytes[12..16].copy_from_slice(&body.to_le_bytes());
    let hash = snapshot_checksum(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash);
    Ok(bytes)
}
fn encode_view(
    writer: &mut Writer,
    snapshot: &Snapshot,
    extra: Option<(&CommitReceipt, &TimelineCandidate, &EffectApplicationStatus)>,
    status_override: Option<(CommitToken, &EffectApplicationStatus)>,
    values: &BTreeMap<&str, &[u8]>,
) -> Result<(), DurableKvError> {
    writer.raw(MAGIC)?;
    writer.u16(DURABLE_KV_VERSION)?;
    writer.u16(0)?;
    writer.u32(0)?;
    encode_policy(writer, &snapshot.config)?;
    writer.u64(snapshot.generation)?;
    writer.count(values.len(), snapshot.config.allowed_keys.len())?;
    for (key, value) in values {
        if !valid_key(key) || !snapshot.config.allowed_keys.contains(*key) {
            return Err(DurableKvError::InvalidSnapshot("value key outside policy"));
        }
        writer.text(key, MAX_KEY)?;
        writer.blob(value, snapshot.config.max_value_bytes)?;
    }
    writer.count(
        snapshot.log.batches().len() + usize::from(extra.is_some()),
        snapshot.config.max_records,
    )?;
    for (batch, status) in snapshot.log.durable_entries() {
        let status = status_override
            .filter(|(token, _)| *token == batch.receipt.token)
            .map_or(status, |(_, status)| status);
        encode_batch(
            writer,
            BatchView {
                receipt: &batch.receipt,
                state: &batch.state,
                output: &batch.output,
                effects: &batch.effects,
                inputs: &batch.inputs_consumed,
                observations: &batch.observations,
                status,
            },
        )?;
    }
    if let Some((receipt, candidate, status)) = extra {
        encode_batch(
            writer,
            BatchView {
                receipt,
                state: &candidate.state,
                output: &candidate.output,
                effects: &candidate.effects,
                inputs: &candidate.inputs_consumed,
                observations: &candidate.observations,
                status,
            },
        )?;
    }
    writer.raw(&[0; CHECKSUM])
}
fn encode_policy(writer: &mut Writer, config: &DurableKvConfig) -> Result<(), DurableKvError> {
    config.validate()?;
    writer.raw(&config.store_id)?;
    for field in [
        config.max_snapshot_bytes,
        config.max_records,
        config.max_effects_per_batch,
        config.max_value_bytes,
    ] {
        writer.u64(field as u64)?;
    }
    writer.count(config.allowed_keys.len(), MAX_RECORDS)?;
    for key in &config.allowed_keys {
        writer.text(key, MAX_KEY)?;
    }
    Ok(())
}
struct BatchView<'a> {
    receipt: &'a CommitReceipt,
    state: &'a [(u64, Value)],
    output: &'a [OutputItem],
    effects: &'a [EffectIntent],
    inputs: &'a [u64],
    observations: &'a ObservationTranscript,
    status: &'a EffectApplicationStatus,
}
fn encode_batch(writer: &mut Writer, batch: BatchView<'_>) -> Result<(), DurableKvError> {
    let BatchView {
        receipt,
        state,
        output,
        effects,
        inputs,
        observations,
        status,
    } = batch;
    writer.raw(&receipt.token.0.to_le_bytes())?;
    writer.u64(receipt.timeline.0)?;
    writer.u64(receipt.batch_digest)?;
    writer.u64(receipt.sequence as u64)?;
    match status {
        EffectApplicationStatus::NotAttempted => writer.u8(0)?,
        EffectApplicationStatus::Applied => writer.u8(1)?,
        EffectApplicationStatus::Failed(message) => {
            writer.u8(2)?;
            writer.text(message, MAX_FAILURE)?;
        }
        EffectApplicationStatus::Applying => {
            return Err(DurableKvError::InvalidSnapshot(
                "unresolved status cannot be published",
            ))
        }
    }
    writer.count(state.len(), MAX_ITEMS)?;
    let mut previous = None;
    for (address, value) in state {
        if previous.is_some_and(|old| old >= *address) {
            return Err(DurableKvError::InvalidSnapshot(
                "noncanonical selected state",
            ));
        }
        previous = Some(*address);
        writer.u64(*address)?;
        encode_value(writer, value)?;
    }
    writer.count(output.len(), MAX_ITEMS)?;
    for item in output {
        match item {
            OutputItem::Val(value) => {
                writer.u8(0)?;
                encode_value(writer, value)?;
            }
            OutputItem::Char(byte) => {
                writer.u8(1)?;
                writer.u8(*byte)?;
            }
        }
    }
    writer.count(effects.len(), MAX_EFFECTS)?;
    for effect in effects {
        encode_effect(writer, effect)?;
    }
    writer.words(inputs)?;
    encode_observations(writer, observations)
}
fn encode_value(writer: &mut Writer, value: &Value) -> Result<(), DurableKvError> {
    writer.u64(value.val)?;
    writer.u8(u8::from(value.prov.saturated) | (u8::from(value.prov.deps.is_some()) << 1))?;
    if let Some(deps) = &value.prov.deps {
        writer.count(deps.len(), MAX_ITEMS)?;
        for address in deps.iter() {
            writer.u64(*address)?;
        }
    }
    Ok(())
}
fn encode_file(writer: &mut Writer, file: &FrozenFileSnapshot) -> Result<(), DurableKvError> {
    writer.text(&file.path, MAX_TEXT)?;
    match &file.contents {
        None => writer.u8(0)?,
        Some(bytes) => {
            writer.u8(1)?;
            writer.blob(bytes, MAX_VALUE)?;
        }
    }
    Ok(())
}
fn encode_effect(writer: &mut Writer, effect: &EffectIntent) -> Result<(), DurableKvError> {
    match effect {
        EffectIntent::FileWrite {
            path,
            offset,
            bytes,
            initial,
        } => {
            writer.u8(0)?;
            writer.text(path, MAX_TEXT)?;
            writer.u64(*offset)?;
            writer.blob(bytes, MAX_VALUE)?;
            encode_file(writer, initial)?;
        }
        EffectIntent::FileSetLength {
            path,
            length,
            initial,
        } => {
            writer.u8(1)?;
            writer.text(path, MAX_TEXT)?;
            writer.u64(*length)?;
            encode_file(writer, initial)?;
        }
        EffectIntent::NetworkSend { endpoint, bytes } => {
            writer.u8(2)?;
            writer.text(endpoint, MAX_TEXT)?;
            writer.blob(bytes, MAX_VALUE)?;
        }
        EffectIntent::ProcessSpawn { program, arguments } => {
            writer.u8(3)?;
            writer.text(program, MAX_TEXT)?;
            writer.count(arguments.len(), MAX_ITEMS)?;
            for argument in arguments {
                writer.text(argument, MAX_TEXT)?;
            }
        }
        EffectIntent::Sleep { milliseconds } => {
            writer.u8(4)?;
            writer.u64(*milliseconds)?;
        }
        EffectIntent::Custom { namespace, payload } => {
            writer.u8(5)?;
            writer.text(namespace, MAX_TEXT)?;
            writer.blob(payload, MAX_VALUE + MAX_TEXT + 16)?;
        }
    }
    Ok(())
}
fn encode_observations(
    writer: &mut Writer,
    observations: &ObservationTranscript,
) -> Result<(), DurableKvError> {
    writer.words(&observations.input)?;
    writer.words(&observations.clock)?;
    writer.words(&observations.random)?;
    writer.count(observations.files.len(), MAX_ITEMS)?;
    for file in &observations.files {
        encode_file(writer, file)?;
    }
    writer.count(observations.endpoints.len(), MAX_ITEMS)?;
    for endpoint in &observations.endpoints {
        writer.text(&endpoint.endpoint, MAX_TEXT)?;
        writer.blob(&endpoint.recv_bytes, MAX_VALUE)?;
    }
    writer.count(observations.processes.len(), MAX_ITEMS)?;
    for process in &observations.processes {
        writer.text(&process.command, MAX_TEXT)?;
        writer.blob(&process.output, MAX_VALUE)?;
        writer.raw(&process.exit_code.to_le_bytes())?;
    }
    Ok(())
}
fn snapshot_checksum(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ourochronos.durable-managed-kv/v1\0");
    hash.update(bytes);
    hash.finalize().into()
}

fn decode_snapshot(bytes: &[u8], expected: &DurableKvConfig) -> Result<Snapshot, DurableKvError> {
    if bytes.len() > expected.max_snapshot_bytes || bytes.len() > MAX_DURABLE_KV_BYTES {
        return Err(DurableKvError::ResourceLimit("snapshot bytes"));
    }
    if bytes.len() < HEADER + CHECKSUM || &bytes[..8] != MAGIC {
        return Err(DurableKvError::InvalidSnapshot("magic or length"));
    }
    let mut header = Reader::new(&bytes[8..HEADER]);
    if header.u16()? != DURABLE_KV_VERSION || header.u16()? != 0 {
        return Err(DurableKvError::InvalidSnapshot("version or reserved flags"));
    }
    let end = HEADER
        .checked_add(header.u32()? as usize)
        .ok_or(DurableKvError::InvalidSnapshot("body size"))?;
    if end.checked_add(CHECKSUM) != Some(bytes.len()) {
        return Err(DurableKvError::InvalidSnapshot(
            "body size or trailing bytes",
        ));
    }
    if snapshot_checksum(&bytes[..end]).as_slice() != &bytes[end..] {
        return Err(DurableKvError::IntegrityMismatch);
    }
    let mut reader = Reader::new(&bytes[HEADER..end]);
    let config = decode_policy(&mut reader)?;
    if &config != expected {
        return Err(DurableKvError::PolicyMismatch);
    }
    let generation = reader.u64()?;
    let count = reader.count(config.allowed_keys.len(), 9)?;
    let mut values = BTreeMap::new();
    let mut previous = None;
    for _ in 0..count {
        let key = reader.text(MAX_KEY)?;
        if !valid_key(key)
            || !config.allowed_keys.contains(key)
            || previous.is_some_and(|old| old >= key)
        {
            return Err(DurableKvError::InvalidSnapshot("managed value keys"));
        }
        let value = reader.blob(config.max_value_bytes)?;
        values.insert(key.to_string(), value.to_vec());
        previous = Some(key);
    }
    let count = reader.count(config.max_records, 81)?;
    let mut entries = reserve(count)?;
    for _ in 0..count {
        entries.push(decode_batch(&mut reader)?);
    }
    if !reader.remaining().is_empty() {
        return Err(DurableKvError::InvalidSnapshot("trailing snapshot fields"));
    }
    let log = CommitLog::restore_durable(entries)?;
    Ok(Snapshot {
        config,
        generation,
        log,
        values,
    })
}
fn decode_policy(reader: &mut Reader<'_>) -> Result<DurableKvConfig, DurableKvError> {
    let store_id = reader
        .take(16)?
        .try_into()
        .map_err(|_| DurableKvError::InvalidSnapshot("store ID"))?;
    let max_snapshot_bytes = reader.usize()?;
    let max_records = reader.usize()?;
    let max_effects_per_batch = reader.usize()?;
    let max_value_bytes = reader.usize()?;
    let count = reader.count(MAX_RECORDS, 5)?;
    let mut allowed_keys = BTreeSet::new();
    let mut previous = None;
    for _ in 0..count {
        let key = reader.text(MAX_KEY)?;
        if !valid_key(key) || previous.is_some_and(|old| old >= key) {
            return Err(DurableKvError::InvalidSnapshot("policy key ordering"));
        }
        allowed_keys.insert(key.to_string());
        previous = Some(key);
    }
    let config = DurableKvConfig {
        store_id,
        allowed_keys,
        max_snapshot_bytes,
        max_records,
        max_effects_per_batch,
        max_value_bytes,
    };
    config.validate()?;
    Ok(config)
}
fn decode_batch(
    reader: &mut Reader<'_>,
) -> Result<(CommittedBatch, EffectApplicationStatus), DurableKvError> {
    let token = CommitToken(u128::from_le_bytes(
        reader
            .take(16)?
            .try_into()
            .map_err(|_| DurableKvError::InvalidSnapshot("token"))?,
    ));
    let receipt = CommitReceipt {
        token,
        timeline: TimelineId(reader.u64()?),
        batch_digest: reader.u64()?,
        sequence: reader.usize()?,
    };
    let status = match reader.u8()? {
        0 => EffectApplicationStatus::NotAttempted,
        1 => EffectApplicationStatus::Applied,
        2 => EffectApplicationStatus::Failed(reader.text(MAX_FAILURE)?.into()),
        _ => return Err(DurableKvError::InvalidSnapshot("application status")),
    };
    let count = reader.count(MAX_ITEMS, 17)?;
    let mut state = reserve(count)?;
    let mut previous = None;
    for _ in 0..count {
        let address = reader.u64()?;
        if previous.is_some_and(|old| old >= address) {
            return Err(DurableKvError::InvalidSnapshot("selected state ordering"));
        }
        state.push((address, decode_value(reader)?));
        previous = Some(address);
    }
    let count = reader.count(MAX_ITEMS, 2)?;
    let mut output = reserve(count)?;
    for _ in 0..count {
        output.push(match reader.u8()? {
            0 => OutputItem::Val(decode_value(reader)?),
            1 => OutputItem::Char(reader.u8()?),
            _ => return Err(DurableKvError::InvalidSnapshot("typed output")),
        });
    }
    let count = reader.count(MAX_EFFECTS, 9)?;
    let mut effects = reserve(count)?;
    for _ in 0..count {
        effects.push(decode_effect(reader)?);
    }
    let inputs_consumed = reader.words()?;
    let observations = decode_observations(reader)?;
    Ok((
        CommittedBatch {
            receipt,
            state,
            output,
            effects,
            inputs_consumed,
            observations,
        },
        status,
    ))
}
fn decode_value(reader: &mut Reader<'_>) -> Result<Value, DurableKvError> {
    let val = reader.u64()?;
    let flags = reader.u8()?;
    if flags & !3 != 0 {
        return Err(DurableKvError::InvalidSnapshot("provenance flags"));
    }
    let deps = if flags & 2 != 0 {
        let count = reader.count(MAX_ITEMS, 8)?;
        let mut deps = BTreeSet::new();
        let mut previous = None;
        for _ in 0..count {
            let address = reader.u64()?;
            if previous.is_some_and(|old| old >= address) {
                return Err(DurableKvError::InvalidSnapshot("provenance ordering"));
            }
            deps.insert(address);
            previous = Some(address);
        }
        Some(Arc::new(deps))
    } else {
        None
    };
    Ok(Value {
        val,
        prov: Provenance {
            saturated: flags & 1 != 0,
            deps,
        },
    })
}
fn decode_file(reader: &mut Reader<'_>) -> Result<FrozenFileSnapshot, DurableKvError> {
    let path = reader.text(MAX_TEXT)?.into();
    let contents = match reader.u8()? {
        0 => None,
        1 => Some(reader.blob(MAX_VALUE)?.to_vec()),
        _ => return Err(DurableKvError::InvalidSnapshot("frozen file marker")),
    };
    Ok(FrozenFileSnapshot { path, contents })
}
fn decode_effect(reader: &mut Reader<'_>) -> Result<EffectIntent, DurableKvError> {
    Ok(match reader.u8()? {
        0 => EffectIntent::FileWrite {
            path: reader.text(MAX_TEXT)?.into(),
            offset: reader.u64()?,
            bytes: reader.blob(MAX_VALUE)?.to_vec(),
            initial: decode_file(reader)?,
        },
        1 => EffectIntent::FileSetLength {
            path: reader.text(MAX_TEXT)?.into(),
            length: reader.u64()?,
            initial: decode_file(reader)?,
        },
        2 => EffectIntent::NetworkSend {
            endpoint: reader.text(MAX_TEXT)?.into(),
            bytes: reader.blob(MAX_VALUE)?.to_vec(),
        },
        3 => {
            let program = reader.text(MAX_TEXT)?.into();
            let count = reader.count(MAX_ITEMS, 4)?;
            let mut arguments = reserve(count)?;
            for _ in 0..count {
                arguments.push(reader.text(MAX_TEXT)?.into());
            }
            EffectIntent::ProcessSpawn { program, arguments }
        }
        4 => EffectIntent::Sleep {
            milliseconds: reader.u64()?,
        },
        5 => EffectIntent::Custom {
            namespace: reader.text(MAX_TEXT)?.into(),
            payload: reader.blob(MAX_VALUE + MAX_TEXT + 16)?.to_vec(),
        },
        _ => return Err(DurableKvError::InvalidSnapshot("effect tag")),
    })
}
fn decode_observations(reader: &mut Reader<'_>) -> Result<ObservationTranscript, DurableKvError> {
    let input = reader.words()?;
    let clock = reader.words()?;
    let random = reader.words()?;
    let count = reader.count(MAX_ITEMS, 5)?;
    let mut files = reserve(count)?;
    for _ in 0..count {
        files.push(decode_file(reader)?);
    }
    let count = reader.count(MAX_ITEMS, 8)?;
    let mut endpoints = reserve(count)?;
    for _ in 0..count {
        endpoints.push(FrozenEndpointTape {
            endpoint: reader.text(MAX_TEXT)?.into(),
            recv_bytes: reader.blob(MAX_VALUE)?.to_vec(),
        });
    }
    let count = reader.count(MAX_ITEMS, 12)?;
    let mut processes = reserve(count)?;
    for _ in 0..count {
        processes.push(FrozenProcessResult {
            command: reader.text(MAX_TEXT)?.into(),
            output: reader.blob(MAX_VALUE)?.to_vec(),
            exit_code: i32::from_le_bytes(
                reader
                    .take(4)?
                    .try_into()
                    .map_err(|_| DurableKvError::InvalidSnapshot("process result"))?,
            ),
        });
    }
    Ok(ObservationTranscript {
        input,
        clock,
        random,
        files,
        endpoints,
        processes,
    })
}
fn validate_history(snapshot: &Snapshot) -> Result<(), DurableKvError> {
    let mut values = BTreeMap::new();
    let mut applied = 0u64;
    let mut pending = false;
    for (batch, status) in snapshot.log.durable_entries() {
        if pending {
            return Err(DurableKvError::InvalidSnapshot("pending batch is not last"));
        }
        match status {
            EffectApplicationStatus::Applied => {
                let commands = commands(&batch.effects, &snapshot.config).map_err(|_| {
                    DurableKvError::InvalidSnapshot("acknowledged commands violate policy")
                })?;
                apply_borrowed(&mut values, &commands);
                applied += 1;
            }
            EffectApplicationStatus::NotAttempted => {
                commands(&batch.effects, &snapshot.config).map_err(|_| {
                    DurableKvError::InvalidSnapshot("pending commands violate policy")
                })?;
                pending = true;
            }
            EffectApplicationStatus::Failed(_) => {}
            EffectApplicationStatus::Applying => {
                return Err(DurableKvError::InvalidSnapshot(
                    "unresolved dispatch marker",
                ))
            }
        }
    }
    if snapshot.generation != snapshot.log.batches().len() as u64 + applied {
        return Err(DurableKvError::InvalidSnapshot("publication generation"));
    }
    if values != borrowed_values(&snapshot.values) {
        return Err(DurableKvError::InvalidSnapshot(
            "managed values disagree with acknowledged history",
        ));
    }
    Ok(())
}

struct Writer {
    bytes: Option<Vec<u8>>,
    length: usize,
    limit: usize,
}
impl Writer {
    fn measure(limit: usize) -> Self {
        Self {
            bytes: None,
            length: 0,
            limit,
        }
    }
    fn buffer(limit: usize, size: usize) -> Result<Self, DurableKvError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| DurableKvError::ResourceLimit("encoding allocation"))?;
        Ok(Self {
            bytes: Some(bytes),
            length: 0,
            limit,
        })
    }
    fn raw(&mut self, bytes: &[u8]) -> Result<(), DurableKvError> {
        let length = self
            .length
            .checked_add(bytes.len())
            .ok_or(DurableKvError::ResourceLimit("encoded size"))?;
        if length > self.limit {
            return Err(DurableKvError::ResourceLimit("encoded size"));
        }
        if let Some(output) = &mut self.bytes {
            output.extend_from_slice(bytes);
        }
        self.length = length;
        Ok(())
    }
    fn u8(&mut self, value: u8) -> Result<(), DurableKvError> {
        self.raw(&[value])
    }
    fn u16(&mut self, value: u16) -> Result<(), DurableKvError> {
        self.raw(&value.to_le_bytes())
    }
    fn u32(&mut self, value: u32) -> Result<(), DurableKvError> {
        self.raw(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<(), DurableKvError> {
        self.raw(&value.to_le_bytes())
    }
    fn count(&mut self, count: usize, limit: usize) -> Result<(), DurableKvError> {
        if count > limit {
            return Err(DurableKvError::ResourceLimit("count"));
        }
        self.u32(u32::try_from(count).map_err(|_| DurableKvError::ResourceLimit("count"))?)
    }
    fn blob(&mut self, bytes: &[u8], limit: usize) -> Result<(), DurableKvError> {
        self.count(bytes.len(), limit)?;
        self.raw(bytes)
    }
    fn text(&mut self, text: &str, limit: usize) -> Result<(), DurableKvError> {
        self.blob(text.as_bytes(), limit)
    }
    fn words(&mut self, words: &[u64]) -> Result<(), DurableKvError> {
        self.count(words.len(), MAX_ITEMS)?;
        for word in words {
            self.u64(*word)?;
        }
        Ok(())
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.position..]
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], DurableKvError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(DurableKvError::InvalidSnapshot("length overflow"))?;
        let result = self
            .bytes
            .get(self.position..end)
            .ok_or(DurableKvError::InvalidSnapshot("truncated fields"))?;
        self.position = end;
        Ok(result)
    }
    fn u8(&mut self) -> Result<u8, DurableKvError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, DurableKvError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| DurableKvError::InvalidSnapshot("u16"))?,
        ))
    }
    fn u32(&mut self) -> Result<u32, DurableKvError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| DurableKvError::InvalidSnapshot("u32"))?,
        ))
    }
    fn u64(&mut self) -> Result<u64, DurableKvError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| DurableKvError::InvalidSnapshot("u64"))?,
        ))
    }
    fn usize(&mut self) -> Result<usize, DurableKvError> {
        usize::try_from(self.u64()?).map_err(|_| DurableKvError::ResourceLimit("host length"))
    }
    fn count(&mut self, limit: usize, min_bytes: usize) -> Result<usize, DurableKvError> {
        let count = self.u32()? as usize;
        if count > limit
            || count
                .checked_mul(min_bytes)
                .is_none_or(|size| size > self.remaining().len())
        {
            return Err(DurableKvError::InvalidSnapshot(
                "count or remaining byte bound",
            ));
        }
        Ok(count)
    }
    fn blob(&mut self, limit: usize) -> Result<&'a [u8], DurableKvError> {
        let count = self.count(limit, 1)?;
        self.take(count)
    }
    fn text(&mut self, limit: usize) -> Result<&'a str, DurableKvError> {
        std::str::from_utf8(self.blob(limit)?)
            .map_err(|_| DurableKvError::InvalidSnapshot("UTF8 identity"))
    }
    fn words(&mut self) -> Result<Vec<u64>, DurableKvError> {
        let count = self.count(MAX_ITEMS, 8)?;
        let mut words = reserve(count)?;
        for _ in 0..count {
            words.push(self.u64()?);
        }
        Ok(words)
    }
}
fn reserve<T>(count: usize) -> Result<Vec<T>, DurableKvError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| DurableKvError::ResourceLimit("decoded allocation"))?;
    Ok(values)
}

struct PublishError {
    message: String,
    renamed: bool,
}
#[cfg(target_os = "linux")]
struct Host {
    directory: std::fs::File,
    lock: std::fs::File,
    new_lock: bool,
}
#[cfg(not(target_os = "linux"))]
struct Host {
    new_lock: bool,
}

#[cfg(target_os = "linux")]
impl Host {
    fn acquire(directory: &Path) -> Result<Self, DurableKvError> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        let directory = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(directory)
            .map_err(|error| unresolved("open controlled directory", error))?;
        check_private(&directory, true)?;
        let (lock, new_lock) = match open_relative(
            &directory,
            b"store.lock\0",
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
        ) {
            Ok(file) => (file, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (
                open_relative(&directory, b"store.lock\0", libc::O_RDWR)
                    .map_err(|error| unresolved("open existing lock", error))?,
                false,
            ),
            Err(error) => return Err(unresolved("create retained lock", error)),
        };
        check_private(&lock, false)?;
        // SAFETY: lock owns a valid descriptor and the operation is a Linux
        // nonblocking advisory lock. File ownership closes/unlocks on drop.
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            return if error.kind() == std::io::ErrorKind::WouldBlock {
                Err(DurableKvError::Busy)
            } else {
                Err(unresolved("exclusive lock", error))
            };
        }
        let host = Self {
            directory,
            lock,
            new_lock,
        };
        host.verify_lock()?;
        Ok(host)
    }
    fn verify_lock(&self) -> Result<(), DurableKvError> {
        use std::os::unix::fs::MetadataExt;
        let current = open_relative(&self.directory, b"store.lock\0", libc::O_RDONLY)
            .map_err(|error| unresolved("verify lock identity", error))?;
        check_private(&current, false)?;
        let expected = self
            .lock
            .metadata()
            .map_err(|error| unresolved("inspect retained lock", error))?;
        let actual = current
            .metadata()
            .map_err(|error| unresolved("inspect named lock", error))?;
        if expected.dev() != actual.dev() || expected.ino() != actual.ino() {
            return Err(DurableKvError::Unresolved {
                message: "retained lock inode was replaced".into(),
            });
        }
        Ok(())
    }
    fn load(&self, limit: usize) -> Result<Option<Vec<u8>>, DurableKvError> {
        use std::io::Read;
        self.verify_lock()?;
        let file = match open_relative(&self.directory, b"store.bin\0", libc::O_RDONLY) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(unresolved("open managed snapshot", error)),
        };
        check_private(&file, false)?;
        let size = file
            .metadata()
            .map_err(|error| unresolved("measure snapshot", error))?
            .len();
        if size > limit as u64 || size > MAX_DURABLE_KV_BYTES as u64 {
            return Err(DurableKvError::ResourceLimit("snapshot bytes"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size as usize)
            .map_err(|_| DurableKvError::ResourceLimit("snapshot allocation"))?;
        file.take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| unresolved("read snapshot", error))?;
        if bytes.len() > limit {
            return Err(DurableKvError::ResourceLimit("snapshot bytes"));
        }
        Ok(Some(bytes))
    }
    fn temp_exists(&self) -> Result<bool, DurableKvError> {
        match open_relative(&self.directory, b"store.next\0", libc::O_RDONLY) {
            Ok(file) => {
                check_private(&file, false)?;
                Ok(true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(unresolved("inspect orphan staging file", error)),
        }
    }
    fn clean_temp(&self) -> Result<(), DurableKvError> {
        use std::os::fd::AsRawFd;
        if self.temp_exists()? {
            // SAFETY: fixed NUL-terminated leaf relative to retained directory;
            // unlinkat removes that entry and never follows a target symlink.
            if unsafe { libc::unlinkat(self.directory.as_raw_fd(), c"store.next".as_ptr(), 0) } != 0
            {
                return Err(unresolved(
                    "remove orphan staging file",
                    std::io::Error::last_os_error(),
                ));
            }
            self.sync_directory()?;
        }
        Ok(())
    }
    fn sync_directory(&self) -> Result<(), DurableKvError> {
        self.directory
            .sync_all()
            .map_err(|error| unresolved("sync controlled directory", error))
    }
    fn publish(
        &self,
        bytes: &[u8],
        applying: bool,
        observer: &mut impl FnMut(DurableCommitPhase),
    ) -> Result<(), PublishError> {
        use std::io::Write;
        use std::os::fd::AsRawFd;
        self.verify_lock()
            .and_then(|_| self.clean_temp())
            .map_err(|error| PublishError {
                message: error.to_string(),
                renamed: true,
            })?;
        let mut file = open_relative(
            &self.directory,
            b"store.next\0",
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )
        .map_err(|error| PublishError {
            message: format!("create exclusive staging file: {error}"),
            renamed: false,
        })?;
        check_private(&file, false).map_err(|error| PublishError {
            message: error.to_string(),
            renamed: true,
        })?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| PublishError {
                message: format!("write/sync staging image: {error}"),
                renamed: false,
            })?;
        observer(if applying {
            DurableCommitPhase::ApplyTempSynced
        } else {
            DurableCommitPhase::RecordTempSynced
        });
        // SAFETY: both fixed leaf names are relative to the retained directory;
        // source is an exclusive, synced regular file on the same filesystem.
        if unsafe {
            libc::renameat(
                self.directory.as_raw_fd(),
                c"store.next".as_ptr(),
                self.directory.as_raw_fd(),
                c"store.bin".as_ptr(),
            )
        } != 0
        {
            return Err(PublishError {
                message: format!("publish atomic image: {}", std::io::Error::last_os_error()),
                renamed: false,
            });
        }
        observer(if applying {
            DurableCommitPhase::ApplyRenamed
        } else {
            DurableCommitPhase::RecordRenamed
        });
        self.sync_directory().map_err(|error| PublishError {
            message: error.to_string(),
            renamed: true,
        })?;
        observer(if applying {
            DurableCommitPhase::ApplyDurable
        } else {
            DurableCommitPhase::RecordDurable
        });
        Ok(())
    }
}
#[cfg(target_os = "linux")]
fn open_relative(
    directory: &std::fs::File,
    name: &[u8],
    flags: libc::c_int,
) -> std::io::Result<std::fs::File> {
    use std::os::fd::{AsRawFd, FromRawFd};
    // SAFETY: all callers supply fixed NUL-terminated leaf names; directory is
    // retained, mode is private, and successful fd ownership moves into File.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr().cast(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            0o600,
        )
    };
    if fd < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { std::fs::File::from_raw_fd(fd) })
    }
}
#[cfg(target_os = "linux")]
fn check_private(file: &std::fs::File, directory: bool) -> Result<(), DurableKvError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|error| unresolved("inspect private object", error))?;
    // SAFETY: geteuid has no arguments and no side effects.
    let uid = unsafe { libc::geteuid() };
    if metadata.uid() != uid
        || metadata.mode() & 0o077 != 0
        || if directory {
            !metadata.is_dir()
        } else {
            !metadata.is_file() || metadata.nlink() != 1
        }
    {
        return Err(DurableKvError::Unresolved {
            message: "profile requires private owned regular files and a private owned directory"
                .into(),
        });
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn unresolved(operation: &str, error: std::io::Error) -> DurableKvError {
    DurableKvError::Unresolved {
        message: format!("{operation}: {error}"),
    }
}
#[cfg(not(target_os = "linux"))]
impl Host {
    fn acquire(_directory: &Path) -> Result<Self, DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn verify_lock(&self) -> Result<(), DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn load(&self, _limit: usize) -> Result<Option<Vec<u8>>, DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn temp_exists(&self) -> Result<bool, DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn clean_temp(&self) -> Result<(), DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn sync_directory(&self) -> Result<(), DurableKvError> {
        Err(DurableKvError::UnsupportedPlatform)
    }
    fn publish(
        &self,
        _bytes: &[u8],
        _applying: bool,
        _observer: &mut impl FnMut(DurableCommitPhase),
    ) -> Result<(), PublishError> {
        Err(PublishError {
            message: "Linux profile unavailable".into(),
            renamed: false,
        })
    }
}
