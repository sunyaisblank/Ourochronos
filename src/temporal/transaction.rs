//! Transactional boundary for temporal candidate evaluation.
//!
//! Candidate epochs receive only frozen inputs and stage observations/effect
//! intents in memory. A selection is explicit, rollback discards every staged
//! candidate, and an external commit log records one selected batch under an
//! idempotency token exactly once. An explicitly authorized adapter may then
//! receive the selected effect batch at most once; candidate evaluation never
//! has access to that adapter.

use crate::core::{Address, OutputItem, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Deterministic staging bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionLimits {
    /// Maximum frozen input words.
    pub max_inputs: usize,
    /// Maximum output items per candidate.
    pub max_outputs: usize,
    /// Maximum conservative retained output bytes per candidate, including
    /// explicit provenance dependencies.
    pub max_output_bytes: usize,
    /// Maximum external effect intents per candidate.
    pub max_effects: usize,
    /// Maximum aggregate payload/path/argument bytes per candidate.
    pub max_effect_bytes: usize,
    /// Maximum aggregate bytes in one exact environment-observation transcript.
    pub max_observation_bytes: usize,
    /// Maximum candidates retained before explicit selection.
    pub max_candidates: usize,
}

impl Default for TransactionLimits {
    fn default() -> Self {
        Self {
            max_inputs: 1_000_000,
            max_outputs: 1_000_000,
            max_output_bytes: 64 * 1024 * 1024,
            max_effects: 100_000,
            max_effect_bytes: 64 * 1024 * 1024,
            max_observation_bytes: 64 * 1024 * 1024,
            max_candidates: 65_536,
        }
    }
}

/// Exact immutable host-file observation used by a candidate epoch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrozenFileSnapshot {
    /// Exact path spelling used by the program and capability checks.
    pub path: String,
    /// Complete contents, or `None` when the path was observed missing.
    pub contents: Option<Vec<u8>>,
}

/// Exact immutable byte stream used by virtual sockets for one endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrozenEndpointTape {
    /// Exact `host:port` endpoint identity.
    pub endpoint: String,
    /// Complete receive stream for one candidate epoch.
    pub recv_bytes: Vec<u8>,
}

/// Exact process observation used without launching during candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrozenProcessResult {
    /// Exact shell command spelling supplied by the program.
    pub command: String,
    /// Frozen combined stdout/stderr bytes.
    pub output: Vec<u8>,
    /// Frozen platform process exit code.
    pub exit_code: i32,
}

/// Complete exact environment transcript that participated in one candidate.
/// This data is equality authority; its stable digest is only an accelerator.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ObservationTranscript {
    /// Scalar INPUT words consumed in order.
    pub input: Vec<u64>,
    /// CLOCK words consumed in order.
    pub clock: Vec<u64>,
    /// RANDOM words consumed in order.
    pub random: Vec<u64>,
    /// File snapshots consumed in first-access order, including missing paths.
    pub files: Vec<FrozenFileSnapshot>,
    /// Endpoint tapes consumed in first-connect order.
    pub endpoints: Vec<FrozenEndpointTape>,
    /// Process results consumed in first-execution order.
    pub processes: Vec<FrozenProcessResult>,
}

impl ObservationTranscript {
    pub(crate) fn byte_len(&self) -> usize {
        let words = self
            .input
            .len()
            .saturating_add(self.clock.len())
            .saturating_add(self.random.len())
            .saturating_mul(std::mem::size_of::<u64>());
        let files = self.files.iter().fold(0usize, |total, snapshot| {
            total
                .saturating_add(snapshot.path.len())
                .saturating_add(snapshot.contents.as_ref().map_or(0, Vec::len))
                .saturating_add(1)
        });
        let endpoints = self.endpoints.iter().fold(0usize, |total, tape| {
            total
                .saturating_add(tape.endpoint.len())
                .saturating_add(tape.recv_bytes.len())
        });
        let processes = self.processes.iter().fold(0usize, |total, result| {
            total
                .saturating_add(result.command.len())
                .saturating_add(result.output.len())
                .saturating_add(std::mem::size_of::<i32>())
        });
        words
            .saturating_add(files)
            .saturating_add(endpoints)
            .saturating_add(processes)
    }

    fn item_count(&self) -> usize {
        self.input
            .len()
            .saturating_add(self.clock.len())
            .saturating_add(self.random.len())
            .saturating_add(self.files.len())
            .saturating_add(self.endpoints.len())
            .saturating_add(self.processes.len())
    }
}

/// An irreversible operation described as data but never executed during
/// candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectIntent {
    /// Replace/write bytes in a named file at an exact offset.
    FileWrite {
        /// Platform-neutral path spelling supplied by the program.
        path: String,
        /// Byte offset.
        offset: u64,
        /// Staged payload.
        bytes: Vec<u8>,
        /// Initial frozen path state against which the host commit is checked.
        initial: FrozenFileSnapshot,
    },
    /// Create a named file if necessary and set its exact byte length.
    /// Candidate evaluation uses this for CREATE/TRUNCATE without touching the
    /// host file system.
    FileSetLength {
        /// Platform-neutral path spelling supplied by the program.
        path: String,
        /// Selected length. Extending a file creates zero bytes.
        length: u64,
        /// Initial frozen path state against which the host commit is checked.
        initial: FrozenFileSnapshot,
    },
    /// Send a payload to an already-resolved endpoint identity.
    NetworkSend {
        /// Frozen endpoint identity, not a live socket handle.
        endpoint: String,
        /// Staged payload.
        bytes: Vec<u8>,
    },
    /// Request a process invocation after selection.
    ProcessSpawn {
        /// Program identity.
        program: String,
        /// Exact argument vector.
        arguments: Vec<String>,
    },
    /// Wait for an exact duration after selection.
    Sleep {
        /// Duration requested by the selected program.
        milliseconds: u64,
    },
    /// Application-defined effect for a separately registered commit adapter.
    Custom {
        /// Stable adapter namespace.
        namespace: String,
        /// Opaque staged payload.
        payload: Vec<u8>,
    },
}

impl EffectIntent {
    pub(crate) fn byte_len(&self) -> usize {
        match self {
            Self::FileWrite {
                path,
                bytes,
                initial,
                ..
            } => path
                .len()
                .saturating_add(bytes.len())
                .saturating_add(initial.path.len())
                .saturating_add(initial.contents.as_ref().map_or(0, Vec::len)),
            Self::FileSetLength { path, initial, .. } => path
                .len()
                .saturating_add(initial.path.len())
                .saturating_add(initial.contents.as_ref().map_or(0, Vec::len)),
            Self::NetworkSend { endpoint, bytes } => endpoint.len().saturating_add(bytes.len()),
            Self::ProcessSpawn { program, arguments } => {
                arguments.iter().fold(program.len(), |total, argument| {
                    total.saturating_add(argument.len())
                })
            }
            Self::Sleep { .. } => std::mem::size_of::<u64>(),
            Self::Custom { namespace, payload } => namespace.len().saturating_add(payload.len()),
        }
    }
}

/// Exact selected state and its staged observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineCandidate {
    /// Canonical sparse temporal state, strictly increasing by address.
    pub state: Vec<(Address, Value)>,
    /// Observations visible only if this candidate is selected.
    pub output: Vec<OutputItem>,
    /// Irreversible operations still represented only as data.
    pub effects: Vec<EffectIntent>,
    /// Prefix of the frozen input tape consumed by this epoch.
    pub inputs_consumed: Vec<u64>,
    /// Complete exact frozen environment transcript for this candidate.
    pub observations: ObservationTranscript,
}

/// Stable candidate identity. Hash collisions are detected and rejected by
/// exact candidate comparison; the digest is never equality authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimelineId(pub u64);

/// Caller-supplied durable idempotency identity for one selected commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommitToken(pub u128);

/// Evidence recorded when a selected batch enters the in-memory commit log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReceipt {
    /// Idempotency token.
    pub token: CommitToken,
    /// Exact selected candidate.
    pub timeline: TimelineId,
    /// Stable digest of the complete committed batch.
    pub batch_digest: u64,
    /// Zero-based append position. A replay retains the same position.
    pub sequence: usize,
}

/// One selected batch exposed to a later, explicitly authorized host adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedBatch {
    /// Receipt/idempotency evidence.
    pub receipt: CommitReceipt,
    /// Exact selected temporal state used to verify digest collisions.
    pub state: Vec<(Address, Value)>,
    /// Selected output.
    pub output: Vec<OutputItem>,
    /// Selected effect intents. Still not executed by this module.
    pub effects: Vec<EffectIntent>,
    /// Frozen input prefix consumed by the selected epoch.
    pub inputs_consumed: Vec<u64>,
    /// Complete exact frozen environment transcript for the selected epoch.
    pub observations: ObservationTranscript,
}

/// Whether a commit appended a batch or was an idempotent replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitOutcome {
    /// First use of the token appended this batch exactly once.
    Committed(CommitReceipt),
    /// The same token and exact batch were already recorded.
    AlreadyCommitted(CommitReceipt),
}

/// Explicit host boundary for a complete selected effect batch.
///
/// The adapter is called only after the exact batch has entered the commit
/// ledger. It should persist/use `receipt.token` as its own idempotency key if
/// it provides stronger crash recovery. A returned error is terminal for this
/// in-memory transaction: retrying the same token will not call the adapter a
/// second time, preserving the language's at-most-once effect invariant. This
/// boundary does not promise atomic rollback across heterogeneous host
/// resources; an adapter can apply a prefix before a runtime failure and must
/// then return the stored failure without reapplying that prefix.
pub trait EffectCommitAdapter {
    /// Apply one authorized, at-most-once batch of selected intents.
    fn apply_selected(
        &mut self,
        receipt: &CommitReceipt,
        effects: &[EffectIntent],
    ) -> Result<(), String>;
}

/// Exact-once in-memory ledger. A durable host can persist the same token,
/// digest, and sequence before an authorized adapter applies effects.
#[derive(Debug, Clone, Default)]
pub struct CommitLog {
    batches: Vec<CommittedBatch>,
    tokens: BTreeMap<CommitToken, usize>,
}

impl CommitLog {
    /// Recorded batches in first-commit order.
    pub fn batches(&self) -> &[CommittedBatch] {
        &self.batches
    }

    fn record(
        &mut self,
        token: CommitToken,
        timeline: TimelineId,
        candidate: &TimelineCandidate,
    ) -> Result<CommitOutcome, TransactionError> {
        let batch_digest = candidate_digest(candidate);
        if let Some(&sequence) = self.tokens.get(&token) {
            let existing = self.batches.get(sequence).ok_or_else(|| {
                TransactionError::Invariant("commit token references a missing batch".to_string())
            })?;
            if existing.receipt.timeline != timeline
                || existing.receipt.batch_digest != batch_digest
                || existing.state != candidate.state
                || existing.output != candidate.output
                || existing.effects != candidate.effects
                || existing.inputs_consumed != candidate.inputs_consumed
                || existing.observations != candidate.observations
            {
                return Err(TransactionError::CommitTokenConflict { token });
            }
            return Ok(CommitOutcome::AlreadyCommitted(existing.receipt.clone()));
        }

        let sequence = self.batches.len();
        let receipt = CommitReceipt {
            token,
            timeline,
            batch_digest,
            sequence,
        };
        self.batches.push(CommittedBatch {
            receipt: receipt.clone(),
            state: candidate.state.clone(),
            output: candidate.output.clone(),
            effects: candidate.effects.clone(),
            inputs_consumed: candidate.inputs_consumed.clone(),
            observations: candidate.observations.clone(),
        });
        self.tokens.insert(token, sequence);
        Ok(CommitOutcome::Committed(receipt))
    }
}

/// Candidate-local evaluation context. Reads are confined to the frozen tape;
/// effects and output can only be staged.
#[derive(Debug, Clone)]
pub struct CandidateContext {
    frozen_inputs: Vec<u64>,
    input_cursor: usize,
    output: Vec<OutputItem>,
    output_bytes: usize,
    effects: Vec<EffectIntent>,
    effect_bytes: usize,
    observations: ObservationTranscript,
    limits: TransactionLimits,
}

impl CandidateContext {
    /// Read the next frozen input word.
    pub fn read_input(&mut self) -> Result<u64, TransactionError> {
        let value = self.frozen_inputs.get(self.input_cursor).copied().ok_or(
            TransactionError::FrozenInputExhausted {
                consumed: self.input_cursor,
            },
        )?;
        self.input_cursor += 1;
        self.observations.input.push(value);
        Ok(value)
    }

    /// Attach the complete exact environment observations produced by the
    /// candidate VM. The scalar input prefix must match transaction reads.
    pub fn set_observation_transcript(
        &mut self,
        observations: ObservationTranscript,
    ) -> Result<(), TransactionError> {
        if observations.input != self.observations.input {
            return Err(TransactionError::Invariant(
                "observation INPUT transcript disagrees with transaction reads".to_string(),
            ));
        }
        validate_observations(&observations, self.limits)?;
        self.observations = observations;
        Ok(())
    }

    /// Stage one observation without exposing it externally.
    pub fn stage_output(&mut self, item: OutputItem) -> Result<(), TransactionError> {
        if self.output.len() >= self.limits.max_outputs {
            return Err(TransactionError::LimitExceeded {
                what: "output",
                limit: self.limits.max_outputs,
            });
        }
        let charge = item.retained_size_charge();
        let next_bytes =
            self.output_bytes
                .checked_add(charge)
                .ok_or(TransactionError::LimitExceeded {
                    what: "output byte",
                    limit: self.limits.max_output_bytes,
                })?;
        if next_bytes > self.limits.max_output_bytes {
            return Err(TransactionError::LimitExceeded {
                what: "output byte",
                limit: self.limits.max_output_bytes,
            });
        }
        self.output.push(item);
        self.output_bytes = next_bytes;
        Ok(())
    }

    /// Stage an irreversible intent without executing it.
    pub fn stage_effect(&mut self, effect: EffectIntent) -> Result<(), TransactionError> {
        if self.effects.len() >= self.limits.max_effects {
            return Err(TransactionError::LimitExceeded {
                what: "effect",
                limit: self.limits.max_effects,
            });
        }
        let bytes = self.effect_bytes.checked_add(effect.byte_len()).ok_or(
            TransactionError::LimitExceeded {
                what: "effect byte",
                limit: self.limits.max_effect_bytes,
            },
        )?;
        if bytes > self.limits.max_effect_bytes {
            return Err(TransactionError::LimitExceeded {
                what: "effect byte",
                limit: self.limits.max_effect_bytes,
            });
        }
        self.effect_bytes = bytes;
        self.effects.push(effect);
        Ok(())
    }

    /// Finish this candidate with a canonical sparse temporal state.
    pub fn finish(
        mut self,
        mut state: Vec<(Address, Value)>,
    ) -> Result<TimelineCandidate, TransactionError> {
        state.sort_by_key(|(address, _)| *address);
        if state.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(TransactionError::DuplicateStateAddress);
        }
        state.retain(|(_, value)| value.val != 0 || !value.prov.is_pure());
        self.output.shrink_to_fit();
        self.effects.shrink_to_fit();
        validate_observations(&self.observations, self.limits)?;
        let inputs_consumed = self.observations.input.clone();
        Ok(TimelineCandidate {
            state,
            output: self.output,
            effects: self.effects,
            inputs_consumed,
            observations: self.observations,
        })
    }
}

/// Transaction lifecycle failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionError {
    /// Configured resource bound is invalid or exhausted.
    LimitExceeded {
        /// Bounded entity.
        what: &'static str,
        /// Configured maximum.
        limit: usize,
    },
    /// Candidate consumed beyond the frozen input tape.
    FrozenInputExhausted {
        /// Words already consumed.
        consumed: usize,
    },
    /// Sparse state contained an address twice.
    DuplicateStateAddress,
    /// A digest collision was observed between unequal candidates.
    TimelineIdentityCollision(TimelineId),
    /// Requested candidate does not exist.
    UnknownTimeline(TimelineId),
    /// Commit was attempted without explicit selection.
    NoSelectedTimeline,
    /// Selection disagrees with an already selected candidate.
    SelectionConflict {
        /// Existing selection.
        selected: TimelineId,
        /// Requested selection.
        requested: TimelineId,
    },
    /// Transaction has been rolled back.
    RolledBack,
    /// Transaction already committed a different idempotency token.
    TransactionAlreadyCommitted(CommitToken),
    /// A commit token was reused for a different exact batch.
    CommitTokenConflict {
        /// Conflicting token.
        token: CommitToken,
    },
    /// The selected batch was durably ledgered but its authorized adapter
    /// failed. It will not be invoked again by this transaction/log pair.
    EffectAdapterFailed {
        /// Commit identity already present in the ledger.
        token: CommitToken,
        /// Adapter-provided failure detail.
        message: String,
    },
    /// Internal ledger invariant failure.
    Invariant(String),
}

impl fmt::Display for TransactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded { what, limit } => {
                write!(formatter, "{what} limit {limit} exceeded")
            }
            Self::FrozenInputExhausted { consumed } => {
                write!(formatter, "frozen input exhausted after {consumed} words")
            }
            Self::DuplicateStateAddress => write!(formatter, "candidate state repeats an address"),
            Self::TimelineIdentityCollision(id) => {
                write!(formatter, "timeline digest collision for {id:?}")
            }
            Self::UnknownTimeline(id) => write!(formatter, "unknown timeline {id:?}"),
            Self::NoSelectedTimeline => write!(formatter, "no timeline has been selected"),
            Self::SelectionConflict {
                selected,
                requested,
            } => write!(
                formatter,
                "timeline {selected:?} is selected; cannot select {requested:?}"
            ),
            Self::RolledBack => write!(formatter, "transaction was rolled back"),
            Self::TransactionAlreadyCommitted(token) => {
                write!(
                    formatter,
                    "transaction already committed with token {token:?}"
                )
            }
            Self::CommitTokenConflict { token } => {
                write!(formatter, "commit token {token:?} names a different batch")
            }
            Self::EffectAdapterFailed { token, message } => {
                write!(formatter, "effect adapter failed for {token:?}: {message}")
            }
            Self::Invariant(message) => write!(formatter, "transaction invariant: {message}"),
        }
    }
}

impl Error for TransactionError {}

/// Frozen-input and staged-effect transaction for one resolution attempt.
#[derive(Debug, Clone)]
pub struct TemporalTransaction {
    frozen_inputs: Vec<u64>,
    limits: TransactionLimits,
    candidates: BTreeMap<TimelineId, TimelineCandidate>,
    selected: Option<TimelineId>,
    committed: Option<CommitToken>,
    rolled_back: bool,
}

impl TemporalTransaction {
    /// Begin a resolution transaction with an exact frozen input tape.
    pub fn new(
        frozen_inputs: Vec<u64>,
        limits: TransactionLimits,
    ) -> Result<Self, TransactionError> {
        if frozen_inputs.len() > limits.max_inputs {
            return Err(TransactionError::LimitExceeded {
                what: "frozen input",
                limit: limits.max_inputs,
            });
        }
        if limits.max_candidates == 0 {
            return Err(TransactionError::LimitExceeded {
                what: "candidate",
                limit: 0,
            });
        }
        Ok(Self {
            frozen_inputs,
            limits,
            candidates: BTreeMap::new(),
            selected: None,
            committed: None,
            rolled_back: false,
        })
    }

    /// Start one isolated candidate evaluation at input position zero.
    pub fn begin_candidate(&self) -> Result<CandidateContext, TransactionError> {
        if self.rolled_back {
            return Err(TransactionError::RolledBack);
        }
        if let Some(token) = self.committed {
            return Err(TransactionError::TransactionAlreadyCommitted(token));
        }
        Ok(CandidateContext {
            frozen_inputs: self.frozen_inputs.clone(),
            input_cursor: 0,
            output: Vec::new(),
            output_bytes: 0,
            effects: Vec::new(),
            effect_bytes: 0,
            observations: ObservationTranscript::default(),
            limits: self.limits,
        })
    }

    /// Retain a finished candidate and return its stable identity.
    pub fn stage_candidate(
        &mut self,
        candidate: TimelineCandidate,
    ) -> Result<TimelineId, TransactionError> {
        validate_observations(&candidate.observations, self.limits)?;
        if candidate.inputs_consumed != candidate.observations.input {
            return Err(TransactionError::Invariant(
                "candidate INPUT prefix disagrees with observation transcript".to_string(),
            ));
        }
        let id = TimelineId(candidate_digest(&candidate));
        self.insert_candidate(id, candidate)?;
        Ok(id)
    }

    fn insert_candidate(
        &mut self,
        id: TimelineId,
        candidate: TimelineCandidate,
    ) -> Result<(), TransactionError> {
        if self.rolled_back {
            return Err(TransactionError::RolledBack);
        }
        if let Some(token) = self.committed {
            return Err(TransactionError::TransactionAlreadyCommitted(token));
        }
        if let Some(existing) = self.candidates.get(&id) {
            if existing == &candidate {
                return Ok(());
            }
            return Err(TransactionError::TimelineIdentityCollision(id));
        }
        if self.candidates.len() >= self.limits.max_candidates {
            return Err(TransactionError::LimitExceeded {
                what: "candidate",
                limit: self.limits.max_candidates,
            });
        }
        self.candidates.insert(id, candidate);
        Ok(())
    }

    /// All candidates remain observable until one is explicitly selected.
    pub fn candidates(&self) -> &BTreeMap<TimelineId, TimelineCandidate> {
        &self.candidates
    }

    /// Select exactly one candidate. Repeating the same selection is safe.
    pub fn select(&mut self, id: TimelineId) -> Result<(), TransactionError> {
        if self.rolled_back {
            return Err(TransactionError::RolledBack);
        }
        if let Some(token) = self.committed {
            return Err(TransactionError::TransactionAlreadyCommitted(token));
        }
        if !self.candidates.contains_key(&id) {
            return Err(TransactionError::UnknownTimeline(id));
        }
        match self.selected {
            None => self.selected = Some(id),
            Some(selected) if selected == id => {}
            Some(selected) => {
                return Err(TransactionError::SelectionConflict {
                    selected,
                    requested: id,
                });
            }
        }
        Ok(())
    }

    /// Record the selected observations/intents exactly once in `log`.
    /// This does not apply those intents to the host environment.
    pub fn commit_selected(
        &mut self,
        token: CommitToken,
        log: &mut CommitLog,
    ) -> Result<CommitOutcome, TransactionError> {
        if self.rolled_back {
            return Err(TransactionError::RolledBack);
        }
        if let Some(committed) = self.committed {
            if committed != token {
                return Err(TransactionError::TransactionAlreadyCommitted(committed));
            }
        }
        let selected = self.selected.ok_or(TransactionError::NoSelectedTimeline)?;
        let candidate = self
            .candidates
            .get(&selected)
            .ok_or(TransactionError::UnknownTimeline(selected))?;
        let outcome = log.record(token, selected, candidate)?;
        self.committed = Some(token);
        Ok(outcome)
    }

    /// Ledger and dispatch the selected effect batch at most once.
    ///
    /// Replaying an already committed exact token returns
    /// [`CommitOutcome::AlreadyCommitted`] without invoking `adapter`.
    pub fn commit_selected_with_adapter(
        &mut self,
        token: CommitToken,
        log: &mut CommitLog,
        adapter: &mut (impl EffectCommitAdapter + ?Sized),
    ) -> Result<CommitOutcome, TransactionError> {
        let outcome = self.commit_selected(token, log)?;
        let CommitOutcome::Committed(receipt) = &outcome else {
            return Ok(outcome);
        };
        let batch = log.batches().get(receipt.sequence).ok_or_else(|| {
            TransactionError::Invariant(
                "new commit receipt references a missing effect batch".to_string(),
            )
        })?;
        adapter
            .apply_selected(receipt, &batch.effects)
            .map_err(|message| TransactionError::EffectAdapterFailed { token, message })?;
        Ok(outcome)
    }

    /// Discard every candidate and make future selection/commit impossible.
    pub fn rollback(&mut self) -> Result<(), TransactionError> {
        if let Some(token) = self.committed {
            return Err(TransactionError::TransactionAlreadyCommitted(token));
        }
        self.candidates.clear();
        self.selected = None;
        self.rolled_back = true;
        Ok(())
    }
}

fn validate_observations(
    observations: &ObservationTranscript,
    limits: TransactionLimits,
) -> Result<(), TransactionError> {
    if observations.item_count() > limits.max_inputs {
        return Err(TransactionError::LimitExceeded {
            what: "environment observation",
            limit: limits.max_inputs,
        });
    }
    if observations.byte_len() > limits.max_observation_bytes {
        return Err(TransactionError::LimitExceeded {
            what: "environment observation byte",
            limit: limits.max_observation_bytes,
        });
    }
    Ok(())
}

fn candidate_digest(candidate: &TimelineCandidate) -> u64 {
    let mut hash = StableHash::new();
    hash.bytes(b"ourochronos.timeline/v2");
    hash.usize(candidate.state.len());
    for (address, value) in &candidate.state {
        hash.u64(*address);
        hash.value(value);
    }
    hash.usize(candidate.output.len());
    for output in &candidate.output {
        match output {
            OutputItem::Val(value) => {
                hash.u8(0);
                hash.value(value);
            }
            OutputItem::Char(character) => {
                hash.u8(1);
                hash.u8(*character);
            }
        }
    }
    hash.usize(candidate.effects.len());
    for effect in &candidate.effects {
        hash.effect(effect);
    }
    hash.usize(candidate.inputs_consumed.len());
    for value in &candidate.inputs_consumed {
        hash.u64(*value);
    }
    hash.observations(&candidate.observations);
    hash.finish()
}

struct StableHash(u64);

impl StableHash {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.raw(&(bytes.len() as u64).to_le_bytes());
        self.raw(bytes);
    }

    fn raw(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn value(&mut self, value: &Value) {
        self.u64(value.val);
        self.u8(u8::from(value.prov.saturated));
        match &value.prov.deps {
            None => self.u8(0),
            Some(dependencies) => {
                self.u8(1);
                self.usize(dependencies.len());
                for dependency in dependencies.iter() {
                    self.u64(*dependency);
                }
            }
        }
    }

    fn effect(&mut self, effect: &EffectIntent) {
        match effect {
            EffectIntent::FileWrite {
                path,
                offset,
                bytes,
                initial,
            } => {
                self.u8(0);
                self.bytes(path.as_bytes());
                self.u64(*offset);
                self.bytes(bytes);
                self.file_snapshot(initial);
            }
            EffectIntent::FileSetLength {
                path,
                length,
                initial,
            } => {
                self.u8(4);
                self.bytes(path.as_bytes());
                self.u64(*length);
                self.file_snapshot(initial);
            }
            EffectIntent::NetworkSend { endpoint, bytes } => {
                self.u8(1);
                self.bytes(endpoint.as_bytes());
                self.bytes(bytes);
            }
            EffectIntent::ProcessSpawn { program, arguments } => {
                self.u8(2);
                self.bytes(program.as_bytes());
                self.usize(arguments.len());
                for argument in arguments {
                    self.bytes(argument.as_bytes());
                }
            }
            EffectIntent::Sleep { milliseconds } => {
                self.u8(5);
                self.u64(*milliseconds);
            }
            EffectIntent::Custom { namespace, payload } => {
                self.u8(3);
                self.bytes(namespace.as_bytes());
                self.bytes(payload);
            }
        }
    }

    fn file_snapshot(&mut self, snapshot: &FrozenFileSnapshot) {
        self.bytes(snapshot.path.as_bytes());
        match &snapshot.contents {
            None => self.u8(0),
            Some(contents) => {
                self.u8(1);
                self.bytes(contents);
            }
        }
    }

    fn observations(&mut self, observations: &ObservationTranscript) {
        self.usize(observations.input.len());
        for value in &observations.input {
            self.u64(*value);
        }
        self.usize(observations.clock.len());
        for value in &observations.clock {
            self.u64(*value);
        }
        self.usize(observations.random.len());
        for value in &observations.random {
            self.u64(*value);
        }
        self.usize(observations.files.len());
        for snapshot in &observations.files {
            self.file_snapshot(snapshot);
        }
        self.usize(observations.endpoints.len());
        for tape in &observations.endpoints {
            self.bytes(tape.endpoint.as_bytes());
            self.bytes(&tape.recv_bytes);
        }
        self.usize(observations.processes.len());
        for result in &observations.processes {
            self.bytes(result.command.as_bytes());
            self.bytes(&result.output);
            self.u64(result.exit_code as u64);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingAdapter {
        calls: Vec<(CommitToken, Vec<EffectIntent>)>,
        fail: bool,
    }

    impl EffectCommitAdapter for RecordingAdapter {
        fn apply_selected(
            &mut self,
            receipt: &CommitReceipt,
            effects: &[EffectIntent],
        ) -> Result<(), String> {
            self.calls.push((receipt.token, effects.to_vec()));
            if self.fail {
                Err("injected adapter failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    fn candidate(transaction: &TemporalTransaction, value: u64) -> TimelineCandidate {
        let mut context = transaction.begin_candidate().unwrap();
        assert_eq!(context.read_input().unwrap(), 7);
        context
            .stage_output(OutputItem::Val(Value::new(value)))
            .unwrap();
        context
            .stage_effect(EffectIntent::Custom {
                namespace: "test".into(),
                payload: vec![value as u8],
            })
            .unwrap();
        context.finish(vec![(0, Value::new(value))]).unwrap()
    }

    #[test]
    fn candidate_epochs_only_read_frozen_inputs_and_stage_effects() {
        let transaction = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let first = candidate(&transaction, 1);
        let second = candidate(&transaction, 1);
        assert_eq!(first, second);
        assert_eq!(first.inputs_consumed, vec![7]);
        assert_eq!(first.effects.len(), 1);

        let mut exhausted = transaction.begin_candidate().unwrap();
        exhausted.read_input().unwrap();
        assert_eq!(
            exhausted.read_input().unwrap_err(),
            TransactionError::FrozenInputExhausted { consumed: 1 }
        );
    }

    #[test]
    fn multiple_candidates_remain_visible_until_explicit_selection() {
        let mut transaction =
            TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let first = transaction
            .stage_candidate(candidate(&transaction, 1))
            .unwrap();
        let second = transaction
            .stage_candidate(candidate(&transaction, 2))
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(transaction.candidates().len(), 2);
        transaction.select(second).unwrap();
        transaction.select(second).unwrap();
        assert!(matches!(
            transaction.select(first),
            Err(TransactionError::SelectionConflict { .. })
        ));
    }

    #[test]
    fn commit_tokens_are_exactly_once_and_cross_transaction_idempotent() {
        let mut transaction =
            TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let timeline = transaction
            .stage_candidate(candidate(&transaction, 9))
            .unwrap();
        transaction.select(timeline).unwrap();
        let token = CommitToken(42);
        let mut log = CommitLog::default();
        assert!(matches!(
            transaction.commit_selected(token, &mut log).unwrap(),
            CommitOutcome::Committed(_)
        ));
        assert!(matches!(
            transaction.commit_selected(token, &mut log).unwrap(),
            CommitOutcome::AlreadyCommitted(_)
        ));
        assert_eq!(log.batches().len(), 1);
        assert!(matches!(
            transaction.commit_selected(CommitToken(43), &mut log),
            Err(TransactionError::TransactionAlreadyCommitted(CommitToken(
                42
            )))
        ));

        let mut replay = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let replay_id = replay.stage_candidate(candidate(&replay, 9)).unwrap();
        replay.select(replay_id).unwrap();
        assert!(matches!(
            replay.commit_selected(token, &mut log).unwrap(),
            CommitOutcome::AlreadyCommitted(_)
        ));
        assert_eq!(log.batches().len(), 1);
    }

    #[test]
    fn authorized_effect_adapter_is_never_replayed() {
        let mut transaction =
            TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let timeline = transaction
            .stage_candidate(candidate(&transaction, 9))
            .unwrap();
        transaction.select(timeline).unwrap();
        let mut log = CommitLog::default();
        let mut adapter = RecordingAdapter::default();

        assert!(matches!(
            transaction
                .commit_selected_with_adapter(CommitToken(71), &mut log, &mut adapter)
                .unwrap(),
            CommitOutcome::Committed(_)
        ));
        assert_eq!(adapter.calls.len(), 1);
        assert!(matches!(
            transaction
                .commit_selected_with_adapter(CommitToken(71), &mut log, &mut adapter)
                .unwrap(),
            CommitOutcome::AlreadyCommitted(_)
        ));
        assert_eq!(
            adapter.calls.len(),
            1,
            "commit replay repeated host effects"
        );

        let mut failed = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let failed_timeline = failed.stage_candidate(candidate(&failed, 10)).unwrap();
        failed.select(failed_timeline).unwrap();
        let mut failing_adapter = RecordingAdapter {
            fail: true,
            ..RecordingAdapter::default()
        };
        assert!(matches!(
            failed.commit_selected_with_adapter(CommitToken(72), &mut log, &mut failing_adapter),
            Err(TransactionError::EffectAdapterFailed {
                token: CommitToken(72),
                ..
            })
        ));
        assert_eq!(failing_adapter.calls.len(), 1);
        assert!(matches!(
            failed
                .commit_selected_with_adapter(CommitToken(72), &mut log, &mut failing_adapter)
                .unwrap(),
            CommitOutcome::AlreadyCommitted(_)
        ));
        assert_eq!(
            failing_adapter.calls.len(),
            1,
            "failed dispatch was replayed"
        );
    }

    #[test]
    fn commit_token_conflict_and_digest_collision_never_change_selection() {
        let mut first = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let first_id = first.stage_candidate(candidate(&first, 1)).unwrap();
        first.select(first_id).unwrap();
        let mut log = CommitLog::default();
        first.commit_selected(CommitToken(5), &mut log).unwrap();

        let mut second = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let second_id = second.stage_candidate(candidate(&second, 2)).unwrap();
        second.select(second_id).unwrap();
        assert!(matches!(
            second.commit_selected(CommitToken(5), &mut log),
            Err(TransactionError::CommitTokenConflict { .. })
        ));
        assert_eq!(log.batches().len(), 1);

        let colliding_id = TimelineId(123);
        let first_candidate = candidate(&second, 3);
        let second_candidate = candidate(&second, 4);
        second
            .insert_candidate(colliding_id, first_candidate)
            .unwrap();
        assert_eq!(
            second
                .insert_candidate(colliding_id, second_candidate)
                .unwrap_err(),
            TransactionError::TimelineIdentityCollision(colliding_id)
        );
    }

    #[test]
    fn complete_observation_transcript_is_exact_batch_identity() {
        fn observed_candidate(transaction: &TemporalTransaction, random: u64) -> TimelineCandidate {
            let mut context = transaction.begin_candidate().unwrap();
            assert_eq!(context.read_input().unwrap(), 7);
            context
                .set_observation_transcript(ObservationTranscript {
                    input: vec![7],
                    clock: vec![100],
                    random: vec![random],
                    files: vec![FrozenFileSnapshot {
                        path: "missing".to_string(),
                        contents: None,
                    }],
                    endpoints: vec![FrozenEndpointTape {
                        endpoint: "host:1".to_string(),
                        recv_bytes: vec![1, 2],
                    }],
                    processes: vec![FrozenProcessResult {
                        command: "command".to_string(),
                        output: vec![3],
                        exit_code: 4,
                    }],
                })
                .unwrap();
            context.finish(vec![(0, Value::new(9))]).unwrap()
        }

        let mut first = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let first_candidate = observed_candidate(&first, 11);
        let first_id = first.stage_candidate(first_candidate.clone()).unwrap();
        first.select(first_id).unwrap();
        let mut log = CommitLog::default();
        first.commit_selected(CommitToken(88), &mut log).unwrap();
        assert_eq!(log.batches()[0].observations, first_candidate.observations);

        let mut changed = TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        let changed_candidate = observed_candidate(&changed, 12);
        let changed_id = changed.stage_candidate(changed_candidate.clone()).unwrap();
        assert_ne!(first_id, changed_id);
        changed.select(changed_id).unwrap();
        assert_eq!(
            changed
                .commit_selected(CommitToken(88), &mut log)
                .unwrap_err(),
            TransactionError::CommitTokenConflict {
                token: CommitToken(88)
            }
        );

        // Even an adversarially forced digest collision cannot replace exact
        // transcript equality authority.
        let forced = TimelineId(0xBAD);
        let mut collision =
            TemporalTransaction::new(vec![7], TransactionLimits::default()).unwrap();
        collision.insert_candidate(forced, first_candidate).unwrap();
        assert_eq!(
            collision
                .insert_candidate(forced, changed_candidate)
                .unwrap_err(),
            TransactionError::TimelineIdentityCollision(forced)
        );
    }

    #[test]
    fn observation_transcript_bytes_are_bounded() {
        let limits = TransactionLimits {
            max_observation_bytes: 4,
            ..TransactionLimits::default()
        };
        let transaction = TemporalTransaction::new(Vec::new(), limits).unwrap();
        let mut context = transaction.begin_candidate().unwrap();
        assert!(matches!(
            context.set_observation_transcript(ObservationTranscript {
                files: vec![FrozenFileSnapshot {
                    path: "long-path".to_string(),
                    contents: None,
                }],
                ..ObservationTranscript::default()
            }),
            Err(TransactionError::LimitExceeded {
                what: "environment observation byte",
                limit: 4
            })
        ));
    }

    #[test]
    fn candidate_output_bytes_are_bounded_separately_from_item_count() {
        let limits = TransactionLimits {
            max_outputs: usize::MAX,
            max_output_bytes: 63,
            ..TransactionLimits::default()
        };
        let transaction = TemporalTransaction::new(Vec::new(), limits).unwrap();
        let mut context = transaction.begin_candidate().unwrap();
        assert!(matches!(
            context.stage_output(OutputItem::Val(Value::new(1))),
            Err(TransactionError::LimitExceeded {
                what: "output byte",
                limit: 63
            })
        ));
    }

    #[test]
    fn rollback_and_resource_limits_are_hard_boundaries() {
        let limits = TransactionLimits {
            max_outputs: 1,
            max_effect_bytes: 3,
            ..TransactionLimits::default()
        };
        let mut transaction = TemporalTransaction::new(vec![7], limits).unwrap();
        let mut context = transaction.begin_candidate().unwrap();
        context.stage_output(OutputItem::Char(b'a')).unwrap();
        assert!(matches!(
            context.stage_output(OutputItem::Char(b'b')),
            Err(TransactionError::LimitExceeded { what: "output", .. })
        ));
        assert!(matches!(
            context.stage_effect(EffectIntent::Custom {
                namespace: "toolong".into(),
                payload: Vec::new()
            }),
            Err(TransactionError::LimitExceeded {
                what: "effect byte",
                ..
            })
        ));
        transaction.rollback().unwrap();
        assert_eq!(transaction.candidates().len(), 0);
        assert_eq!(
            transaction.begin_candidate().unwrap_err(),
            TransactionError::RolledBack
        );
    }
}
