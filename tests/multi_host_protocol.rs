//! Bounded research prototype, with no production adapter or external effects.
//!
//! Two synthetic managed-KV participants have atomic stable images containing
//! both a value and an exact token record. One coordinator has an atomic stable
//! intent/decision image. Crashes discard volatile votes/acknowledgments, not
//! stable records. Fixed membership, one coordinator authority and trusted
//! messages are assumptions. Delivery can be omitted, duplicated or partitioned.
//! A prepared participant cannot decide by timeout. Only eventual node recovery
//! and delivery are used in the finite healing demonstrations below.
//!
//! The protocol may expose one applied participant while another is prepared.
//! Completion evidence is distinct from simultaneous visibility across hosts.
//! There are no native irreversible commands, physical storage claims, leader
//! elections, authentication claims or universal termination/exactly-once claim.
//!
//! Primary protocol/assumption reference: Jim Gray and Leslie Lamport,
//! "Consensus on Transaction Commit", sections 1-3 and appendix A.2:
//! https://arxiv.org/pdf/cs/0408036 . Two-phase commit can block while its single
//! coordinator is unavailable; stable state and non-forged messages matter.

use std::collections::{HashSet, VecDeque};

const HOSTS: usize = 2;
const MAX_DEPTH: usize = 16;
const MAX_STATES: usize = 100_000;

/// The only supported command is a synthetic logical-key replacement. The
/// complete two-host payload, including order/membership, is part of identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Identity {
    token: u128,
    values: [u8; HOSTS],
}

impl Identity {
    fn validate(self) -> Result<Self, ProtocolError> {
        if self.token == 0 || self.values.iter().any(|value| *value > 3) {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        Ok(self)
    }
}

fn identity() -> Identity {
    Identity {
        token: 17,
        values: [1, 2],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolError {
    IdentityConflict,
    OutsideFiniteProfile,
    NotPrepared,
    TerminalDecision,
    MissingVotes,
    InvalidAcknowledgment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Phase {
    Prepared,
    Applied,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Record {
    identity: Identity,
    phase: Phase,
}

/// One model step replaces this complete stable image atomically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Participant {
    record: Option<Record>,
    value: u8,
    application_count: u8,
}

impl Participant {
    fn check_identity(&self, query: Identity) -> Result<(), ProtocolError> {
        query.validate()?;
        if self.record.is_some_and(|record| record.identity != query) {
            return Err(ProtocolError::IdentityConflict);
        }
        Ok(())
    }

    fn prepare(&mut self, query: Identity) -> Result<(), ProtocolError> {
        self.check_identity(query)?;
        if self.record.is_none() {
            self.record = Some(Record {
                identity: query,
                phase: Phase::Prepared,
            });
        }
        Ok(())
    }

    fn vote(&self, query: Identity, host: usize) -> Result<Vote, ProtocolError> {
        self.check_identity(query)?;
        if host >= HOSTS {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        let record = self.record.ok_or(ProtocolError::NotPrepared)?;
        Ok(Vote {
            identity: record.identity,
            host,
            prepared: record.phase != Phase::Aborted,
        })
    }

    /// Decision messages come from the sole trusted coordinator in the model.
    /// No local timeout or unrelated native callback can enter this operation.
    fn decide(&mut self, decision: DecisionRecord, host: usize) -> Result<(), ProtocolError> {
        self.check_identity(decision.identity)?;
        if host >= HOSTS {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        let phase = self.record.map(|record| record.phase);
        match decision.decision {
            Decision::Undecided => Err(ProtocolError::TerminalDecision),
            Decision::Commit => match phase {
                Some(Phase::Applied) => Ok(()),
                Some(Phase::Prepared) => {
                    // The value, application marker and retained identity are
                    // one atomic participant publication, not three writes.
                    *self = Self {
                        record: Some(Record {
                            identity: decision.identity,
                            phase: Phase::Applied,
                        }),
                        value: decision.identity.values[host],
                        application_count: 1,
                    };
                    Ok(())
                }
                Some(Phase::Aborted) => Err(ProtocolError::TerminalDecision),
                None => Err(ProtocolError::NotPrepared),
            },
            Decision::Abort => match phase {
                Some(Phase::Applied) => Err(ProtocolError::TerminalDecision),
                Some(Phase::Aborted) => Ok(()),
                Some(Phase::Prepared) | None => {
                    self.record = Some(Record {
                        identity: decision.identity,
                        phase: Phase::Aborted,
                    });
                    Ok(())
                }
            },
        }
    }

    fn acknowledgment(&self, query: Identity, host: usize) -> Result<Ack, ProtocolError> {
        self.check_identity(query)?;
        if host >= HOSTS {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        let record = self.record.ok_or(ProtocolError::InvalidAcknowledgment)?;
        if record.phase == Phase::Prepared {
            return Err(ProtocolError::InvalidAcknowledgment);
        }
        Ok(Ack {
            identity: record.identity,
            host,
            phase: record.phase,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Decision {
    Undecided,
    Commit,
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DecisionRecord {
    identity: Identity,
    decision: Decision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Vote {
    identity: Identity,
    host: usize,
    prepared: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Ack {
    identity: Identity,
    host: usize,
    phase: Phase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coordinator {
    stable: DecisionRecord,
    votes: [Option<Vote>; HOSTS],
    acknowledgments: [Option<Ack>; HOSTS],
}

impl Coordinator {
    fn new(identity: Identity) -> Result<Self, ProtocolError> {
        Ok(Self {
            stable: DecisionRecord {
                identity: identity.validate()?,
                decision: Decision::Undecided,
            },
            votes: [None; HOSTS],
            acknowledgments: [None; HOSTS],
        })
    }

    fn start(&self, query: Identity) -> Result<(), ProtocolError> {
        query.validate()?;
        if self.stable.identity != query {
            return Err(ProtocolError::IdentityConflict);
        }
        Ok(())
    }

    fn receive_vote(&mut self, vote: Vote) -> Result<(), ProtocolError> {
        self.start(vote.identity)?;
        if vote.host >= HOSTS {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        if self.stable.decision == Decision::Undecided {
            self.votes[vote.host] = Some(vote);
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<(), ProtocolError> {
        if self.stable.decision == Decision::Abort {
            return Err(ProtocolError::TerminalDecision);
        }
        if self.stable.decision == Decision::Commit {
            return Ok(());
        }
        if (0..HOSTS).any(|host| {
            self.votes[host]
                != Some(Vote {
                    identity: self.stable.identity,
                    host,
                    prepared: true,
                })
        }) {
            return Err(ProtocolError::MissingVotes);
        }
        // Publish the durable decision before any commit delivery.
        self.stable.decision = Decision::Commit;
        Ok(())
    }

    fn abort(&mut self) -> Result<(), ProtocolError> {
        if self.stable.decision == Decision::Commit {
            return Err(ProtocolError::TerminalDecision);
        }
        self.stable.decision = Decision::Abort;
        Ok(())
    }

    fn receive_acknowledgment(&mut self, ack: Ack) -> Result<(), ProtocolError> {
        self.start(ack.identity)?;
        if ack.host >= HOSTS {
            return Err(ProtocolError::OutsideFiniteProfile);
        }
        let expected = match self.stable.decision {
            Decision::Commit => Phase::Applied,
            Decision::Abort => Phase::Aborted,
            Decision::Undecided => return Err(ProtocolError::InvalidAcknowledgment),
        };
        if ack.phase != expected {
            return Err(ProtocolError::InvalidAcknowledgment);
        }
        self.acknowledgments[ack.host] = Some(ack);
        Ok(())
    }

    fn crash_reload(&mut self) {
        self.votes = [None; HOSTS];
        self.acknowledgments = [None; HOSTS];
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Knowledge {
    KnownApplied,
    KnownNotApplied,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct World {
    coordinator: Coordinator,
    participants: [Participant; HOSTS],
    coordinator_online: bool,
    participant_online: [bool; HOSTS],
    links: [bool; HOSTS],
}

impl World {
    fn new() -> Self {
        Self {
            coordinator: Coordinator::new(identity()).unwrap(),
            participants: [Participant::default(); HOSTS],
            coordinator_online: true,
            participant_online: [true; HOSTS],
            links: [true; HOSTS],
        }
    }

    fn can_deliver(&self, host: usize) -> bool {
        self.coordinator_online && self.participant_online[host] && self.links[host]
    }

    /// A cached acknowledgment never substitutes for exact recovered identity.
    /// This prototype conservatively rechecks both reachable stable images.
    fn query(&self, query: Identity) -> Result<Knowledge, ProtocolError> {
        self.coordinator.start(query)?;
        if !self.coordinator_online || (0..HOSTS).any(|host| !self.can_deliver(host)) {
            return Ok(Knowledge::Unresolved);
        }
        let expected = match self.coordinator.stable.decision {
            Decision::Commit => Phase::Applied,
            Decision::Abort => Phase::Aborted,
            Decision::Undecided => return Ok(Knowledge::Unresolved),
        };
        for host in 0..HOSTS {
            if self.coordinator.acknowledgments[host]
                != Some(Ack {
                    identity: query,
                    host,
                    phase: expected,
                })
                || self.participants[host].record
                    != Some(Record {
                        identity: query,
                        phase: expected,
                    })
            {
                return Ok(Knowledge::Unresolved);
            }
            let (value, count) = if expected == Phase::Applied {
                (query.values[host], 1)
            } else {
                (0, 0)
            };
            if self.participants[host].value != value
                || self.participants[host].application_count != count
            {
                return Ok(Knowledge::Unresolved);
            }
        }
        Ok(if expected == Phase::Applied {
            Knowledge::KnownApplied
        } else {
            Knowledge::KnownNotApplied
        })
    }

    fn step(&mut self, action: Action) {
        let query = self.coordinator.stable.identity;
        match action {
            Action::Prepare(host) if self.can_deliver(host) => {
                let _ = self.participants[host].prepare(query);
            }
            Action::Vote(host) if self.can_deliver(host) => {
                if let Ok(vote) = self.participants[host].vote(query, host) {
                    self.coordinator.receive_vote(vote).unwrap();
                }
            }
            Action::Commit if self.coordinator_online => {
                let _ = self.coordinator.commit();
            }
            Action::Abort if self.coordinator_online => {
                let _ = self.coordinator.abort();
            }
            Action::DeliverDecision(host) if self.can_deliver(host) => {
                let _ = self.participants[host].decide(self.coordinator.stable, host);
            }
            Action::Ack(host) if self.can_deliver(host) => {
                if let Ok(ack) = self.participants[host].acknowledgment(query, host) {
                    let _ = self.coordinator.receive_acknowledgment(ack);
                }
            }
            Action::LoseVote(host) => self.coordinator.votes[host] = None,
            Action::LoseAck(host) => self.coordinator.acknowledgments[host] = None,
            Action::CrashCoordinator => {
                self.coordinator_online = false;
                self.coordinator.crash_reload();
            }
            Action::ReloadCoordinator => {
                if !self.coordinator_online {
                    self.coordinator.crash_reload();
                    self.coordinator_online = true;
                }
            }
            Action::CrashParticipant(host) => self.participant_online[host] = false,
            Action::ReloadParticipant(host) => self.participant_online[host] = true,
            Action::Partition(host) => self.links[host] = false,
            Action::Heal(host) => self.links[host] = true,
            // A prepared participant has no safe timeout decision here.
            Action::ParticipantTimeout(host) => {
                assert!(host < HOSTS);
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Action {
    Prepare(usize),
    Vote(usize),
    Commit,
    Abort,
    DeliverDecision(usize),
    Ack(usize),
    LoseVote(usize),
    LoseAck(usize),
    CrashCoordinator,
    ReloadCoordinator,
    CrashParticipant(usize),
    ReloadParticipant(usize),
    Partition(usize),
    Heal(usize),
    ParticipantTimeout(usize),
}

fn actions() -> Vec<Action> {
    let mut actions = vec![
        Action::Commit,
        Action::Abort,
        Action::CrashCoordinator,
        Action::ReloadCoordinator,
    ];
    for host in 0..HOSTS {
        actions.extend([
            Action::Prepare(host),
            Action::Vote(host),
            Action::DeliverDecision(host),
            Action::Ack(host),
            Action::LoseVote(host),
            Action::LoseAck(host),
            Action::CrashParticipant(host),
            Action::ReloadParticipant(host),
            Action::Partition(host),
            Action::Heal(host),
            Action::ParticipantTimeout(host),
        ]);
    }
    actions
}

fn prepare_both(world: &mut World) {
    for host in 0..HOSTS {
        world.step(Action::Prepare(host));
        world.step(Action::Vote(host));
    }
}

/// Direct predicates over stable records, independent of protocol query logic.
fn check_safety(world: &World) {
    let query = identity();
    assert_eq!(world.coordinator.stable.identity, query);
    let mut applied = 0;
    let mut aborted = 0;
    for host in 0..HOSTS {
        let participant = world.participants[host];
        if let Some(record) = participant.record {
            assert_eq!(record.identity, query);
            match record.phase {
                Phase::Applied => {
                    applied += 1;
                    assert_eq!(world.coordinator.stable.decision, Decision::Commit);
                    assert_eq!(participant.value, query.values[host]);
                    assert_eq!(participant.application_count, 1);
                }
                Phase::Aborted => {
                    aborted += 1;
                    assert_eq!(world.coordinator.stable.decision, Decision::Abort);
                    assert_eq!(participant.value, 0);
                    assert_eq!(participant.application_count, 0);
                }
                Phase::Prepared => {
                    assert_eq!(participant.value, 0);
                    assert_eq!(participant.application_count, 0);
                }
            }
        } else {
            assert_eq!(participant.value, 0);
            assert_eq!(participant.application_count, 0);
        }
    }
    assert!(applied == 0 || aborted == 0);
    if world.coordinator.stable.decision == Decision::Commit {
        assert!(world.participants.iter().all(|participant| {
            participant
                .record
                .is_some_and(|record| record.phase != Phase::Aborted)
        }));
    }
    match world.query(query).unwrap() {
        Knowledge::KnownApplied => assert_eq!(applied, HOSTS),
        Knowledge::KnownNotApplied => assert_eq!(aborted, HOSTS),
        Knowledge::Unresolved => {}
    }
}

fn check_stability(before: &World, after: &World) {
    if before.coordinator.stable.decision != Decision::Undecided {
        assert_eq!(before.coordinator.stable, after.coordinator.stable);
    }
    for host in 0..HOSTS {
        let old = before.participants[host];
        if old
            .record
            .is_some_and(|record| record.phase != Phase::Prepared)
        {
            assert_eq!(old, after.participants[host]);
        }
    }
}

/// Fixed explicit healing schedule. It assumes all three nodes and links have
/// recovered, then delivers each required message; it proves no fairness bound.
fn healed(mut world: World) -> World {
    world.coordinator_online = true;
    world.participant_online = [true; HOSTS];
    world.links = [true; HOSTS];
    world.coordinator.crash_reload();
    if world.coordinator.stable.decision == Decision::Undecided {
        prepare_both(&mut world);
        world.step(Action::Commit);
    }
    for host in 0..HOSTS {
        world.step(Action::DeliverDecision(host));
        world.step(Action::Ack(host));
    }
    world
}

#[test]
fn bounded_all_schedules_preserve_safety_and_explicit_healing_completes() {
    let initial = World::new();
    let mut visited = HashSet::from([initial]);
    let mut queue = VecDeque::from([(initial, 0usize)]);
    let actions = actions();
    let mut transitions = 0;
    let mut depth_cutoffs = 0;
    let mut knowledge = [0usize; 3];
    let mut partial_visibility = 0;
    while let Some((world, depth)) = queue.pop_front() {
        check_safety(&world);
        knowledge[match world.query(identity()).unwrap() {
            Knowledge::KnownApplied => 0,
            Knowledge::KnownNotApplied => 1,
            Knowledge::Unresolved => 2,
        }] += 1;
        if world
            .participants
            .map(|participant| {
                participant
                    .record
                    .is_some_and(|record| record.phase == Phase::Applied)
            })
            .into_iter()
            .filter(|applied| *applied)
            .count()
            == 1
        {
            partial_visibility += 1;
        }
        let resolved = healed(world);
        check_safety(&resolved);
        check_stability(&world, &resolved);
        assert_eq!(
            resolved.query(identity()).unwrap(),
            if world.coordinator.stable.decision == Decision::Abort {
                Knowledge::KnownNotApplied
            } else {
                Knowledge::KnownApplied
            }
        );
        if depth == MAX_DEPTH {
            depth_cutoffs += 1;
            continue;
        }
        for action in &actions {
            let mut next = world;
            next.step(*action);
            transitions += 1;
            check_stability(&world, &next);
            if next != world && visited.insert(next) {
                assert!(visited.len() <= MAX_STATES, "finite state cap exhausted");
                queue.push_back((next, depth + 1));
            }
        }
    }
    assert!(
        visited.len() > 1000,
        "campaign did not exercise the fault states"
    );
    assert!(knowledge.iter().all(|count| *count != 0));
    assert!(partial_visibility != 0);
    println!(
        "two-host finite campaign: {} states, {transitions} attempted transitions, \
         depth {MAX_DEPTH}, {depth_cutoffs} depth cutoffs, state cap {MAX_STATES}; \
         knowledge {knowledge:?} (Applied/not-applied/unresolved), \
         {partial_visibility} partial-visibility states",
        visited.len()
    );
}

#[test]
fn partial_visibility_and_lost_acknowledgment_remain_unresolved() {
    let mut world = World::new();
    prepare_both(&mut world);
    world.step(Action::Commit);
    world.step(Action::DeliverDecision(0));
    world.step(Action::Ack(0));
    assert_eq!(
        [world.participants[0].value, world.participants[1].value],
        [1, 0]
    );
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    assert_eq!(
        world.coordinator.abort(),
        Err(ProtocolError::TerminalDecision)
    );
    world.step(Action::CrashCoordinator);
    world.step(Action::Partition(1));
    world.step(Action::ReloadCoordinator);
    for _ in 0..20 {
        world.step(Action::DeliverDecision(0));
        world.step(Action::Ack(0));
        world.step(Action::ParticipantTimeout(1));
    }
    assert_eq!(world.participants[0].application_count, 1);
    assert_eq!(world.participants[1].record.unwrap().phase, Phase::Prepared);
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world.step(Action::Heal(1));
    world.step(Action::DeliverDecision(1));
    assert_eq!(
        [world.participants[0].value, world.participants[1].value],
        [1, 2]
    );
    // Both values are applied, but the second exact acknowledgment is missing.
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world.step(Action::Ack(1));
    assert_eq!(world.query(identity()).unwrap(), Knowledge::KnownApplied);
    world.step(Action::LoseAck(0));
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world.step(Action::Ack(0));
    assert_eq!(world.query(identity()).unwrap(), Knowledge::KnownApplied);
    check_safety(&world);
}

#[test]
fn prepared_nodes_block_during_coordinator_loss_and_abort_is_terminal() {
    let mut world = World::new();
    prepare_both(&mut world);
    world.step(Action::CrashCoordinator);
    for _ in 0..20 {
        for host in 0..HOSTS {
            world.step(Action::ParticipantTimeout(host));
            world.step(Action::DeliverDecision(host));
        }
    }
    assert!(world.participants.iter().all(|participant| {
        participant.record.unwrap().phase == Phase::Prepared && participant.value == 0
    }));
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world.step(Action::ReloadCoordinator);
    world.step(Action::Abort);
    for host in 0..HOSTS {
        world.step(Action::DeliverDecision(host));
        world.step(Action::Ack(host));
    }
    assert_eq!(world.query(identity()).unwrap(), Knowledge::KnownNotApplied);
    assert_eq!(
        world.coordinator.commit(),
        Err(ProtocolError::TerminalDecision)
    );
    let stable = world;
    for host in 0..HOSTS {
        world.step(Action::Prepare(host));
        world.step(Action::Vote(host));
        world.step(Action::Commit);
        world.step(Action::DeliverDecision(host));
    }
    assert_eq!(world, stable);
    check_safety(&world);
}

#[test]
fn crash_reload_retains_exact_identity_and_never_reapplies_managed_values() {
    let mut world = healed(World::new());
    world.step(Action::CrashCoordinator);
    world.step(Action::CrashParticipant(0));
    world.step(Action::CrashParticipant(1));
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world.step(Action::ReloadCoordinator);
    for host in 0..HOSTS {
        world.step(Action::ReloadParticipant(host));
        for _ in 0..20 {
            world.step(Action::DeliverDecision(host));
            world.step(Action::Ack(host));
        }
    }
    assert_eq!(world.query(identity()).unwrap(), Knowledge::KnownApplied);
    assert_eq!(
        world
            .participants
            .map(|participant| participant.application_count),
        [1, 1]
    );
    check_safety(&world);
}

#[test]
fn conflicting_tokens_payloads_and_stale_acknowledgments_cannot_upgrade_knowledge() {
    let mut world = World::new();
    prepare_both(&mut world);
    let baseline = world;
    for changed in [
        Identity {
            token: 18,
            ..identity()
        },
        Identity {
            values: [3, 2],
            ..identity()
        },
        Identity {
            values: [1, 3],
            ..identity()
        },
        Identity {
            values: [2, 1],
            ..identity()
        },
    ] {
        assert_eq!(
            world.coordinator.start(changed),
            Err(ProtocolError::IdentityConflict)
        );
        assert_eq!(world.query(changed), Err(ProtocolError::IdentityConflict));
        for host in 0..HOSTS {
            assert_eq!(
                world.participants[host].prepare(changed),
                Err(ProtocolError::IdentityConflict)
            );
            assert_eq!(
                world.participants[host].decide(
                    DecisionRecord {
                        identity: changed,
                        decision: Decision::Commit,
                    },
                    host
                ),
                Err(ProtocolError::IdentityConflict)
            );
            assert_eq!(
                world.coordinator.receive_vote(Vote {
                    identity: changed,
                    host,
                    prepared: true,
                }),
                Err(ProtocolError::IdentityConflict)
            );
            assert_eq!(
                world.coordinator.receive_acknowledgment(Ack {
                    identity: changed,
                    host,
                    phase: Phase::Applied,
                }),
                Err(ProtocolError::IdentityConflict)
            );
        }
        assert_eq!(world, baseline);
    }
    world.step(Action::Commit);
    // A stale or injected cached acknowledgment is rechecked against the exact
    // recovered participant record, even if its token/payload look plausible.
    for host in 0..HOSTS {
        world
            .coordinator
            .receive_acknowledgment(Ack {
                identity: identity(),
                host,
                phase: Phase::Applied,
            })
            .unwrap();
    }
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world = healed(world);
    assert_eq!(world.query(identity()).unwrap(), Knowledge::KnownApplied);
    let completed = world;
    world.participants[1]
        .record
        .as_mut()
        .unwrap()
        .identity
        .values[0] = 3;
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world = completed;
    world.participants[1].value = 0;
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    world = completed;
    world.coordinator.acknowledgments[1].as_mut().unwrap().host = 0;
    assert_eq!(world.query(identity()).unwrap(), Knowledge::Unresolved);
    assert_eq!(
        Identity {
            token: 0,
            ..identity()
        }
        .validate(),
        Err(ProtocolError::OutsideFiniteProfile)
    );
    assert_eq!(
        Identity {
            values: [4, 2],
            ..identity()
        }
        .validate(),
        Err(ProtocolError::OutsideFiniteProfile)
    );
}
