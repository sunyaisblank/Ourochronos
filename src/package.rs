//! Deterministic portable package container for linked bytecode.
//!
//! A package is deliberately small in scope: a UTF-8 program identity, exact
//! runtime and temporal-resolution contracts, and one already-linked
//! validated bytecode artifact. Native launchers and dependency bundles can
//! wrap these bytes, but they must not reinterpret the executable payload or
//! silently replace a declared global solve with an orbit search.

use crate::ast::OpCode;
use crate::bytecode::{BytecodeError, BytecodeProgram, Instruction, MAX_ARTIFACT_BYTES};
use crate::bytecode_verifier::verify_default as verify_bytecode;
use crate::bytecode_vm::{bytecode_vm_supports, BytecodeVm, BytecodeVmConfig, BytecodeVmStatus};
use crate::core::error::BoundsPolicy;
use crate::core::{PagedMemory, Value};
use std::error::Error;
use std::fmt;

const MAGIC: &[u8; 8] = b"OUROPK\0\0";
const FORMAT_VERSION: u16 = 3;
const FLAG_REPLAY_WITNESS: u16 = 1;
const KNOWN_FORMAT_FLAGS: u16 = FLAG_REPLAY_WITNESS;
/// Runtime ABI implemented by this crate's portable bytecode machine.
pub const CURRENT_RUNTIME_ABI: u16 = 1;
/// Resolution-policy contract implemented by this package format/runtime.
pub const CURRENT_RESOLUTION_POLICY_VERSION: u16 = 1;
/// Exact Ourochronos-to-Z3 global-point solver contract implemented here.
pub const CURRENT_Z3_SOLVER_CONTRACT_VERSION: u16 = 1;
/// Maximum UTF-8 byte length of a package name.
pub const MAX_PACKAGE_NAME_BYTES: usize = 255;
/// Maximum complete package size accepted before decoding.
pub const MAX_PACKAGE_BYTES: usize = MAX_ARTIFACT_BYTES + MAX_PACKAGE_WITNESS_CELLS * 16 + 4096;
/// Maximum dense temporal-memory width accepted from an untrusted package.
pub const MAX_PACKAGE_MEMORY_CELLS: u64 = 1_048_576;
/// Maximum per-epoch gas budget accepted from an untrusted package manifest.
pub const MAX_PACKAGE_INSTRUCTIONS: u64 = 10_000_000;
/// Maximum canonical nonzero cells in an embedded replay witness.
pub const MAX_PACKAGE_WITNESS_CELLS: usize = MAX_PACKAGE_MEMORY_CELLS as usize;

/// Temporal resolution contract selected by a portable package.
///
/// The policy is deliberately closed rather than stringly typed: a runtime
/// must either implement the exact policy/version or reject the package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageResolutionPolicy {
    /// Follow the deterministic zero-seed orbit at runtime.
    Orbit,
    /// Begin from and independently replay an embedded point witness.
    EmbeddedPointWitness,
    /// Invoke the manifest-declared global point solver at runtime.
    RuntimeGlobalPoint,
}

/// Bounded solver backend descriptors understood by the package ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageSolverDescriptor {
    /// Ourochronos' typed QF_ABV global-point encoding executed by Z3.
    Z3,
}

/// An exact runtime solver dependency, including its versioned contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageSolverDependency {
    /// Solver backend required by the package.
    pub descriptor: PackageSolverDescriptor,
    /// Exact Ourochronos/backend interface contract, not a version range.
    pub contract_version: u16,
}

/// Exact Z3 dependency implemented by the current runtime.
pub const CURRENT_Z3_SOLVER_DEPENDENCY: PackageSolverDependency = PackageSolverDependency {
    descriptor: PackageSolverDescriptor::Z3,
    contract_version: CURRENT_Z3_SOLVER_CONTRACT_VERSION,
};

impl PackageSolverDependency {
    /// The exact Z3 global-point contract implemented by this runtime.
    pub const Z3_GLOBAL_POINT: Self = CURRENT_Z3_SOLVER_DEPENDENCY;

    /// Whether this dependency is exactly implemented by the current runtime.
    pub const fn is_runtime_compatible(self) -> bool {
        matches!(self.descriptor, PackageSolverDescriptor::Z3)
            && self.contract_version == CURRENT_Z3_SOLVER_CONTRACT_VERSION
    }
}

/// Manifest carried by a portable Ourochronos package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageManifest {
    /// Human/tool-facing package identity. It is not a filesystem path.
    pub name: String,
    /// Minimum exact runtime ABI needed by this artifact.
    pub runtime_abi: u16,
    /// Temporal memory width required by the linked program.
    pub memory_cells: u64,
    /// Per-epoch language instruction budget required by the package.
    pub max_instructions: u64,
    /// Exact address-normalization semantics used to compile/replay it.
    pub memory_bounds: BoundsPolicy,
    /// Version of the closed resolution-policy contract.
    pub resolution_policy_version: u16,
    /// Whether execution follows an orbit, replays a witness, or solves.
    pub resolution_policy: PackageResolutionPolicy,
    /// Exact solver contract required by `RuntimeGlobalPoint`, if any.
    pub solver_dependency: Option<PackageSolverDependency>,
}

impl PackageManifest {
    /// Create a manifest for the current runtime ABI.
    pub fn current(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            runtime_abi: CURRENT_RUNTIME_ABI,
            memory_cells: crate::core::MEMORY_SIZE as u64,
            max_instructions: 10_000_000,
            memory_bounds: BoundsPolicy::Wrap,
            resolution_policy_version: CURRENT_RESOLUTION_POLICY_VERSION,
            resolution_policy: PackageResolutionPolicy::Orbit,
            solver_dependency: None,
        }
    }

    /// Create a manifest for the current ABI and an exact temporal schema.
    pub fn with_memory(name: impl Into<String>, memory_cells: usize) -> Self {
        Self {
            name: name.into(),
            runtime_abi: CURRENT_RUNTIME_ABI,
            memory_cells: memory_cells as u64,
            max_instructions: 10_000_000,
            memory_bounds: BoundsPolicy::Wrap,
            resolution_policy_version: CURRENT_RESOLUTION_POLICY_VERSION,
            resolution_policy: PackageResolutionPolicy::Orbit,
            solver_dependency: None,
        }
    }

    /// Create a manifest with exact runtime resource/bounds semantics.
    pub fn with_runtime(
        name: impl Into<String>,
        memory_cells: usize,
        max_instructions: u64,
        memory_bounds: BoundsPolicy,
    ) -> Self {
        Self {
            name: name.into(),
            runtime_abi: CURRENT_RUNTIME_ABI,
            memory_cells: memory_cells as u64,
            max_instructions,
            memory_bounds,
            resolution_policy_version: CURRENT_RESOLUTION_POLICY_VERSION,
            resolution_policy: PackageResolutionPolicy::Orbit,
            solver_dependency: None,
        }
    }

    /// Declare that this package must carry a replay-verified point witness.
    pub fn embedded_point_witness(mut self) -> Self {
        self.resolution_policy = PackageResolutionPolicy::EmbeddedPointWitness;
        self.solver_dependency = None;
        self
    }

    /// Declare an exact runtime global-point solver dependency.
    pub fn runtime_global_point(mut self, dependency: PackageSolverDependency) -> Self {
        self.resolution_policy = PackageResolutionPolicy::RuntimeGlobalPoint;
        self.solver_dependency = Some(dependency);
        self
    }
}

/// Replay-verified point state embedded explicitly by a build action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageWitness {
    /// Canonical strictly increasing nonzero temporal cells.
    pub state: Vec<(u64, u64)>,
    /// Recomputable evidence digest binding the package manifest, executable,
    /// canonical point state, and replay instruction count. The historical
    /// serialized field name is retained, but arbitrary solver metadata is
    /// never trusted as package evidence.
    pub constraint_digest: u64,
    /// Independent bytecode replay instruction count.
    pub replay_instructions: u64,
}

impl PackageWitness {
    /// Construct package-bound witness evidence from an independently replayed
    /// state. An orbit manifest is normalized to the embedded-point policy in
    /// the same way as [`PortablePackage::with_replay_witness`].
    pub fn replay_bound(
        manifest: &PackageManifest,
        program: &BytecodeProgram,
        state: Vec<(u64, u64)>,
        replay_instructions: u64,
    ) -> Result<Self, PackageError> {
        let normalized = match manifest.resolution_policy {
            PackageResolutionPolicy::Orbit => manifest.clone().embedded_point_witness(),
            _ => manifest.clone(),
        };
        let constraint_digest =
            package_witness_digest(&normalized, program, &state, replay_instructions)?;
        Ok(Self {
            state,
            constraint_digest,
            replay_instructions,
        })
    }
}

/// A decoded portable package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortablePackage {
    /// Package metadata.
    pub manifest: PackageManifest,
    /// Sole executable authority.
    pub program: BytecodeProgram,
    /// Optional build-time solver witness, independently replayed on decode.
    pub witness: Option<PackageWitness>,
}

/// Package construction or decoding failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageError {
    /// Complete input exceeded the allocation limit.
    PackageTooLarge(usize),
    /// Name is empty, too long, or contains a NUL byte.
    InvalidName(String),
    /// Declared dense temporal memory is zero or exceeds the package limit.
    InvalidMemoryCells(u64),
    /// Runtime instruction budget is zero.
    InvalidInstructionBudget,
    /// Packaged bytecode failed independent exact-artifact verification.
    BytecodeVerification(String),
    /// Resolution-policy contract version is unsupported.
    UnsupportedResolutionPolicyVersion(u16),
    /// Encoded resolution-policy tag is unknown.
    UnsupportedResolutionPolicy(u8),
    /// The policy requires a replay witness, but none was supplied.
    MissingWitness,
    /// A replay witness was supplied to a policy that forbids one.
    UnexpectedWitness(PackageResolutionPolicy),
    /// The runtime-global policy omitted its exact solver dependency.
    MissingSolverDependency,
    /// A non-solver policy declared a solver dependency.
    UnexpectedSolverDependency(PackageResolutionPolicy),
    /// The declared solver contract is not implemented by this runtime.
    UnsupportedSolverDependency(PackageSolverDependency),
    /// Encoded solver descriptor tag is unknown.
    UnsupportedSolverDescriptor(u8),
    /// Solver descriptor and version fields are not a canonical option.
    MalformedSolverDependency,
    /// Embedded point state is noncanonical or fails independent replay.
    InvalidWitness(String),
    /// Package magic is absent or corrupt.
    BadMagic,
    /// Container version is unsupported.
    UnsupportedVersion(u16),
    /// Reserved flags are unsupported.
    UnsupportedFlags(u16),
    /// Artifact requires a different runtime ABI.
    UnsupportedRuntimeAbi(u16),
    /// Input ended during a field.
    Truncated {
        /// Byte offset where the field began.
        offset: usize,
        /// Required field width.
        needed: usize,
    },
    /// Declared name was not valid UTF-8.
    InvalidUtf8,
    /// Bytes remained after the declared package.
    TrailingBytes(usize),
    /// Embedded bytecode was malformed.
    InvalidBytecode(BytecodeError),
    /// The current package runtime ABI cannot execute a primitive.
    UnsupportedPrimitive(OpCode),
    /// The current package runtime ABI has no linked foreign host table.
    UnsupportedForeignCall,
    /// A length could not fit the format.
    LengthOverflow(&'static str),
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PackageTooLarge(size) => {
                write!(formatter, "package size {size} exceeds {MAX_PACKAGE_BYTES}")
            }
            Self::InvalidName(name) => write!(formatter, "invalid package name {name:?}"),
            Self::InvalidMemoryCells(cells) => write!(
                formatter,
                "package memory width {cells} is outside 1..={MAX_PACKAGE_MEMORY_CELLS}"
            ),
            Self::InvalidInstructionBudget => {
                write!(
                    formatter,
                    "package instruction budget must be within 1..={MAX_PACKAGE_INSTRUCTIONS}"
                )
            }
            Self::BytecodeVerification(message) => {
                write!(
                    formatter,
                    "packaged bytecode verification failed: {message}"
                )
            }
            Self::UnsupportedResolutionPolicyVersion(version) => write!(
                formatter,
                "unsupported package resolution-policy version {version}"
            ),
            Self::UnsupportedResolutionPolicy(policy) => {
                write!(
                    formatter,
                    "unsupported package resolution policy tag {policy}"
                )
            }
            Self::MissingWitness => {
                formatter.write_str("embedded-point package policy requires a replay witness")
            }
            Self::UnexpectedWitness(policy) => {
                write!(
                    formatter,
                    "package policy {policy:?} forbids a replay witness"
                )
            }
            Self::MissingSolverDependency => formatter
                .write_str("runtime-global package policy requires an exact solver dependency"),
            Self::UnexpectedSolverDependency(policy) => write!(
                formatter,
                "package policy {policy:?} forbids a solver dependency"
            ),
            Self::UnsupportedSolverDependency(dependency) => write!(
                formatter,
                "unsupported package solver dependency {dependency:?}"
            ),
            Self::UnsupportedSolverDescriptor(descriptor) => {
                write!(
                    formatter,
                    "unsupported package solver descriptor tag {descriptor}"
                )
            }
            Self::MalformedSolverDependency => formatter
                .write_str("package solver descriptor and contract version fields disagree"),
            Self::InvalidWitness(message) => write!(formatter, "invalid replay witness: {message}"),
            Self::BadMagic => write!(formatter, "invalid package magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported package version {version}")
            }
            Self::UnsupportedFlags(flags) => {
                write!(formatter, "unsupported package flags 0x{flags:04x}")
            }
            Self::UnsupportedRuntimeAbi(abi) => {
                write!(formatter, "unsupported package runtime ABI {abi}")
            }
            Self::Truncated { offset, needed } => {
                write!(
                    formatter,
                    "truncated package at {offset}: need {needed} bytes"
                )
            }
            Self::InvalidUtf8 => write!(formatter, "package name is not valid UTF-8"),
            Self::TrailingBytes(count) => write!(formatter, "{count} trailing package bytes"),
            Self::InvalidBytecode(error) => write!(formatter, "invalid packaged bytecode: {error}"),
            Self::UnsupportedPrimitive(opcode) => write!(
                formatter,
                "package runtime ABI does not support primitive {}",
                opcode.name()
            ),
            Self::UnsupportedForeignCall => {
                formatter.write_str("package manifest does not declare foreign host dependencies")
            }
            Self::LengthOverflow(field) => write!(formatter, "{field} length does not fit format"),
        }
    }
}

impl Error for PackageError {}

impl From<BytecodeError> for PackageError {
    fn from(value: BytecodeError) -> Self {
        Self::InvalidBytecode(value)
    }
}

impl PortablePackage {
    /// Construct a package only from bytecode that passes structural
    /// validation. Stack/type verification remains a mandatory compiler gate.
    pub fn new(manifest: PackageManifest, program: BytecodeProgram) -> Result<Self, PackageError> {
        validate_package_parts(&manifest, &program, None)?;
        Ok(Self {
            manifest,
            program,
            witness: None,
        })
    }

    /// Construct a package carrying an explicit independently replayed point
    /// state. Package execution begins from this state but still executes the
    /// bytecode and checks convergence; it never trusts solver bytes alone.
    pub fn with_replay_witness(
        mut manifest: PackageManifest,
        program: BytecodeProgram,
        witness: PackageWitness,
    ) -> Result<Self, PackageError> {
        match manifest.resolution_policy {
            PackageResolutionPolicy::Orbit => {
                // Backwards-compatible construction is sound because the
                // resulting manifest explicitly records the embedded policy.
                manifest = manifest.embedded_point_witness();
            }
            PackageResolutionPolicy::EmbeddedPointWitness => {}
            policy @ PackageResolutionPolicy::RuntimeGlobalPoint => {
                return Err(PackageError::UnexpectedWitness(policy));
            }
        }
        validate_package_parts(&manifest, &program, Some(&witness))?;
        Ok(Self {
            manifest,
            program,
            witness: Some(witness),
        })
    }

    /// Construct a package that declares an exact runtime global solver.
    ///
    /// No solver-produced state is accepted here: the runtime must invoke the
    /// declared dependency and independently replay its result.
    pub fn with_runtime_global_point(
        mut manifest: PackageManifest,
        program: BytecodeProgram,
        dependency: PackageSolverDependency,
    ) -> Result<Self, PackageError> {
        if matches!(
            manifest.resolution_policy,
            PackageResolutionPolicy::EmbeddedPointWitness
        ) {
            return Err(PackageError::MissingWitness);
        }
        manifest = manifest.runtime_global_point(dependency);
        Self::new(manifest, program)
    }

    /// Encode deterministically using fixed-width little-endian fields.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PackageError> {
        validate_package_parts(&self.manifest, &self.program, self.witness.as_ref())?;
        let bytecode = self.program.to_bytes()?;
        let name = self.manifest.name.as_bytes();
        let name_len =
            u16::try_from(name.len()).map_err(|_| PackageError::LengthOverflow("package name"))?;
        let bytecode_len =
            u32::try_from(bytecode.len()).map_err(|_| PackageError::LengthOverflow("bytecode"))?;
        let witness_count = self
            .witness
            .as_ref()
            .map_or(0, |witness| witness.state.len());
        let witness_count = u32::try_from(witness_count)
            .map_err(|_| PackageError::LengthOverflow("replay witness"))?;
        let flags = if self.witness.is_some() {
            FLAG_REPLAY_WITNESS
        } else {
            0
        };
        let (constraint_digest, replay_instructions) =
            self.witness.as_ref().map_or((0, 0), |witness| {
                (witness.constraint_digest, witness.replay_instructions)
            });
        let policy = encode_resolution_policy(self.manifest.resolution_policy);
        let (solver, solver_contract_version) =
            self.manifest
                .solver_dependency
                .map_or((0, 0), |dependency| {
                    (
                        encode_solver_descriptor(dependency.descriptor),
                        dependency.contract_version,
                    )
                });

        let mut bytes =
            Vec::with_capacity(64 + name.len() + witness_count as usize * 16 + bytecode.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&self.manifest.runtime_abi.to_le_bytes());
        bytes.push(encode_bounds(self.manifest.memory_bounds));
        bytes.push(policy);
        bytes.push(solver);
        bytes.push(0);
        bytes.extend_from_slice(&self.manifest.resolution_policy_version.to_le_bytes());
        bytes.extend_from_slice(&solver_contract_version.to_le_bytes());
        bytes.extend_from_slice(&name_len.to_le_bytes());
        bytes.extend_from_slice(&bytecode_len.to_le_bytes());
        bytes.extend_from_slice(&witness_count.to_le_bytes());
        bytes.extend_from_slice(&self.manifest.memory_cells.to_le_bytes());
        bytes.extend_from_slice(&self.manifest.max_instructions.to_le_bytes());
        bytes.extend_from_slice(&constraint_digest.to_le_bytes());
        bytes.extend_from_slice(&replay_instructions.to_le_bytes());
        bytes.extend_from_slice(name);
        if let Some(witness) = &self.witness {
            for (address, value) in &witness.state {
                bytes.extend_from_slice(&address.to_le_bytes());
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&bytecode);
        if bytes.len() > MAX_PACKAGE_BYTES {
            return Err(PackageError::PackageTooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    /// Decode with limits before allocation and validate the embedded program.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PackageError> {
        if bytes.len() > MAX_PACKAGE_BYTES {
            return Err(PackageError::PackageTooLarge(bytes.len()));
        }
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(PackageError::BadMagic);
        }
        let version = reader.u16()?;
        if version != FORMAT_VERSION {
            return Err(PackageError::UnsupportedVersion(version));
        }
        let flags = reader.u16()?;
        if flags & !KNOWN_FORMAT_FLAGS != 0 {
            return Err(PackageError::UnsupportedFlags(flags));
        }
        let runtime_abi = reader.u16()?;
        if runtime_abi != CURRENT_RUNTIME_ABI {
            return Err(PackageError::UnsupportedRuntimeAbi(runtime_abi));
        }
        let memory_bounds = decode_bounds(reader.u8()?)?;
        let resolution_policy = decode_resolution_policy(reader.u8()?)?;
        let solver_descriptor = reader.u8()?;
        let reserved = reader.u8()?;
        if reserved != 0 {
            return Err(PackageError::UnsupportedFlags(
                flags | (u16::from(reserved) << 8),
            ));
        }
        let resolution_policy_version = reader.u16()?;
        let solver_contract_version = reader.u16()?;
        let solver_dependency = match (solver_descriptor, solver_contract_version) {
            (0, 0) => None,
            (0, _) => return Err(PackageError::MalformedSolverDependency),
            (descriptor, 0) => {
                decode_solver_descriptor(descriptor)?;
                return Err(PackageError::MalformedSolverDependency);
            }
            (descriptor, contract_version) => Some(PackageSolverDependency {
                descriptor: decode_solver_descriptor(descriptor)?,
                contract_version,
            }),
        };
        let name_len = reader.u16()? as usize;
        if name_len > MAX_PACKAGE_NAME_BYTES {
            return Err(PackageError::InvalidName(format!("{name_len} bytes")));
        }
        let bytecode_len = reader.u32()? as usize;
        if bytecode_len > MAX_ARTIFACT_BYTES {
            return Err(PackageError::InvalidBytecode(
                BytecodeError::LimitExceeded {
                    what: "artifact byte",
                    count: bytecode_len,
                    limit: MAX_ARTIFACT_BYTES,
                },
            ));
        }
        let witness_count = reader.u32()? as usize;
        if witness_count > MAX_PACKAGE_WITNESS_CELLS {
            return Err(PackageError::InvalidWitness(format!(
                "{witness_count} cells exceeds {MAX_PACKAGE_WITNESS_CELLS}"
            )));
        }
        let memory_cells = reader.u64()?;
        let max_instructions = reader.u64()?;
        let constraint_digest = reader.u64()?;
        let replay_instructions = reader.u64()?;
        let has_witness = flags & FLAG_REPLAY_WITNESS != 0;
        if !has_witness
            && (witness_count != 0 || constraint_digest != 0 || replay_instructions != 0)
        {
            return Err(PackageError::InvalidWitness(
                "witness flag/count/metadata disagree".to_string(),
            ));
        }
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|_| PackageError::InvalidUtf8)?
            .to_string();
        let mut state = Vec::with_capacity(witness_count);
        for _ in 0..witness_count {
            state.push((reader.u64()?, reader.u64()?));
        }
        let program = BytecodeProgram::from_bytes(reader.take(bytecode_len)?)?;
        if reader.remaining() != 0 {
            return Err(PackageError::TrailingBytes(reader.remaining()));
        }
        let manifest = PackageManifest {
            name,
            runtime_abi,
            memory_cells,
            max_instructions,
            memory_bounds,
            resolution_policy_version,
            resolution_policy,
            solver_dependency,
        };
        let witness = if has_witness {
            Some(PackageWitness {
                state,
                constraint_digest,
                replay_instructions,
            })
        } else {
            None
        };
        validate_package_parts(&manifest, &program, witness.as_ref())?;
        Ok(Self {
            manifest,
            program,
            witness,
        })
    }
}

fn validate_package_parts(
    manifest: &PackageManifest,
    program: &BytecodeProgram,
    witness: Option<&PackageWitness>,
) -> Result<(), PackageError> {
    validate_manifest(manifest)?;
    if manifest.runtime_abi != CURRENT_RUNTIME_ABI {
        return Err(PackageError::UnsupportedRuntimeAbi(manifest.runtime_abi));
    }
    validate_resolution_contract(manifest, witness)?;
    program.validate()?;
    validate_runtime_compatibility(program)?;
    verify_bytecode(program)
        .map_err(|error| PackageError::BytecodeVerification(error.to_string()))?;
    if let Some(witness) = witness {
        validate_witness(manifest, program, witness)?;
    }
    Ok(())
}

fn validate_manifest(manifest: &PackageManifest) -> Result<(), PackageError> {
    let name = manifest.name.as_bytes();
    if name.is_empty() || name.len() > MAX_PACKAGE_NAME_BYTES || name.contains(&0) {
        return Err(PackageError::InvalidName(manifest.name.clone()));
    }
    if manifest.memory_cells == 0 || manifest.memory_cells > MAX_PACKAGE_MEMORY_CELLS {
        return Err(PackageError::InvalidMemoryCells(manifest.memory_cells));
    }
    if manifest.max_instructions == 0 || manifest.max_instructions > MAX_PACKAGE_INSTRUCTIONS {
        return Err(PackageError::InvalidInstructionBudget);
    }
    if manifest.resolution_policy_version != CURRENT_RESOLUTION_POLICY_VERSION {
        return Err(PackageError::UnsupportedResolutionPolicyVersion(
            manifest.resolution_policy_version,
        ));
    }
    Ok(())
}

fn validate_resolution_contract(
    manifest: &PackageManifest,
    witness: Option<&PackageWitness>,
) -> Result<(), PackageError> {
    match manifest.resolution_policy {
        PackageResolutionPolicy::Orbit => {
            if manifest.solver_dependency.is_some() {
                return Err(PackageError::UnexpectedSolverDependency(
                    PackageResolutionPolicy::Orbit,
                ));
            }
            if witness.is_some() {
                return Err(PackageError::UnexpectedWitness(
                    PackageResolutionPolicy::Orbit,
                ));
            }
        }
        PackageResolutionPolicy::EmbeddedPointWitness => {
            if manifest.solver_dependency.is_some() {
                return Err(PackageError::UnexpectedSolverDependency(
                    PackageResolutionPolicy::EmbeddedPointWitness,
                ));
            }
            if witness.is_none() {
                return Err(PackageError::MissingWitness);
            }
        }
        PackageResolutionPolicy::RuntimeGlobalPoint => {
            if witness.is_some() {
                return Err(PackageError::UnexpectedWitness(
                    PackageResolutionPolicy::RuntimeGlobalPoint,
                ));
            }
            let dependency = manifest
                .solver_dependency
                .ok_or(PackageError::MissingSolverDependency)?;
            if !dependency.is_runtime_compatible() {
                return Err(PackageError::UnsupportedSolverDependency(dependency));
            }
        }
    }
    Ok(())
}

fn validate_witness(
    manifest: &PackageManifest,
    program: &BytecodeProgram,
    witness: &PackageWitness,
) -> Result<(), PackageError> {
    if witness.state.len() > MAX_PACKAGE_WITNESS_CELLS {
        return Err(PackageError::InvalidWitness("too many cells".to_string()));
    }
    if witness.replay_instructions == 0 || witness.replay_instructions > manifest.max_instructions {
        return Err(PackageError::InvalidWitness(
            "replay instruction count is outside package gas".to_string(),
        ));
    }
    let expected = package_witness_digest(
        manifest,
        program,
        &witness.state,
        witness.replay_instructions,
    )?;
    if witness.constraint_digest != expected {
        return Err(PackageError::InvalidWitness(format!(
            "evidence digest {:016x} does not match package-bound digest {expected:016x}",
            witness.constraint_digest
        )));
    }
    let mut previous = None;
    let mut memory = PagedMemory::with_size(manifest.memory_cells as usize)
        .map_err(|error| PackageError::InvalidWitness(error.to_string()))?;
    for (address, value) in &witness.state {
        if *address >= manifest.memory_cells {
            return Err(PackageError::InvalidWitness(format!(
                "address {address} is outside {} cells",
                manifest.memory_cells
            )));
        }
        if previous.is_some_and(|prior| prior >= *address) || *value == 0 {
            return Err(PackageError::InvalidWitness(
                "cells must be strictly increasing and nonzero".to_string(),
            ));
        }
        memory
            .write(*address, Value::new(*value))
            .map_err(|error| PackageError::InvalidWitness(error.to_string()))?;
        previous = Some(*address);
    }
    let execution = BytecodeVm::with_config(BytecodeVmConfig {
        // One instruction beyond the claimed count is sufficient to prove a
        // mismatch while bounding malicious/nonterminating witness replay by
        // the witness's own already-validated evidence field.
        max_instructions: witness
            .replay_instructions
            .saturating_add(1)
            .min(manifest.max_instructions),
        memory_bounds: manifest.memory_bounds,
        ..BytecodeVmConfig::default()
    })
    .run(program, &memory)
    .map_err(|error| PackageError::InvalidWitness(error.to_string()))?;
    if !matches!(
        execution.status,
        BytecodeVmStatus::Finished | BytecodeVmStatus::Halted
    ) || !execution.present.numeric_values_equal(&memory)
        || execution.instructions_executed != witness.replay_instructions
    {
        return Err(PackageError::InvalidWitness(
            "state did not independently replay to the same point state".to_string(),
        ));
    }
    Ok(())
}

fn package_witness_digest(
    manifest: &PackageManifest,
    program: &BytecodeProgram,
    state: &[(u64, u64)],
    replay_instructions: u64,
) -> Result<u64, PackageError> {
    let bytecode = program.to_bytes()?;
    let mut hash = PackageEvidenceHash::new();
    hash.bytes(b"ourochronos.package-witness/v1");
    hash.bytes(manifest.name.as_bytes());
    hash.u16(manifest.runtime_abi);
    hash.u64(manifest.memory_cells);
    hash.u64(manifest.max_instructions);
    hash.u8(encode_bounds(manifest.memory_bounds));
    hash.u16(manifest.resolution_policy_version);
    hash.u8(encode_resolution_policy(manifest.resolution_policy));
    match manifest.solver_dependency {
        None => hash.u8(0),
        Some(dependency) => {
            hash.u8(1);
            hash.u8(encode_solver_descriptor(dependency.descriptor));
            hash.u16(dependency.contract_version);
        }
    }
    hash.bytes(&bytecode);
    hash.u64(state.len() as u64);
    for (address, value) in state {
        hash.u64(*address);
        hash.u64(*value);
    }
    hash.u64(replay_instructions);
    Ok(hash.finish())
}

struct PackageEvidenceHash(u64);

impl PackageEvidenceHash {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
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

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

const fn encode_bounds(bounds: BoundsPolicy) -> u8 {
    match bounds {
        BoundsPolicy::Error => 0,
        BoundsPolicy::Wrap => 1,
        BoundsPolicy::Clamp => 2,
    }
}

fn decode_bounds(tag: u8) -> Result<BoundsPolicy, PackageError> {
    match tag {
        0 => Ok(BoundsPolicy::Error),
        1 => Ok(BoundsPolicy::Wrap),
        2 => Ok(BoundsPolicy::Clamp),
        _ => Err(PackageError::InvalidWitness(format!(
            "unknown bounds policy tag {tag}"
        ))),
    }
}

const fn encode_resolution_policy(policy: PackageResolutionPolicy) -> u8 {
    match policy {
        PackageResolutionPolicy::Orbit => 0,
        PackageResolutionPolicy::EmbeddedPointWitness => 1,
        PackageResolutionPolicy::RuntimeGlobalPoint => 2,
    }
}

fn decode_resolution_policy(tag: u8) -> Result<PackageResolutionPolicy, PackageError> {
    match tag {
        0 => Ok(PackageResolutionPolicy::Orbit),
        1 => Ok(PackageResolutionPolicy::EmbeddedPointWitness),
        2 => Ok(PackageResolutionPolicy::RuntimeGlobalPoint),
        _ => Err(PackageError::UnsupportedResolutionPolicy(tag)),
    }
}

const fn encode_solver_descriptor(descriptor: PackageSolverDescriptor) -> u8 {
    match descriptor {
        PackageSolverDescriptor::Z3 => 1,
    }
}

fn decode_solver_descriptor(tag: u8) -> Result<PackageSolverDescriptor, PackageError> {
    match tag {
        1 => Ok(PackageSolverDescriptor::Z3),
        _ => Err(PackageError::UnsupportedSolverDescriptor(tag)),
    }
}

fn validate_runtime_compatibility(program: &BytecodeProgram) -> Result<(), PackageError> {
    // The current portable manifest has no host-dependency section. Even an
    // uncalled descriptor would be an undeclared environment dependency.
    if !program.foreigns.is_empty() {
        return Err(PackageError::UnsupportedForeignCall);
    }
    for instruction in &program.instructions {
        match instruction {
            Instruction::Primitive(opcode)
                if !bytecode_vm_supports(*opcode)
                    || matches!(
                        opcode,
                        OpCode::Input
                            | OpCode::Clock
                            | OpCode::Random
                            | OpCode::FileOpen
                            | OpCode::FileRead
                            | OpCode::FileWrite
                            | OpCode::FileSeek
                            | OpCode::FileFlush
                            | OpCode::FileClose
                            | OpCode::FileExists
                            | OpCode::FileSize
                            | OpCode::TcpConnect
                            | OpCode::SocketSend
                            | OpCode::SocketRecv
                            | OpCode::SocketClose
                            | OpCode::ProcExec
                            | OpCode::Sleep
                    ) =>
            {
                return Err(PackageError::UnsupportedPrimitive(*opcode));
            }
            Instruction::CallForeign(_) => return Err(PackageError::UnsupportedForeignCall),
            _ => {}
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], PackageError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(PackageError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(PackageError::Truncated {
                offset: self.offset,
                needed: count,
            })?;
        self.offset = end;
        Ok(result)
    }

    fn u16(&mut self) -> Result<u16, PackageError> {
        let bytes: [u8; 2] = self.take(2)?.try_into().expect("exact reader width");
        Ok(u16::from_le_bytes(bytes))
    }

    fn u8(&mut self) -> Result<u8, PackageError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, PackageError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("exact reader width");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, PackageError> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("exact reader width");
        Ok(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Program, Stmt};
    use crate::core::Value;
    use crate::hir::HirProgram;
    use crate::parser::FFIDeclaration;
    use crate::runtime::ffi::{FFIEffect, FFISignature};

    fn package(name: &str) -> PortablePackage {
        let mut source = Program::new();
        source.body.push(Stmt::Push(Value::new(42)));
        let hir = HirProgram::resolve(&source).unwrap();
        let program = BytecodeProgram::compile(&hir).unwrap();
        PortablePackage::new(PackageManifest::current(name), program).unwrap()
    }

    #[test]
    fn deterministic_round_trip_retains_exact_linked_program() {
        let package = package("example");
        let first = package.to_bytes().unwrap();
        let second = package.to_bytes().unwrap();
        assert_eq!(first, second);
        let decoded = PortablePackage::from_bytes(&first).unwrap();
        assert_eq!(decoded, package);
        assert_eq!(decoded.to_bytes().unwrap(), first);
    }

    #[test]
    fn rejects_bad_manifest_version_abi_and_flags() {
        assert!(matches!(
            PortablePackage::new(PackageManifest::current(""), package("x").program),
            Err(PackageError::InvalidName(_))
        ));

        let good = package("example").to_bytes().unwrap();
        let mut version = good.clone();
        version[8..10].copy_from_slice(&4u16.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&version).unwrap_err(),
            PackageError::UnsupportedVersion(4)
        );
        let mut flags = good.clone();
        flags[10..12].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&flags).unwrap_err(),
            PackageError::UnsupportedFlags(2)
        );
        let mut abi = good;
        abi[12..14].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&abi).unwrap_err(),
            PackageError::UnsupportedRuntimeAbi(2)
        );
    }

    #[test]
    fn rejects_temporal_memory_outside_the_portable_schema() {
        let program = package("valid").program;
        for memory_cells in [0, MAX_PACKAGE_MEMORY_CELLS + 1] {
            let manifest = PackageManifest {
                memory_cells,
                ..PackageManifest::current("invalid-memory")
            };
            assert_eq!(
                PortablePackage::new(manifest, program.clone()).unwrap_err(),
                PackageError::InvalidMemoryCells(memory_cells)
            );
        }
    }

    #[test]
    fn rejects_unbounded_manifest_gas_before_any_replay() {
        let program = package("valid-gas").program;
        for max_instructions in [0, MAX_PACKAGE_INSTRUCTIONS + 1, u64::MAX] {
            let manifest = PackageManifest {
                max_instructions,
                ..PackageManifest::current("invalid-gas")
            };
            assert_eq!(
                PortablePackage::new(manifest, program.clone()).unwrap_err(),
                PackageError::InvalidInstructionBudget
            );
        }
    }

    #[test]
    fn exact_verifier_runs_before_packaged_bytecode_can_be_replayed() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::ONE),
            Stmt::If {
                then_branch: vec![Stmt::Push(Value::ONE)],
                else_branch: Some(Vec::new()),
            },
        ];
        let invalid = BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap();
        assert!(
            invalid.validate().is_ok(),
            "fixture must be structurally valid"
        );
        assert!(matches!(
            PortablePackage::new(PackageManifest::current("invalid-stack-join"), invalid),
            Err(PackageError::BytecodeVerification(_))
        ));
    }

    #[test]
    fn rejects_bytecode_the_current_package_runtime_cannot_execute() {
        for opcode in [OpCode::Input, OpCode::Clock, OpCode::Random] {
            let mut nondeterministic = package("nondeterministic").program;
            nondeterministic.instructions[0] = Instruction::Primitive(opcode);
            assert_eq!(
                PortablePackage::new(
                    PackageManifest::current("nondeterministic"),
                    nondeterministic
                )
                .unwrap_err(),
                PackageError::UnsupportedPrimitive(opcode)
            );
        }

        for opcode in [
            OpCode::FileOpen,
            OpCode::TcpConnect,
            OpCode::SocketSend,
            OpCode::SocketRecv,
            OpCode::SocketClose,
            OpCode::ProcExec,
            OpCode::Sleep,
        ] {
            let mut external = package("external").program;
            external.instructions[0] = Instruction::Primitive(opcode);
            assert_eq!(
                PortablePackage::new(PackageManifest::current("external"), external).unwrap_err(),
                PackageError::UnsupportedPrimitive(opcode)
            );
        }

        let mut source = Program::new();
        source.ffi_declarations.push(FFIDeclaration {
            signature: FFISignature::new("host", "process").effects(vec![FFIEffect::IO]),
            symbol_name: None,
        });
        let foreign = BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap();
        assert_eq!(
            PortablePackage::new(PackageManifest::current("foreign"), foreign).unwrap_err(),
            PackageError::UnsupportedForeignCall
        );
    }

    #[test]
    fn truncation_trailing_and_embedded_corruption_are_rejected() {
        let good = package("example").to_bytes().unwrap();
        for length in 0..good.len() {
            assert!(
                PortablePackage::from_bytes(&good[..length]).is_err(),
                "accepted truncation at {length}"
            );
        }
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(
            PortablePackage::from_bytes(&trailing).unwrap_err(),
            PackageError::TrailingBytes(1)
        );

        let mut corrupt = good;
        let name_len = u16::from_le_bytes([corrupt[22], corrupt[23]]) as usize;
        corrupt[64 + name_len] ^= 0xff;
        assert!(matches!(
            PortablePackage::from_bytes(&corrupt),
            Err(PackageError::InvalidBytecode(_))
        ));
    }

    #[test]
    fn replay_witness_round_trips_and_is_independently_checked() {
        let mut source = Program::new();
        source.body = vec![
            Stmt::Push(Value::new(42)),
            Stmt::Push(Value::new(0)),
            Stmt::Op(OpCode::Prophecy),
        ];
        let program = BytecodeProgram::compile(&HirProgram::resolve(&source).unwrap()).unwrap();
        let manifest = PackageManifest::with_memory("witness", 4);
        let memory = {
            let mut memory = PagedMemory::with_size(4).unwrap();
            memory.write(0, Value::new(42)).unwrap();
            memory
        };
        let execution = BytecodeVm::with_config(BytecodeVmConfig {
            memory_bounds: BoundsPolicy::Wrap,
            ..BytecodeVmConfig::default()
        })
        .run(&program, &memory)
        .unwrap();
        assert!(execution.present.numeric_values_equal(&memory));
        let witness = PackageWitness::replay_bound(
            &manifest,
            &program,
            vec![(0, 42)],
            execution.instructions_executed,
        )
        .unwrap();
        let package = PortablePackage::with_replay_witness(
            manifest.clone(),
            program.clone(),
            witness.clone(),
        )
        .unwrap();
        assert_eq!(
            package.manifest.resolution_policy,
            PackageResolutionPolicy::EmbeddedPointWitness
        );
        assert_eq!(package.manifest.solver_dependency, None);
        let bytes = package.to_bytes().unwrap();
        assert_eq!(PortablePackage::from_bytes(&bytes).unwrap(), package);

        let mut invalid = witness.clone();
        invalid.state[0].1 = 41;
        assert!(matches!(
            PortablePackage::with_replay_witness(manifest.clone(), program.clone(), invalid),
            Err(PackageError::InvalidWitness(_))
        ));

        let mut invalid_digest = witness;
        invalid_digest.constraint_digest ^= 1;
        assert!(matches!(
            PortablePackage::with_replay_witness(manifest, program, invalid_digest),
            Err(PackageError::InvalidWitness(message)) if message.contains("evidence digest")
        ));
    }

    #[test]
    fn resolution_policies_enforce_exact_witness_and_solver_contracts() {
        let program = package("policy-program").program;

        let missing_witness = PackageManifest::current("missing-witness").embedded_point_witness();
        assert_eq!(
            PortablePackage::new(missing_witness, program.clone()).unwrap_err(),
            PackageError::MissingWitness
        );

        let missing_solver = PackageManifest {
            resolution_policy: PackageResolutionPolicy::RuntimeGlobalPoint,
            ..PackageManifest::current("missing-solver")
        };
        assert_eq!(
            PortablePackage::new(missing_solver, program.clone()).unwrap_err(),
            PackageError::MissingSolverDependency
        );

        let incompatible = PackageSolverDependency {
            descriptor: PackageSolverDescriptor::Z3,
            contract_version: CURRENT_Z3_SOLVER_CONTRACT_VERSION + 1,
        };
        assert_eq!(
            PortablePackage::with_runtime_global_point(
                PackageManifest::current("incompatible-solver"),
                program.clone(),
                incompatible,
            )
            .unwrap_err(),
            PackageError::UnsupportedSolverDependency(incompatible)
        );

        let runtime = PortablePackage::with_runtime_global_point(
            PackageManifest::current("runtime-global"),
            program,
            PackageSolverDependency::Z3_GLOBAL_POINT,
        )
        .unwrap();
        assert_eq!(
            runtime.manifest.resolution_policy,
            PackageResolutionPolicy::RuntimeGlobalPoint
        );
        assert_eq!(
            runtime.manifest.solver_dependency,
            Some(PackageSolverDependency::Z3_GLOBAL_POINT)
        );
        assert!(runtime.witness.is_none());
        let bytes = runtime.to_bytes().unwrap();
        assert_eq!(PortablePackage::from_bytes(&bytes).unwrap(), runtime);

        let mut runtime_with_fake_witness = runtime;
        runtime_with_fake_witness.witness = Some(PackageWitness {
            state: Vec::new(),
            constraint_digest: 1,
            replay_instructions: 1,
        });
        assert_eq!(
            runtime_with_fake_witness.to_bytes().unwrap_err(),
            PackageError::UnexpectedWitness(PackageResolutionPolicy::RuntimeGlobalPoint)
        );

        let embedded_with_solver = PackageManifest {
            solver_dependency: Some(PackageSolverDependency::Z3_GLOBAL_POINT),
            ..PackageManifest::current("embedded-with-solver").embedded_point_witness()
        };
        assert_eq!(
            PortablePackage::with_replay_witness(
                embedded_with_solver,
                package("embedded-program").program,
                PackageWitness {
                    state: Vec::new(),
                    constraint_digest: 1,
                    replay_instructions: 1,
                },
            )
            .unwrap_err(),
            PackageError::UnexpectedSolverDependency(PackageResolutionPolicy::EmbeddedPointWitness)
        );
    }

    #[test]
    fn malformed_resolution_policy_encodings_are_rejected() {
        let orbit = package("malformed-policy").to_bytes().unwrap();

        let mut unknown_policy = orbit.clone();
        unknown_policy[15] = 0xff;
        assert_eq!(
            PortablePackage::from_bytes(&unknown_policy).unwrap_err(),
            PackageError::UnsupportedResolutionPolicy(0xff)
        );

        let mut unknown_policy_version = orbit.clone();
        unknown_policy_version[18..20].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&unknown_policy_version).unwrap_err(),
            PackageError::UnsupportedResolutionPolicyVersion(2)
        );

        let mut version_without_solver = orbit.clone();
        version_without_solver[20..22].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&version_without_solver).unwrap_err(),
            PackageError::MalformedSolverDependency
        );

        let mut solver_on_orbit = orbit.clone();
        solver_on_orbit[16] = 1;
        solver_on_orbit[20..22].copy_from_slice(&CURRENT_Z3_SOLVER_CONTRACT_VERSION.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&solver_on_orbit).unwrap_err(),
            PackageError::UnexpectedSolverDependency(PackageResolutionPolicy::Orbit)
        );

        let mut witness_flag_on_orbit = orbit;
        witness_flag_on_orbit[10..12].copy_from_slice(&FLAG_REPLAY_WITNESS.to_le_bytes());
        assert_eq!(
            PortablePackage::from_bytes(&witness_flag_on_orbit).unwrap_err(),
            PackageError::UnexpectedWitness(PackageResolutionPolicy::Orbit)
        );
    }
}
