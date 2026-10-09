//! Explicit safe embedding with locally approved finite scalar observations.
//! Run `cargo run --locked --example host_snapshot`. No native library is loaded.

use ourochronos::hir::{ForeignId, HirProgram};
use ourochronos::parser::parse;
use ourochronos::runtime::host_manifest::{FrozenScalarObservation, HostManifest};
use ourochronos::{BytecodeProgram, BytecodeVm, BytecodeVmConfig, PagedMemory};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = parse("FOREIGN \"embedding\" { PROC adjust ( unsigned: u64 signed: i64 -- result: i64 ) PURE; } INPUT 18446744073709551615 adjust")?;
    let hir = HirProgram::resolve(&source).map_err(|errors| format!("{errors:?}"))?;
    let program = BytecodeProgram::compile(&hir)?;
    // These locally selected values are trusted configuration in this example;
    // production embedders must establish snapshot provenance independently.
    let snapshot = HostManifest::new(
        &program,
        vec![7],
        vec![FrozenScalarObservation {
            foreign: ForeignId::try_from_index(0).ok_or("foreign identity does not fit")?,
            arguments: vec![7, u64::MAX],
            result: Some(6),
        }],
    )?;
    let approved = snapshot.digest()?; // Approval of locally trusted data.
    let bytes = snapshot.to_bytes()?;
    // A transported blob must match both separately retained expected identities.
    let checked = HostManifest::from_bytes(&bytes)?.check(&program, approved)?;
    let config = BytecodeVmConfig {
        input: checked.input().to_vec(),
        max_instructions: 64,
        allow_interactive_input: false,
        ..BytecodeVmConfig::default()
    };
    let vm =
        BytecodeVm::with_config(config).with_foreign_host(Arc::new(checked.frozen_host_table()?));
    let result = vm.run(&program, &PagedMemory::with_size(8)?)?;
    assert_eq!(result.stack[0].val, 6);
    println!(
        "approved finite scalar replay: result={}, fetched_records={}, host_effects={}",
        result.stack[0].val,
        result.instructions_executed,
        result.effects.len()
    );
    Ok(())
}
