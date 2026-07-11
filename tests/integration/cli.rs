//! CLI contract tests: exit codes and argument handling.
//!
//! The binary's exit code is the scripting interface to convergence:
//! 0 consistent, 1 error, 2 paradox, 3 epoch exhaustion. These tests pin
//! that contract by invoking the built binary directly.

use std::process::Command;

fn ouro() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ourochronos"))
}

#[test]
fn help_exits_zero_and_prints_usage() {
    let out = ouro().arg("--help").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Usage: ourochronos"));
    assert!(text.contains("Exit codes"));
}

#[test]
fn version_exits_zero() {
    let out = ouro().arg("--version").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn no_arguments_exits_one_with_usage() {
    let out = ouro().output().expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage"));
}

#[test]
fn missing_file_exits_one_without_panic() {
    let out = ouro().arg("does-not-exist.ouro").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("cannot read"));
    assert!(!err.contains("panicked"));
}

#[test]
fn unknown_flag_exits_one() {
    let out = ouro()
        .args(["examples/hello.ouro", "--bogus"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown option"));
}

#[test]
fn malformed_flag_value_exits_one() {
    let out = ouro()
        .args(["examples/hello.ouro", "--seed", "abc"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("requires a number"));
}

#[test]
fn consistent_program_exits_zero() {
    let out = ouro().arg("examples/hello.ouro").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Hello World!"));
}

#[test]
fn oscillating_paradox_exits_two() {
    let out = ouro().arg("examples/paradox.ouro").output().expect("binary runs");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("OSCILLATION"));
}

#[test]
fn non_converging_program_exits_three() {
    // A strictly increasing counter never revisits a state and never
    // converges, so the search exhausts max_epochs.
    let dir = std::env::temp_dir().join("ouro-cli-tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("diverge.ouro");
    std::fs::write(&path, "0 ORACLE 1 ADD 0 PROPHECY\n").expect("write programme");

    let out = ouro().arg(path.to_str().unwrap()).output().expect("binary runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "stdout was: {}", text);
    assert!(text.contains("TIMEOUT"));
}

#[test]
fn diagnostic_mode_freezes_interactive_input() {
    // The Temporal Input Invariant: the first epoch's inputs are replayed in
    // every later epoch. Before the fix only standard mode froze inputs;
    // diagnostic mode re-read stdin each epoch, so with input lines 5, 9, 7
    // the search would drift to the EOF fallback (0) instead of converging
    // on the first value read.
    use std::io::Write;
    use std::process::Stdio;

    let dir = std::env::temp_dir().join("ouro-cli-tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("frozen_input.ouro");
    std::fs::write(&path, "INPUT DUP 0 PROPHECY OUTPUT 0 ORACLE POP\n").expect("write programme");

    let mut child = ouro()
        .args([path.to_str().unwrap(), "--diagnostic"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary runs");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"5\n9\n7\n")
        .expect("pipe input");
    let out = child.wait_with_output().expect("binary finishes");

    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout was: {}", text);
    assert!(text.contains("[5]"), "expected frozen input 5, stdout was: {}", text);
}

#[test]
fn effects_flag_gates_nondeterminism_in_search() {
    let dir = std::env::temp_dir().join("ouro-cli-tests");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("clock_in_loop.ouro");
    std::fs::write(&path, "0 ORACLE POP CLOCK POP 0 0 PROPHECY\n").expect("write programme");

    // Default policy declines with a message naming the opcode.
    let out = ouro().arg(path.to_str().unwrap()).output().expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("CLOCK"));

    // Unrestricted policy lets the search run to a consistent timeline.
    let out = ouro()
        .args([path.to_str().unwrap(), "--effects", "unrestricted"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(0));

    // Garbage policy value is a usage error.
    let out = ouro()
        .args([path.to_str().unwrap(), "--effects", "sometimes"])
        .output()
        .expect("binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--effects"));
}
