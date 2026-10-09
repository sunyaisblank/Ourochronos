#![cfg(target_os = "linux")]

use ourochronos::runtime::isolation::{run_bounded, ProcessLimits, ProcessStop};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn supervised_probe() {
    let Ok(mode) = std::env::var("OUROCHRONOS_SUPERVISOR_PROBE") else {
        return;
    };
    match mode.as_str() {
        "exit" => {
            println!("stdout marker");
            eprintln!("stderr marker");
        }
        "blocked" => std::thread::sleep(Duration::from_secs(30)),
        "output" => loop {
            println!("{}", "x".repeat(4096));
        },
        "memory" => {
            // Request address space without touching/allocating a GiB. The
            // kernel must reject this under the child's 64 MiB hard limit.
            let memory = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    1024 * 1024 * 1024,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_eq!(memory, libc::MAP_FAILED);
            println!("address space denied");
        }
        "descendant" => {
            // The group child holds inherited output pipes after the leader
            // exits. Supervision must terminate it and finish draining.
            let mut child = Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", "supervised_probe", "--nocapture"])
                .env("OUROCHRONOS_SUPERVISOR_PROBE", "blocked");
            // This leader must exit without waiting to exercise inherited-pipe
            // cleanup. The supervisor kills the group; the orphan is reparented
            // to the OS reaper. Waiting here would remove the tested boundary.
            #[allow(clippy::zombie_processes)]
            let _descendant = child.spawn().unwrap();
            println!("leader finished");
        }
        _ => panic!("unknown probe"),
    }
}

fn probe(mode: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "supervised_probe",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env("OUROCHRONOS_SUPERVISOR_PROBE", mode);
    command
}
fn limits() -> ProcessLimits {
    ProcessLimits {
        wall_time: Duration::from_secs(2),
        address_space_bytes: 64 * 1024 * 1024,
        output_bytes: 16 * 1024,
    }
}

#[test]
fn successful_supervision_preserves_status_and_both_streams() {
    let output = run_bounded(probe("exit"), limits(), None).unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stopped, None);
    assert!(String::from_utf8_lossy(&output.stdout).contains("stdout marker"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("stderr marker"));
    let output = run_bounded(probe("descendant"), limits(), None).unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stopped, None);
    assert!(String::from_utf8_lossy(&output.stdout).contains("leader finished"));
}

#[test]
fn blocked_child_output_and_address_space_have_enforced_limits() {
    let start = Instant::now();
    let output = run_bounded(
        probe("blocked"),
        ProcessLimits {
            wall_time: Duration::from_millis(40),
            ..limits()
        },
        None,
    )
    .unwrap();
    assert_eq!(output.stopped, Some(ProcessStop::WallTime));
    assert!(!output.status.success());
    assert!(start.elapsed() < Duration::from_secs(2));
    let output = run_bounded(
        probe("output"),
        ProcessLimits {
            output_bytes: 512,
            ..limits()
        },
        None,
    )
    .unwrap();
    assert_eq!(output.stopped, Some(ProcessStop::OutputLimit));
    assert!(output.stdout.len() + output.stderr.len() <= 512);
    let output = run_bounded(probe("memory"), limits(), None).unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stopped, None);
    assert!(String::from_utf8_lossy(&output.stdout).contains("address space denied"));
}

#[test]
fn cancellation_and_invalid_limits_cannot_produce_success() {
    let cancel = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(30));
            cancel.store(true, Ordering::Relaxed);
        });
        let output = run_bounded(probe("blocked"), limits(), Some(&cancel)).unwrap();
        assert_eq!(output.stopped, Some(ProcessStop::Cancelled));
        assert!(!output.status.success());
    });
    assert_eq!(
        run_bounded(probe("exit"), limits(), Some(&cancel))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Interrupted
    );
    assert_eq!(
        run_bounded(
            probe("exit"),
            ProcessLimits {
                wall_time: Duration::ZERO,
                ..limits()
            },
            None
        )
        .unwrap_err()
        .kind(),
        std::io::ErrorKind::InvalidInput
    );
}
