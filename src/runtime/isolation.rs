//! Linux process resource supervision for finite work that may not cooperate
//! with cancellation. This bounds address space, CPU, retained output and wall
//! time; it is not filesystem/network confinement or an arbitrary-code sandbox.
//! The supervised process must not escape its process group. Scheduling and
//! killable kernel work remain OS assumptions. A killed run supplies no proof.

use std::io;
use std::process::{Command, ExitStatus};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessLimits {
    pub wall_time: Duration,
    pub address_space_bytes: u64,
    pub output_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessStop {
    WallTime,
    Cancelled,
    OutputLimit,
}

#[derive(Debug)]
pub struct SupervisedOutput {
    pub status: ExitStatus,
    pub stopped: Option<ProcessStop>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub elapsed: Duration,
}

/// Supervise an explicitly supplied command. Stdin is closed and stdout/stderr
/// share one byte ceiling. Setup failure returns an error before execution.
/// Non-Linux systems reject this profile before starting a child.
pub fn run_bounded(
    mut command: Command,
    limits: ProcessLimits,
    cancellation: Option<&AtomicBool>,
) -> io::Result<SupervisedOutput> {
    if limits.wall_time.is_zero()
        || limits.wall_time > Duration::from_secs(3600)
        || !(16 * 1024 * 1024..=16 * 1024 * 1024 * 1024).contains(&limits.address_space_bytes)
        || limits.output_bytes > 16 * 1024 * 1024
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "process limits exceed the supported profile",
        ));
    }
    platform::run(&mut command, limits, cancellation)
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;
    pub fn run(
        _: &mut Command,
        _: ProcessLimits,
        _: Option<&AtomicBool>,
    ) -> io::Result<SupervisedOutput> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process resource supervision requires Linux",
        ))
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    struct ChildGuard(Option<Child>);
    impl ChildGuard {
        fn stop_and_wait(&mut self) -> io::Result<ExitStatus> {
            let child = self
                .0
                .as_mut()
                .expect("supervised child is retained until wait");
            // The group leader has not been reaped, so its pid cannot be reused.
            // SAFETY: negating the positive retained child pid addresses only
            // the process group installed for that child before exec.
            if unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) } != 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error);
                }
            }
            let status = child.wait()?;
            self.0 = None;
            Ok(status)
        }
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if self.0.is_some() {
                let _ = self.stop_and_wait();
            }
        }
    }

    fn nonblocking(fd: libc::c_int) -> io::Result<()> {
        // SAFETY: fd belongs to a live child pipe; no ownership is transferred.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn exited_without_reaping(pid: u32) -> io::Result<bool> {
        // SAFETY: zero is valid siginfo storage; waitid initializes it. WNOWAIT
        // retains the child identity until group termination and Child::wait.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                return Ok(false);
            }
            return Err(error);
        }
        Ok(unsafe { info.si_pid() } != 0)
    }

    /// At most one bounded chunk per pipe per poll, keeping cancellation fair
    /// even when a child produces bytes faster than the parent can read them.
    fn read_chunk(
        reader: &mut impl Read,
        target: &mut Vec<u8>,
        retained: &mut usize,
        limit: usize,
    ) -> io::Result<bool> {
        let mut bytes = [0u8; 4096];
        match reader.read(&mut bytes) {
            Ok(0) => Ok(false),
            Ok(count) => {
                let keep = count.min(limit.saturating_sub(*retained));
                target.try_reserve(keep).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        "supervised output allocation failed",
                    )
                })?;
                target.extend_from_slice(&bytes[..keep]);
                *retained = retained.saturating_add(count);
                Ok(true)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }

    pub fn run(
        command: &mut Command,
        limits: ProcessLimits,
        cancellation: Option<&AtomicBool>,
    ) -> io::Result<SupervisedOutput> {
        if cancellation.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "process cancelled before spawn",
            ));
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        let cpu_seconds = limits.wall_time.as_secs().saturating_add(1);
        // SAFETY: this closure makes only async-signal-safe libc calls between
        // fork and exec; captured limits are plain integers and no locks/heap
        // allocations or Rust environment mutation occur here.
        unsafe {
            command.pre_exec(move || {
                for (resource, value) in [
                    (libc::RLIMIT_AS, limits.address_space_bytes),
                    (libc::RLIMIT_CPU, cpu_seconds),
                    (libc::RLIMIT_CORE, 0),
                ] {
                    let bound = libc::rlimit {
                        rlim_cur: value,
                        rlim_max: value,
                    };
                    if libc::setrlimit(resource, &bound) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let start = Instant::now();
        let mut guard = ChildGuard(Some(command.spawn()?));
        let child = guard.0.as_mut().expect("new child");
        let pid = child.id();
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");
        nonblocking(stdout.as_raw_fd())?;
        nonblocking(stderr.as_raw_fd())?;
        let mut output = Vec::new();
        let mut errors = Vec::new();
        let mut retained = 0;
        let mut stopped = None;
        let mut leader_exited = false;
        loop {
            let out_open =
                read_chunk(&mut stdout, &mut output, &mut retained, limits.output_bytes)?;
            let err_open =
                read_chunk(&mut stderr, &mut errors, &mut retained, limits.output_bytes)?;
            if retained > limits.output_bytes {
                stopped = Some(ProcessStop::OutputLimit);
                break;
            }
            if cancellation.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                stopped = Some(ProcessStop::Cancelled);
                break;
            }
            if start.elapsed() >= limits.wall_time {
                stopped = Some(ProcessStop::WallTime);
                break;
            }
            if !leader_exited && exited_without_reaping(pid)? {
                leader_exited = true;
                // Terminate descendants that inherited a pipe, before any
                // reap permits the group leader's numeric pid to be reused.
                if unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) } != 0 {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(error);
                    }
                }
            }
            if leader_exited && !out_open && !err_open {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let status = guard.stop_and_wait()?;
        Ok(SupervisedOutput {
            status,
            stopped,
            stdout: output,
            stderr: errors,
            elapsed: start.elapsed(),
        })
    }
}
