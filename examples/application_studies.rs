//! Configurable self-consistency, circular dataflow, and rule-game workflows.
//! Linux default runs under 15 s / 512 MiB / 1 MiB output supervision. `raw`
//! explicitly bypasses OS supervision on every platform; logical caps remain.
mod studies;

fn raw(args: &[String]) -> i32 {
    let result = (|| -> Result<bool, String> {
        let (study, gas, save) = studies::parse_cli(args)?;
        let report = studies::run(study, gas)?;
        let text = report.text()?;
        if let Some(path) = save {
            studies::save_new(&path, &text)?;
        }
        print!("{text}");
        Ok(report.is_unknown())
    })();
    match result {
        Ok(false) => 0,
        Ok(true) => 3,
        Err(error) => {
            eprintln!("Application study error: {error}");
            1
        }
    }
}

fn supervised(args: &[String]) -> Result<i32, String> {
    use ourochronos::runtime::isolation::{run_bounded, ProcessLimits};
    use std::io::Write;
    let (_, _, save) = studies::parse_cli(args)?;
    let mut worker_args = Vec::new();
    let mut cursor = 0;
    while cursor < args.len() {
        if args[cursor] == "--save" {
            cursor += 2;
        } else {
            worker_args.push(args[cursor].clone());
            cursor += 1;
        }
    }
    let mut command =
        std::process::Command::new(std::env::current_exe().map_err(|error| error.to_string())?);
    command.arg("raw").args(worker_args);
    let output = run_bounded(
        command,
        ProcessLimits {
            wall_time: std::time::Duration::from_secs(15),
            address_space_bytes: 512 * 1024 * 1024,
            output_bytes: 1024 * 1024,
        },
        None,
    )
    .map_err(|error| {
        format!("supervision unavailable: {error}; use explicit raw mode for logical caps only")
    })?;
    if output.stopped.is_some() || output.status.code().is_none() {
        eprintln!("UNKNOWN application workflow: stop={:?} status={} elapsed_ms={}; partial output withheld, requested result file not created", output.stopped, output.status, output.elapsed.as_millis());
        return Ok(3);
    }
    let code = output.status.code().unwrap();
    if code == 0 || code == 3 {
        let text = format!("supervision: Linux wall_ms=15000 address_space_bytes=536870912 output_bytes=1048576 stop=None elapsed_ms={}\n{}", output.elapsed.as_millis(), String::from_utf8(output.stdout).map_err(|error| error.to_string())?);
        if let Some(path) = save {
            studies::save_new(&path, &text)?;
        }
        print!("{text}");
    } else {
        std::io::stdout()
            .write_all(&output.stdout)
            .map_err(|error| error.to_string())?;
    }
    std::io::stderr()
        .write_all(&output.stderr)
        .map_err(|error| error.to_string())?;
    Ok(code)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = if args.first().is_some_and(|argument| argument == "raw") {
        raw(&args[1..])
    } else {
        match supervised(&args) {
            Ok(code) => code,
            Err(error) => {
                eprintln!("Application study error: {error}");
                1
            }
        }
    };
    std::process::exit(code);
}
