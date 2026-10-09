#!/usr/bin/env python3
"""Qualify actual CLI, REPL and feature-enabled LSP stdio interfaces.

Run against an explicitly selected, already built runtime, for example:
  python3 scripts/qualify_interfaces.py --runtime target/debug/ourochronos

This finite gate uses only synthetic temporary sources and managed compiler
outputs. It builds or installs nothing. Every child has input, output and wall
time caps, and is reaped on failure. LSP clients use increasing versions and
the server's advertised FULL synchronization; incremental edits and rename
are outside this server's advertised interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable


MAX_RUNTIME_BYTES = 512 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024
MAX_WRITE_BYTES = 4096
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_MESSAGE_BYTES = 128 * 1024
MAX_HEADER_BYTES = 8192
MAX_MESSAGES = 128
MAX_ARTIFACT_BYTES = 1024 * 1024
PROCESS_SECONDS = 10.0
CAMPAIGN_SECONDS = 60.0


class Failure(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def excerpt(data: bytes) -> str:
    return data[:1024].decode("utf-8", errors="replace")


def parse_version(stdout: str) -> tuple[str, dict[str, Any] | None]:
    """The first line is semver; later lines describe this actual build."""
    lines = stdout.splitlines()
    require(bool(lines) and re.fullmatch(r"ourochronos [0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?",
                                        lines[0]) is not None, "unexpected version response")
    require(len(lines[0]) < 128, "version line exceeds expected bound")
    if len(lines) == 1:
        require(lines[0] == "ourochronos 0.2.0", "current runtime omits platform/features discovery")
        return lines[0], None  # Explicitly allow the previously qualified pilot.
    require(len(lines) == 3, "unexpected version discovery line count")
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    operating_system = {"linux": "linux", "win32": "windows", "darwin": "macos"}.get(sys.platform)
    require(operating_system is not None, "gate does not recognize this host platform")
    expected_platform = f"{machine}-{operating_system}"
    require(lines[1] == f"platform: {expected_platform}; runtime ABI 1; native effect commits: Linux only",
            "version platform/runtime ABI disagrees with this gate's host and supported ABI")
    features = re.fullmatch(r"compiled features: lsp=(true|false) dynamic-ffi=(true|false); Z3 native library required",
                            lines[2])
    require(features is not None, "missing or malformed compiled feature discovery")
    require(features[1] == "true", "actual runtime lacks the LSP feature required by this interface gate")
    return lines[0], {"platform": expected_platform, "runtime_abi": 1,
                      "compiled_features": {"lsp": features[1] == "true", "dynamic-ffi": features[2] == "true"}}


class Child:
    """Bounded pipe readers work with both POSIX and Windows subprocesses."""

    def __init__(self, runtime: Path, args: list[str], deadline: float):
        self.deadline = min(deadline, time.monotonic() + PROCESS_SECONDS)
        self.started = time.monotonic()
        self.condition = threading.Condition()
        self.output = {"stdout": bytearray(), "stderr": bytearray()}
        self.eof = {"stdout": False, "stderr": False}
        self.problem: str | None = None
        self.input_bytes = 0
        self.cursor = 0
        self.process = subprocess.Popen(
            [str(runtime), *args], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            start_new_session=(os.name == "posix"),
        )
        self.readers = []
        for name in ("stdout", "stderr"):
            thread = threading.Thread(target=self._read, args=(name,), daemon=True)
            thread.start()
            self.readers.append(thread)

    def kill(self) -> None:
        try:
            if os.name == "posix":
                # The leader may already have exited while another child
                # still retains a pipe. Reap its isolated group as well.
                os.killpg(self.process.pid, signal.SIGKILL)
            elif self.process.poll() is None:
                self.process.kill()
        except ProcessLookupError:
            pass

    def _read(self, name: str) -> None:
        pipe = getattr(self.process, name)
        try:
            while chunk := pipe.read(4096):
                with self.condition:
                    remaining = MAX_OUTPUT_BYTES - sum(map(len, self.output.values()))
                    self.output[name].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        self.problem = "combined stdout/stderr exceeded output cap"
                    self.condition.notify_all()
                if self.problem:
                    self.kill()
                    break
        except OSError as error:
            with self.condition:
                self.problem = self.problem or f"pipe read failed: {error}"
        finally:
            with self.condition:
                self.eof[name] = True
                self.condition.notify_all()

    def send(self, payload: bytes) -> None:
        require(len(payload) <= MAX_WRITE_BYTES, "per-write input cap exceeded")
        self.input_bytes += len(payload)
        require(self.input_bytes <= MAX_INPUT_BYTES, "process input cap exceeded")
        done = threading.Event()
        errors: list[Exception] = []

        def write() -> None:
            try:
                view = memoryview(payload)
                while view:
                    count = self.process.stdin.write(view)
                    if not count:
                        raise BrokenPipeError("stdin write made no progress")
                    view = view[count:]
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                errors.append(error)
            finally:
                done.set()

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        if not done.wait(max(0.0, self.deadline - time.monotonic())):
            self.kill()
            writer.join(timeout=1)
            raise Failure("child stdin write timed out")
        writer.join()
        require(not errors, f"child stdin failed: {errors}")

    def close_input(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def wait(self) -> int:
        try:
            code = self.process.wait(timeout=max(0.0, self.deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            raise Failure("child did not exit within its wall time cap") from error
        for reader in self.readers:
            reader.join(timeout=max(0.0, self.deadline - time.monotonic()))
        require(all(not reader.is_alive() for reader in self.readers), "pipe drain timed out")
        require(self.problem is None, self.problem or "pipe failure")
        return code

    def message(self) -> dict[str, Any]:
        with self.condition:
            while True:
                require(self.problem is None, self.problem or "pipe failure")
                data = self.output["stdout"]
                boundary = data.find(b"\r\n\r\n", self.cursor)
                if boundary == -1:
                    require(len(data) - self.cursor <= MAX_HEADER_BYTES, "LSP header cap exceeded")
                else:
                    require(boundary - self.cursor <= MAX_HEADER_BYTES, "LSP header cap exceeded")
                    header = bytes(data[self.cursor:boundary])
                    lengths = []
                    for line in header.split(b"\r\n"):
                        name, separator, value = line.partition(b":")
                        require(bool(separator), "malformed LSP header")
                        if name.lower() == b"content-length":
                            value = value.strip()
                            require(value.isdigit() and len(value) <= 9, "invalid LSP content length")
                            lengths.append(int(value))
                    require(len(lengths) == 1, "missing or duplicate LSP content length")
                    length = lengths[0]
                    require(0 < length <= MAX_MESSAGE_BYTES, "LSP message cap exceeded")
                    end = boundary + 4 + length
                    if len(data) >= end:
                        body = bytes(data[boundary + 4:end])
                        self.cursor = end
                        try:
                            message = json.loads(body)
                        except (ValueError, RecursionError) as error:
                            raise Failure(f"invalid LSP JSON: {error}") from error
                        require(isinstance(message, dict) and message.get("jsonrpc") == "2.0",
                                "invalid JSON-RPC message envelope")
                        return message
                require(not self.eof["stdout"], "LSP stdout ended before the expected message")
                remaining = self.deadline - time.monotonic()
                require(remaining > 0, "LSP response timed out")
                self.condition.wait(remaining)

    def stats(self) -> dict[str, Any]:
        return {"pid": self.process.pid, "input_bytes": self.input_bytes,
                "stdout_bytes": len(self.output["stdout"]),
                "stderr_bytes": len(self.output["stderr"]),
                "seconds": round(time.monotonic() - self.started, 3)}

    def close(self) -> None:
        self.kill()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self.close_input()
        for pipe in (self.process.stdout, self.process.stderr):
            pipe.close()
        for reader in self.readers:
            reader.join(timeout=1)

    def __enter__(self) -> Child:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class Lsp:
    def __init__(self, child: Child):
        self.child = child
        self.next_id = 1
        self.count = 0
        self.pending: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        body = json.dumps({"jsonrpc": "2.0", **message}, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        self.child.send(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)

    def notify(self, method: str, params: Any) -> None:
        self.send({"method": method, "params": params})

    def receive(self) -> dict[str, Any]:
        self.count += 1
        require(self.count <= MAX_MESSAGES, "LSP message count cap exceeded")
        return self.child.message()

    def request(self, method: str, params: Any, *, error_code: int | None = None) -> Any:
        request_id = self.next_id
        self.next_id += 1
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = self.receive()
            if "id" not in message:
                self.pending.append(message)
                continue
            require(message["id"] == request_id, f"unexpected LSP response id: {message['id']}")
            if error_code is not None:
                require(message.get("error", {}).get("code") == error_code,
                        f"{method} did not return error {error_code}")
                return message["error"]
            require("result" in message and "error" not in message,
                    f"{method} failed: {str(message)[:1024]}")
            return message["result"]

    def diagnostics(self, uri: str, version: int | None) -> list[dict[str, Any]]:
        while True:
            message = self.pending.pop(0) if self.pending else self.receive()
            require("id" not in message, "unexpected response while awaiting diagnostics")
            if message.get("method") != "textDocument/publishDiagnostics":
                continue
            params = message.get("params", {})
            require(params.get("uri") == uri, "diagnostics published for an unexpected source")
            require(params.get("version") == version, "diagnostics have a stale or unexpected version")
            diagnostics = params.get("diagnostics")
            require(isinstance(diagnostics, list), "diagnostics must be an array")
            return diagnostics


def runtime_identity(runtime: Path) -> dict[str, Any]:
    require(runtime.is_file(), "runtime must be a regular file")
    size = runtime.stat().st_size
    require(0 < size <= MAX_RUNTIME_BYTES, "runtime size exceeds gate limit")
    digest = hashlib.sha256()
    count = 0
    with runtime.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            require(count <= MAX_RUNTIME_BYTES, "runtime grew beyond gate limit")
            digest.update(chunk)
    require(count == size, "runtime changed while hashing")
    return {"path": str(runtime), "bytes": size, "sha256": digest.hexdigest()}


class Campaign:
    def __init__(self, runtime: Path):
        self.runtime = runtime
        self.started = time.monotonic()
        self.deadline = self.started + CAMPAIGN_SECONDS
        self.cases: list[dict[str, Any]] = []
        self.current = "runtime identity"
        self.last_child: Child | None = None
        self.discovery: dict[str, Any] | None = None
        self.version_output = ""

    def case(self, name: str, check: Callable[[], None]) -> None:
        self.current = name
        require(time.monotonic() < self.deadline, "campaign time cap exceeded")
        started = time.monotonic()
        check()
        self.cases.append({"name": name, "seconds": round(time.monotonic() - started, 3)})

    def cli(self, args: list[str], code: int = 0, input_bytes: bytes = b"") -> tuple[str, str]:
        with Child(self.runtime, args, self.deadline) as child:
            self.last_child = child
            if input_bytes:
                child.send(input_bytes)
            child.close_input()
            actual = child.wait()
            stdout, stderr = (bytes(child.output[name]) for name in ("stdout", "stderr"))
            require(actual == code,
                    f"exit {actual}, expected {code}; stdout={excerpt(stdout)!r}; stderr={excerpt(stderr)!r}")
            return stdout.decode("utf-8"), stderr.decode("utf-8")


def qualify(campaign: Campaign, directory: Path) -> str:
    version = ""

    def version_case() -> None:
        nonlocal version
        stdout, _ = campaign.cli(["--version"])
        version, campaign.discovery = parse_version(stdout)
        campaign.version_output = stdout.replace("\r\n", "\n")

    campaign.case("CLI version", version_case)

    def help_case() -> None:
        stdout, _ = campaign.cli(["--help"])
        for value in ("Usage:", "ourochronos repl", "run-package", "--check",
                      "--emit-bytecode", "--build", "--lsp"):
            require(value in stdout, f"help omits {value}")

    campaign.case("CLI help and discovery", help_case)
    source = directory / "ordinary.ouro"
    source.write_text("40 2 ADD OUTPUT\n", encoding="utf-8")
    limits = ["--memory-cells", "8", "--max-inst", "1000"]

    def ordinary_case() -> None:
        stdout, _ = campaign.cli([str(source), *limits])
        require(stdout == "[42]\n", "ordinary arithmetic did not output exactly [42]")

    campaign.case("CLI first ordinary program", ordinary_case)
    library_dir, app_dir = directory / "library", directory / "application"
    library_dir.mkdir()
    app_dir.mkdir()
    library, module = library_dir / "dependency.ouro", app_dir / "main.ouro"
    module.write_text('IMPORT "../library/dependency.ouro"\nanswer OUTPUT\n', encoding="utf-8")
    library.write_text("PROCEDURE answer { 40 UNKNOWN_OPCODE ADD }\n", encoding="utf-8")

    def invalid_module_case() -> None:
        _, stderr = campaign.cli([str(module), *limits], code=1)
        require("dependency.ouro" in stderr and "UNKNOWN_OPCODE" in stderr,
                "invalid imported source diagnostic lost its source or offending token")

    campaign.case("CLI imported source error", invalid_module_case)
    library.write_text("PROCEDURE answer { 40 2 ADD }\n", encoding="utf-8")

    def module_case() -> None:
        stdout, _ = campaign.cli([str(module), *limits])
        require(stdout == "[42]\n", "corrected importer-relative module did not execute")
        campaign.cli([str(module), "--check", *limits])

    campaign.case("CLI corrected module and check", module_case)
    bytecode, package = directory / "main.ourobc", directory / "main.ouropkg"

    def build_case() -> None:
        for flag, path in (("--emit-bytecode", bytecode), ("--build", package)):
            campaign.cli([str(module), flag, str(path), *limits])
            require(0 < path.stat().st_size <= MAX_ARTIFACT_BYTES, "artifact size outside gate limit")
            original = path.read_bytes()
            require(original.startswith(b"OUROPA"), "build did not use portable artifact envelope")
            campaign.cli([str(module), flag, str(path), *limits])
            require(path.read_bytes() == original, "identical source build was not deterministic")

    campaign.case("CLI deterministic bytecode and package build", build_case)

    def repl_case() -> None:
        commands = (":help\nunknown OUTPUT\n40 2 ADD OUTPUT\n1 0 PROPHECY\n"
                    "0 ORACLE DUP OUTPUT 0 PROPHECY\n:history\n:clear\n"
                    f":load {module}\n:quit\n").encode("utf-8")
        stdout, _ = campaign.cli(["repl"], input_bytes=commands)
        require(f"OUROCHRONOS REPL v{version.removeprefix('ourochronos ')}" in stdout,
                "REPL banner disagrees with the compiled CLI version")
        for value in ("OUROCHRONOS REPL", "REPL Commands", "P001", "Output: [42]",
                      "Output: [1]", "Memory cleared.", "1: unknown OUTPUT", "Goodbye!"):
            require(value in stdout, f"REPL journey omitted {value}")
        require(stdout.count("Output: [42]") == 2,
                "REPL did not recover and load the corrected importer-relative module")
        require(stdout.index("P001") < stdout.index("Output: [42]"), "REPL error recovery order differs")

    campaign.case("REPL help, error recovery, retained state, clear and module load", repl_case)
    module.unlink()
    library.unlink()

    def reload_case() -> None:
        stdout, _ = campaign.cli(["run-package", str(package)])
        require(stdout == "[42]\n", "package reload after source removal did not output exactly [42]")
        # The CLI exposes classical bytecode execution through halt-slice; it
        # does not advertise running bare bytecode as a positional source.
        stdout, _ = campaign.cli(["halt-slice", str(bytecode), str(directory / "saved.ourocp"),
                                  "1000", "8", "1000"])
        require("HALTED after" in stdout and "[42]" in stdout,
                "emitted bytecode did not reload and execute after source removal")

    campaign.case("CLI package and bytecode reload without original sources", reload_case)

    def bad_artifact_case() -> None:
        _, stderr = campaign.cli(["run-package", str(bytecode)], code=1)
        require("no package runtime/resolution manifest" in stderr, "package kind gate failed")
        corrupt = bytearray(package.read_bytes())
        corrupt[-1] ^= 1
        bad = directory / "corrupt.ouropkg"
        bad.write_bytes(corrupt)
        _, stderr = campaign.cli(["run-package", str(bad)], code=1)
        require("checksum" in stderr, "corrupt artifact was not rejected by checksum")
        stdout, _ = campaign.cli(["run-package", str(package)])
        require(stdout == "[42]\n", "CLI did not recover after rejected artifact")

    campaign.case("CLI wrong artifact kind, corruption and recovery", bad_artifact_case)
    qualify_lsp(campaign, directory, version)

    def final_version_case() -> None:
        stdout, _ = campaign.cli(["--version"])
        require(stdout.replace("\r\n", "\n") == campaign.version_output,
                "version/platform/features changed during qualification")
        require(parse_version(stdout) == (version, campaign.discovery), "final discovery identity differs")

    campaign.case("CLI final version, platform and feature identity", final_version_case)
    return version


def qualify_lsp(campaign: Campaign, directory: Path, version: str) -> None:
    with Child(campaign.runtime, ["--lsp"], campaign.deadline) as child:
        campaign.last_child = child
        lsp = Lsp(child)
        uri = (directory / "editor.ouro").as_uri()
        fixed = '"😀" POP\nPROCEDURE answer { 41 1 ADD }\nanswer OUTPUT\n\n'

        def initialize_case() -> None:
            result = lsp.request("initialize", {"processId": os.getpid(),
                                 "rootUri": directory.as_uri(), "capabilities": {}})
            require(isinstance(result, dict), "initialize result must be an object")
            capabilities = result.get("capabilities", {})
            require("capabilities" not in capabilities, "initialize double-wrapped capabilities")
            for value in ("completionProvider", "hoverProvider", "definitionProvider",
                          "documentSymbolProvider", "semanticTokensProvider"):
                require(bool(capabilities.get(value)), f"LSP does not advertise {value}")
            require(capabilities.get("textDocumentSync", {}).get("change") == 1,
                    "gate requires advertised FULL document synchronization")
            require("renameProvider" not in capabilities, "gate's supported-set contract needs updating")
            info = result.get("serverInfo", {})
            require(info.get("name") == "ourochronos-lsp"
                    and info.get("version") == version.removeprefix("ourochronos "),
                    "LSP server identity/version does not match CLI")
            lsp.notify("initialized", {})

        campaign.case("LSP initialize shape and advertised supported set", initialize_case)

        def open_case() -> None:
            lsp.notify("textDocument/didOpen", {"textDocument": {"uri": uri,
                       "languageId": "ourochronos", "version": 1, "text": "unknown OUTPUT\n"}})
            diagnostics = lsp.diagnostics(uri, 1)
            require(any(item.get("severity") == 1 for item in diagnostics),
                    "invalid opened document did not publish an error")

        campaign.case("LSP invalid open diagnostics", open_case)

        def change_case() -> None:
            lsp.notify("textDocument/didChange", {"textDocument": {"uri": uri, "version": 2},
                       "contentChanges": [{"text": "still_unknown OUTPUT\n"}, {"text": fixed}]})
            require(not lsp.diagnostics(uri, 2), "ordered FULL changes did not leave corrected final text")
            lsp.notify("textDocument/didChange", {"textDocument": {"uri": uri, "version": 7},
                       "contentChanges": [{"text": fixed}]})
            require(not lsp.diagnostics(uri, 7), "increasing nonconsecutive version was not synchronized")

        campaign.case("LSP ordered FULL edits and versioned recovery", change_case)

        def hover_case() -> None:
            result = lsp.request("textDocument/hover", {"textDocument": {"uri": uri},
                                 "position": {"line": 0, "character": 5}})
            require(result is not None and "**POP**" in str(result.get("contents")),
                    "UTF-16 position after astral character did not resolve POP hover")
            require(result.get("range") == {"start": {"line": 0, "character": 5},
                                             "end": {"line": 0, "character": 8}},
                    "hover range is not in UTF-16 document coordinates")

        campaign.case("LSP useful hover and UTF-16 coordinates", hover_case)

        def definition_case() -> None:
            result = lsp.request("textDocument/definition", {"textDocument": {"uri": uri},
                                 "position": {"line": 2, "character": 1}})
            require(isinstance(result, dict) and result.get("uri") == uri
                    and result.get("range", {}).get("start", {}).get("line") == 1,
                    "procedure call did not navigate to its actual definition")

        campaign.case("LSP procedure definition", definition_case)

        def completion_case() -> None:
            result = lsp.request("textDocument/completion", {"textDocument": {"uri": uri},
                                 "position": {"line": 3, "character": 0}})
            items = result.get("items") if isinstance(result, dict) else result
            require(isinstance(items, list), "completion did not return items")
            labels = {item.get("label") for item in items}
            require({"ADD", "OUTPUT", "answer"} <= labels, "completion omitted opcode or user procedure")

        campaign.case("LSP useful opcode and procedure completion", completion_case)

        def symbols_case() -> None:
            result = lsp.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
            require(isinstance(result, list) and any(item.get("name") == "answer" for item in result),
                    "document symbols omit the defined procedure")
            result = lsp.request("textDocument/semanticTokens/full", {"textDocument": {"uri": uri}})
            data = result.get("data", []) if isinstance(result, dict) else []
            require(bool(data) and len(data) % 5 == 0
                    and all(type(value) is int and 0 <= value <= 0xFFFFFFFF for value in data),
                    "semantic token stream is empty or malformed")
            line = column = 0
            tokens = []
            for delta_line, delta_start, length, kind, modifiers in zip(*[iter(data)] * 5):
                line += delta_line
                column = delta_start if delta_line else column + delta_start
                tokens.append((line, column, length, kind, modifiers))
            require(any(token[:3] == (0, 5, 3) for token in tokens),
                    "semantic tokens do not place POP after the astral character in UTF-16")
            lsp.request("textDocument/rename", {"textDocument": {"uri": uri},
                        "position": {"line": 2, "character": 1}, "newName": "new_answer"},
                        error_code=-32601)

        campaign.case("LSP symbols, semantic tokens and explicit unsupported rename", symbols_case)

        def close_case() -> None:
            lsp.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
            require(not lsp.diagnostics(uri, None), "closing a document did not clear diagnostics")

        campaign.case("LSP close clears diagnostics", close_case)

        def shutdown_case() -> None:
            require(lsp.request("shutdown", None) is None, "shutdown result must be null")
            lsp.notify("exit", None)
            child.close_input()
            require(child.wait() == 0, "LSP shutdown/exit returned a failure")

        campaign.case("LSP shutdown, exit and process cleanup", shutdown_case)
        campaign.cases[-1]["process"] = child.stats()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", required=True, type=Path,
                        help="exact already-built LSP-enabled runtime path; never builds or installs")
    parser.add_argument("--report", type=Path, help="optional JSON report output (also printed to stdout)")
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    campaign = Campaign(runtime)
    identity: dict[str, Any] = {"path": str(runtime)}
    report_path: Path | None = None
    try:
        if args.report:
            proposed_report = args.report.expanduser().resolve()
            require(proposed_report not in (runtime, Path(__file__).resolve()),
                    "report output collides with the runtime or qualification gate")
            report_path = proposed_report
        identity = runtime_identity(runtime)
        # Spaces exercise argument handling, :load and percent-encoded URIs.
        with tempfile.TemporaryDirectory(prefix="ouro interface qualification ") as temporary:
            version = qualify(campaign, Path(temporary).resolve())
        campaign.current = "final runtime identity"
        require(runtime_identity(runtime) == identity, "runtime bytes changed during qualification")
        report = {"schema": "ourochronos.interface-qualification/1", "status": "passed",
                  "runtime": {**identity, "version": version, "discovery": campaign.discovery}, "cases": campaign.cases,
                  "case_count": len(campaign.cases),
                  "seconds": round(time.monotonic() - campaign.started, 3),
                  "bounds": {"campaign_seconds": CAMPAIGN_SECONDS,
                             "process_seconds": PROCESS_SECONDS, "process_input_bytes": MAX_INPUT_BYTES,
                             "combined_process_output_bytes": MAX_OUTPUT_BYTES,
                             "lsp_message_bytes": MAX_MESSAGE_BYTES, "lsp_messages": MAX_MESSAGES},
                  "scope": "finite synthetic actual-process gate; FULL sync, increasing client versions; no native host effects"}
    except (Failure, OSError, UnicodeError, ValueError, TypeError, AttributeError, KeyError) as error:
        report = {"schema": "ourochronos.interface-qualification/1", "status": "failed",
                  "runtime": identity, "case": campaign.current, "error": str(error)[:2048],
                  "passed_cases": campaign.cases,
                  "seconds": round(time.monotonic() - campaign.started, 3)}
        if campaign.last_child:
            report["last_process"] = campaign.last_child.stats()
            report["stdout_excerpt"] = excerpt(bytes(campaign.last_child.output["stdout"]))
            report["stderr_excerpt"] = excerpt(bytes(campaign.last_child.output["stderr"]))
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if report_path:
        report_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
