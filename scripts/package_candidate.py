#!/usr/bin/env python3
"""Create deterministic unsigned runtime archives with exact provenance/notices.

Run from the source repository after cargo package verification. This performs
no build, installation, signing or publication. The caller supplies the actual
qualified runtime and native Z3 bytes; metadata describes this local candidate.
Python 3.11+ is needed only by this packaging tool, never by the runtime.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, timeout=120)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    names = sorted(set(command("git", "ls-files", "-co", "--exclude-standard", "-z").split(b"\0")) - {b""})
    identity = hashlib.sha256()
    for name in names:
        path = ROOT / name.decode()
        if not path.is_file():
            raise ValueError(f"source path is missing or not a file: {name!r}")
        identity.update(name + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return identity.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def check_source_archive(data, package):
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("source archive exceeds 128 MiB")
    prefix = f"{package['name']}-{package['version']}/"
    # Cargo owns normalization, target discovery and generated VCS metadata.
    # Recreate these without building in an isolated directory: checking only
    # Cargo.toml.orig would permit altered build paths/dependencies in Cargo.toml.
    with tempfile.TemporaryDirectory(prefix="ourochronos-package-identity-") as directory:
        command("cargo", "package", "--no-verify", "--allow-dirty", "--locked", "--target-dir", directory)
        generated = Path(directory) / "package" / f"{package['name']}-{package['version']}.crate"
        with tarfile.open(generated, mode="r:gz") as archive:
            expected = {entry.name[len(prefix):]: archive.extractfile(entry).read() for entry in archive}
    seen = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for entry in archive:
            if not entry.isfile() or not entry.name.startswith(prefix):
                raise ValueError("source archive has an invalid entry kind or package identity")
            name = entry.name[len(prefix):]
            if name not in expected or name in seen or len(seen) >= 4096:
                raise ValueError(f"source archive has unexpected/duplicate paths: {name}")
            if entry.size > 8 * 1024 * 1024 or total + entry.size > 64 * 1024 * 1024:
                raise ValueError("source archive exceeds bounded expanded size")
            total += entry.size
            seen.add(name)
            contents = archive.extractfile(entry).read()
            if contents != expected[name]:
                raise ValueError(f"source archive is stale or altered: {name}")
    if seen != set(expected):
        raise ValueError("source archive is missing current packaged files")


def build_files(args):
    windows = args.target == "x86_64-pc-windows-gnu"
    expected_names = ("ourochronos.exe", "libz3.dll") if windows else ("ourochronos", "libz3.so.4")
    if (args.runtime.name, args.native_lib.name) != expected_names:
        raise ValueError(f"target requires runtime/native-library names {expected_names}")
    snapshot = source_identity()
    package = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    checksums = {(row["name"], row["version"]): row.get("checksum") for row in lock["package"]}
    feature_flags = [] if args.features == "default" else ["--features", args.features]
    compiler = command("rustc", "--version", "--verbose").decode()
    host = next(line.removeprefix("host: ") for line in compiler.splitlines() if line.startswith("host: "))
    metadata = json.loads(command("cargo", "metadata", *feature_flags, "--locked", "--format-version", "1"))
    # Cargo's tree traversal accounts for compiler-host build units even when
    # the runtime target differs. A target-filtered metadata graph does not.
    # Notices need the union; we do not claim separate binary/build attribution.
    tree = command("cargo", "tree", *feature_flags, "--locked", "--target", args.target,
                   "--edges", "normal,build", "--prefix", "none", "--format", "{p}").decode()
    required = set()
    for line in tree.splitlines():
        match = re.match(r"^(\S+) v(\S+)(?:\s|$)", line)
        if not match:
            raise ValueError(f"unrecognized Cargo dependency identity: {line}")
        required.add(match.groups())
    package_ids = {}
    for item in metadata["packages"]:
        identity = (item["name"], item["version"])
        if identity in package_ids:
            raise ValueError(f"ambiguous Cargo dependency identity: {identity}")
        package_ids[identity] = item["id"]
    if not required <= package_ids.keys():
        raise ValueError("Cargo dependency tree and inventory disagree")
    files = {
        "LICENSE": (ROOT / "LICENSE").read_bytes(),
        "README.md": (ROOT / "README.md").read_bytes(),
        "examples/hello.ouro": (ROOT / "examples/hello.ouro").read_bytes(),
        "runtime/" + args.runtime.name: args.runtime.read_bytes(),
        "runtime/" + args.native_lib.name: args.native_lib.read_bytes(),
        "licenses/native-z3.txt": args.native_license.read_bytes(),
    }
    source_archive = args.source_archive.read_bytes()
    check_source_archive(source_archive, package)
    files["source/" + args.source_archive.name] = source_archive
    inventory = []
    for item in sorted(metadata["packages"], key=lambda value: (value["name"], value["version"])):
        row = {"name": item["name"], "version": item["version"], "license_expression": item["license"],
               "registry_source": item["source"], "registry_checksum": checksums.get((item["name"], item["version"])),
               "role": "resolved-runtime-or-build" if (item["name"], item["version"]) in required else "development-or-other-target"}
        inventory.append(row)
        if (item["name"], item["version"]) not in required or item["source"] is None:
            continue
        directory = Path(item["manifest_path"]).parent
        notices = [path for path in directory.iterdir() if path.is_file() and path.name.lower().startswith(("license", "notice", "copying", "copyright"))]
        if item["license_file"]:
            notices.append(directory / item["license_file"])
        if not notices:
            raise ValueError(f"required dependency notices unavailable: {item['name']} {item['version']}")
        for path in sorted(set(notices)):
            files[f"licenses/{item['name']}-{item['version']}/{path.name}"] = path.read_bytes()
    rust_docs = Path(command("rustc", "--print", "sysroot").decode().strip()) / "share/doc/rust"
    for name in ("COPYRIGHT", "LICENSE-APACHE", "LICENSE-MIT"):
        files["licenses/rust/" + name] = (rust_docs / name).read_bytes()
    files["dependencies.json"] = json_bytes(inventory)
    provenance = {"package": package["name"], "version": package["version"], "target": args.target,
                  "signature": "unsigned-local-qualification-candidate", "features": args.features,
                  "source_revision": command("git", "rev-parse", "HEAD").decode().strip(),
                  "source_snapshot_sha256": snapshot, "source_archive_sha256": sha(source_archive),
                  "compiler": compiler, "compiler_host": host,
                  "build_provenance": args.build_provenance,
                  "native_solver": "Z3 4.8.12; caller-supplied qualified bytes and license",
                  "runtime_sha256": sha(files["runtime/" + args.runtime.name]),
                  "native_solver_sha256": sha(files["runtime/" + args.native_lib.name]),
                  "formats": {"bytecode": 2, "package": 3, "portable_envelope": 1, "finite_proof": 1,
                              "classical_checkpoint": 1, "host_snapshot": 1}}
    files["provenance.json"] = json_bytes(provenance)
    files["SHA256SUMS"] = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(files.items())).encode()
    if source_identity() != snapshot:
        raise ValueError("source changed while collecting the candidate")
    if (args.runtime.read_bytes() != files["runtime/" + args.runtime.name]
            or args.native_lib.read_bytes() != files["runtime/" + args.native_lib.name]
            or args.native_license.read_bytes() != files["licenses/native-z3.txt"]
            or args.source_archive.read_bytes() != source_archive):
        raise ValueError("candidate inputs changed while collecting the archive")
    return files


def archive(files, windows):
    out = io.BytesIO()
    if windows:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as writer:
            for name, data in sorted(files.items()):
                entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = (0o100644 << 16)
                entry.compress_type = zipfile.ZIP_DEFLATED
                writer.writestr(entry, data)
    else:
        with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as writer:
                for name, data in sorted(files.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    entry.mode = 0o755 if name == "runtime/ourochronos" else 0o644
                    entry.mtime = 0
                    writer.addfile(entry, io.BytesIO(data))
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--native-license", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--target", choices=("x86_64-unknown-linux-gnu", "x86_64-pc-windows-gnu"), required=True)
    parser.add_argument("--features", required=True, help="exact qualified build features, e.g. lsp,dynamic-ffi or default")
    parser.add_argument("--build-provenance", required=True, help="actual builder/tool/native dependency identity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = archive(build_files(args), args.target == "x86_64-pc-windows-gnu")
    with args.output.open("xb") as output:
        output.write(payload)
    print(f"{sha(payload)}  {args.output}")


if __name__ == "__main__":
    main()
