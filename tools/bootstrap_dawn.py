# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_versions(path):
    pattern = re.compile(r'^set\(([A-Z0-9_]+) "([^"]*)"\)$')
    versions = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                versions[match.group(1)] = match.group(2)

    required = [
        "NIKKI_DAWN_COMMIT",
        "NIKKI_DAWN_DEPS_SHA256",
        "NIKKI_EMDAWNWEBGPU_PORT_FILENAME",
        "NIKKI_EMDAWNWEBGPU_PORT_SHA256",
        "NIKKI_EMSDK_VERSION",
        "NIKKI_ASYNCIFY_STACK_SIZE",
        "NIKKI_WASM_STACK_SIZE",
    ]
    missing = [name for name in required if name not in versions]
    if missing:
        raise RuntimeError("missing version variables: " + ", ".join(missing))

    return versions


def load_stamp(path):
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError("invalid dependency stamp %s: %s" % (path, e))


def write_stamp(path, dawn_commit, deps_sha256):
    stamp = {
        "dawn_commit": dawn_commit,
        "deps_sha256": deps_sha256,
        "schema": 1,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(stamp, f, indent=2, sort_keys=True)
        f.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the dependencies for the Dawn revision frozen by nikki."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace a stale nikki dependency stamp",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    versions_path = project_root / "CMakeLists.txt"
    versions = load_versions(versions_path)

    dawn_path = project_root / "third_party" / "dawn"
    if not (dawn_path / "CMakeLists.txt").is_file():
        raise RuntimeError("Dawn is not initialized at third_party/dawn")

    expected_commit = versions["NIKKI_DAWN_COMMIT"]
    deps_path = dawn_path / "DEPS"
    actual_deps_sha256 = sha256_file(deps_path)
    expected_deps_sha256 = versions["NIKKI_DAWN_DEPS_SHA256"]
    if actual_deps_sha256 != expected_deps_sha256:
        raise RuntimeError(
            "dawn DEPS SHA256 mismatch: expected %s, got %s"
            % (expected_deps_sha256, actual_deps_sha256)
        )

    port_path = (
        project_root
        / "ports"
        / versions["NIKKI_EMDAWNWEBGPU_PORT_FILENAME"]
    )
    if not port_path.is_file():
        raise RuntimeError("missing frozen Emdawnwebgpu port: %s" % port_path)

    actual_port_sha256 = sha256_file(port_path)
    expected_port_sha256 = versions["NIKKI_EMDAWNWEBGPU_PORT_SHA256"]
    if actual_port_sha256 != expected_port_sha256:
        raise RuntimeError(
            "Emdawnwebgpu port SHA256 mismatch: expected %s, got %s"
            % (expected_port_sha256, actual_port_sha256)
        )

    stamp_path = dawn_path / "out" / ".nikki-deps-stamp"
    expected_stamp = {
        "dawn_commit": expected_commit,
        "deps_sha256": expected_deps_sha256,
        "schema": 1,
    }
    stamp = load_stamp(stamp_path)
    if stamp == expected_stamp and not args.force:
        print("Dawn dependencies already match the nikki stamp.")
        return 0

    if stamp is not None and not args.force:
        raise RuntimeError(
            "Dawn dependency stamp does not match the frozen version set; "
            "inspect dawn/third_party and rerun with --force to bootstrap explicitly"
        )

    fetch_script = dawn_path / "tools" / "fetch_dawn_dependencies.py"
    subprocess.run(
        [
            sys.executable,
            str(fetch_script),
            "--directory",
            str(dawn_path),
        ],
        check=True,
        cwd=project_root,
    )

    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    write_stamp(stamp_path, expected_commit, expected_deps_sha256)
    print("Dawn dependency bootstrap complete.")
    print("  Dawn commit: %s" % expected_commit)
    print("  DEPS SHA256: %s" % expected_deps_sha256)
    print("  Emscripten: %s" % versions["NIKKI_EMSDK_VERSION"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print("bootstrap_dawn.py: %s" % e, file=sys.stderr)
        sys.exit(1)
