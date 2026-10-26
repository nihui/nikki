# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
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



# The upstream writer omits i8/u8 in matrix type emission. Keep this narrow
# downstream patch reproducible, pinned to both the revision and exact file bytes.
DAWN_PATCH_COMMIT = "1716e88cc9e58398fab5d4473f7f2e858667b677"
DAWN_PATCH_SOURCE = "src/tint/lang/wgsl/writer/ir_to_program/ir_to_program.cc"
DAWN_PATCH_PRISTINE_SHA256 = "8df9803a95ca20ee833f0c762310efdd399a71e60588fd5a5c0d5e43fa496d2d"
DAWN_PATCH_APPLIED_SHA256 = "2e678e7ebde7b897eb7a962ea3810caedb9f510ca30d0c7a7ee8d460af38dc32"


def apply_dawn_patch(dawn_path, patch_path, expected_commit):
    actual_commit = subprocess.check_output(
        ["git", "-C", str(dawn_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_commit != DAWN_PATCH_COMMIT or actual_commit != expected_commit:
        raise RuntimeError("Dawn patch revision mismatch: expected %s, got %s"
                           % (DAWN_PATCH_COMMIT, actual_commit))
    source_path = dawn_path / DAWN_PATCH_SOURCE
    source_hash = sha256_file(source_path)
    if source_hash == DAWN_PATCH_APPLIED_SHA256:
        # Also ensure the tracked patch still describes these applied bytes.
        # This is read-only and catches a stale/tampered patch on cache hits.
        subprocess.run(["git", "-C", str(dawn_path), "apply", "--reverse", "--check", str(patch_path)], check=True)
        return
    if source_hash != DAWN_PATCH_PRISTINE_SHA256:
        raise RuntimeError("Dawn matrix writer has unexpected contents; refusing to overwrite "
                           + str(source_path))
    command = ["git", "-C", str(dawn_path), "apply"]
    subprocess.run(command + ["--check", str(patch_path)], check=True)
    subprocess.run(command + [str(patch_path)], check=True)
    if sha256_file(source_path) != DAWN_PATCH_APPLIED_SHA256:
        raise RuntimeError("Dawn matrix writer patch produced unexpected contents")



def verify_dawn_dependencies(dawn_path):
    # Upstream's fetch helper can return success after a failed git subprocess.
    # Check actual checkouts against the pinned tree, including optional
    # dependencies already present, before trusting or writing our stamp.
    fetch_script = dawn_path / "tools" / "fetch_dawn_dependencies.py"
    tree = ast.parse(fetch_script.read_text(encoding="utf-8"))
    required = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "required_submodules"
            for target in node.targets
        ):
            required = set(ast.literal_eval(node.value))
            break
    if not required:
        raise RuntimeError("cannot determine the pinned Dawn fetch dependency set")
    entries = subprocess.check_output(
        ["git", "-C", str(dawn_path), "ls-tree", "-r", "HEAD"], text=True
    )
    checked = set()
    for line in entries.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, expected = metadata.split()
        if mode != "160000":
            continue
        dependency = dawn_path / path
        if not (dependency / ".git").exists():
            if path in required:
                raise RuntimeError("required Dawn dependency is missing: " + path)
            continue
        actual = subprocess.check_output(
            ["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError("Dawn dependency revision mismatch for %s: expected %s, got %s"
                               % (path, expected, actual))
        checked.add(path)
    if required - checked:
        raise RuntimeError("unverified required Dawn dependencies: " + ", ".join(sorted(required - checked)))


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


def write_stamp(path, stamp):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(stamp, f, indent=2, sort_keys=True)
        f.write("\n")
    temporary.replace(path)


def verify_dawn_state(dawn_path, project_root, versions):
    """Read-only validation for configure and packaging; never fetch or patch."""
    actual_commit = subprocess.check_output(
        ["git", "-C", str(dawn_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != versions["NIKKI_DAWN_COMMIT"] or actual_commit != DAWN_PATCH_COMMIT:
        raise RuntimeError("Dawn revision does not match the pinned patched dependency")
    if sha256_file(dawn_path / "DEPS") != versions["NIKKI_DAWN_DEPS_SHA256"]:
        raise RuntimeError("Dawn DEPS checksum mismatch")
    if sha256_file(dawn_path / DAWN_PATCH_SOURCE) != DAWN_PATCH_APPLIED_SHA256:
        raise RuntimeError("Dawn matrix writer patch is missing or modified; run tools/bootstrap_dawn.py")
    patch_hash = sha256_file(project_root / "patches/dawn-wgsl-matrix-i8.patch")
    expected_stamp = {
        "dawn_commit": actual_commit,
        "deps_sha256": versions["NIKKI_DAWN_DEPS_SHA256"],
        "matrix_writer_patch_sha256": patch_hash,
        "schema": 2,
    }
    if load_stamp(dawn_path / "out/.nikki-deps-stamp") != expected_stamp:
        raise RuntimeError("Dawn dependency stamp does not match the pinned patch; run tools/bootstrap_dawn.py --force")
    verify_dawn_dependencies(dawn_path)
    return patch_hash


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the dependencies for the Dawn revision frozen by nikki."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace a stale nikki dependency stamp",
    )
    parser.add_argument("--check", action="store_true", help="verify without modifying or downloading files")
    parser.add_argument("--dawn-source-dir", type=Path, help="Dawn checkout to bootstrap or verify")
    args = parser.parse_args()
    if args.check and args.force:
        parser.error("--check and --force cannot be combined")

    project_root = Path(__file__).resolve().parents[1]
    versions_path = project_root / "CMakeLists.txt"
    versions = load_versions(versions_path)

    dawn_path = (args.dawn_source_dir or project_root / "third_party" / "dawn").resolve()
    if args.check:
        verify_dawn_state(dawn_path, project_root, versions)
        print("Dawn revision, dependencies and applied matrix writer patch verified.")
        return 0
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
    patch_path = project_root / "patches" / "dawn-wgsl-matrix-i8.patch"
    expected_stamp = {
        "dawn_commit": expected_commit,
        "deps_sha256": expected_deps_sha256,
        "matrix_writer_patch_sha256": sha256_file(patch_path),
        "schema": 2,
    }
    stamp = load_stamp(stamp_path)
    if stamp is not None and stamp != expected_stamp and not args.force:
        raise RuntimeError(
            "Dawn dependency stamp does not match the frozen version set; "
            "inspect dawn/third_party and rerun with --force to bootstrap explicitly"
        )

    apply_dawn_patch(dawn_path, patch_path, expected_commit)
    if stamp == expected_stamp and not args.force:
        verify_dawn_dependencies(dawn_path)
        print("Dawn dependencies and matrix writer patch match the nikki stamp.")
        return 0

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

    verify_dawn_dependencies(dawn_path)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    write_stamp(stamp_path, expected_stamp)
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
