#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0
"""Validate an installed WASM SDK and create the date-versioned release archive."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import tarfile

from bootstrap_dawn import load_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"20\d{6}", args.version):
        parser.error("version must be YYYYMMDD in 2000-2099")
    datetime.strptime(args.version, "%Y%m%d")
    root = Path(__file__).resolve().parents[1]
    stage = args.stage.resolve()
    versions = load_versions(root / "CMakeLists.txt")
    expected = ["lib/libnikki.a", "lib/cmake/nikki/nikkiConfig.cmake",
                "lib/cmake/nikki/nikkiConfigVersion.cmake", "include/vulkan/vulkan.h",
                "include/nikki/version.h", "LICENSE.txt", "THIRD_PARTY_NOTICES.md",
                "share/nikki/ports/" + versions["NIKKI_EMDAWNWEBGPU_PORT_FILENAME"],
                "share/nikki/ports/webgpu_immediates.externs.js"]
    for name in expected:
        if not (stage / name).is_file():
            raise SystemExit(f"missing package input: {name}")
    header = (stage / "include/nikki/version.h").read_text()
    if f'#define NIKKI_RELEASE_VERSION_STRING "{args.version}"' not in header:
        raise SystemExit("installed driver version does not match release version")
    libraries = list(stage.glob("lib/nikki/*.a"))
    if not libraries or not any("SPIRV-Tools" in path.name for path in libraries) or not any("tint" in path.name for path in libraries):
        raise SystemExit("missing Tint/SPIRV-Tools link closure")
    for path in [stage / "lib/libnikki.a", *libraries]:
        with path.open("rb") as stream:
            if stream.read(8) != b"!<arch>\n":
                raise SystemExit(f"not a normal self-contained static archive: {path}")
    for path in stage.glob("lib/cmake/nikki/*.cmake"):
        if str(root) in path.read_text() or str(stage) in path.read_text():
            raise SystemExit(f"non-relocatable package CMake file: {path}")
    def git_revision(directory):
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    metadata = {"release-version.txt": args.version,
                "source-revision.txt": git_revision(root),
                "dawn-revision.txt": git_revision(root / "third_party/dawn"),
                "emsdk-version.txt": subprocess.check_output(["emcc", "-dumpversion"], text=True).strip()}
    if metadata["dawn-revision.txt"] != versions["NIKKI_DAWN_COMMIT"]:
        raise SystemExit("Dawn revision does not match the pinned dependency")
    if metadata["emsdk-version.txt"] != versions["NIKKI_EMSDK_VERSION"]:
        raise SystemExit("Emscripten version does not match the pinned toolchain")
    for filename, content in metadata.items():
        (stage / filename).write_text(content + "\n")
    (stage / "README.txt").write_text(
        "Prebuilt wasm32 Emscripten SDK, not a JavaScript module.\n"
        "With emsdk 5.0.6 use find_package(nikki CONFIG REQUIRED), then link nikki::static.\n"
        "Set nikki_DIR to this directory/lib/cmake/nikki. Tint/SPIRV-Tools and Vulkan headers are included.\n"
        "The pinned Emdawnwebgpu port downloads its runtime on first use; no Dawn source is needed.\n"
        "Browser module exports and application flags remain the consumer's responsibility.\n")
    name = f"nikki-{args.version}-wasm32-emscripten"
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / (name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(stage, arcname=name)
    print(f"{archive}: {archive.stat().st_size:,} bytes; {len(libraries)} dependency archives")


if __name__ == "__main__":
    main()
