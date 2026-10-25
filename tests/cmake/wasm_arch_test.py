#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0
"""Check installed SDK architecture metadata and reject mismatched consumers."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--arch", choices=("wasm32", "wasm64"), required=True)
    args = parser.parse_args()
    bits = args.arch[4:]
    other = "32" if bits == "64" else "64"
    package = args.stage.resolve() / "lib/cmake/nikki"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.22)\n'
            'project(architecture_check LANGUAGES C CXX)\n'
            'find_package(nikki CONFIG REQUIRED)\n'
            f'if(NOT nikki_WASM_ARCH STREQUAL "{args.arch}" OR NOT nikki_WASM_POINTER_SIZE EQUAL {int(bits) // 8})\n'
            '  message(FATAL_ERROR "Incorrect SDK architecture metadata")\n'
            'endif()\n')
        cases = [(bits, bits, True, "matching"), (other, other, False, "mismatch"),
                 (bits, other, False, "mixed"), (bits, other, False, "matching"),
                 (bits, bits, True, "matching")]
        for c_bits, cxx_bits, accepted, build in cases:
            command = ["emcmake", "cmake", "-S", str(source), "-B", str(root / build),
                       "-G", "Ninja", f"-Dnikki_DIR={package}",
                       f"-DCMAKE_C_FLAGS=-m{c_bits}", f"-DCMAKE_CXX_FLAGS=-m{cxx_bits}"]
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if (result.returncode == 0) != accepted:
                raise AssertionError(f"unexpected architecture configure result: {command}\n{result.stdout}")
            diagnostic = " ".join(result.stdout.split())
            version_mismatch = f"({bits}bit)" in diagnostic and (
                "not compatible" in diagnostic or
                'Could not find a configuration file for package "nikki" that is compatible' in diagnostic)
            if not accepted and f"requires {args.arch}" not in result.stdout and not version_mismatch:
                raise AssertionError(f"configure failed for an unexpected reason:\n{result.stdout}")
        config = (package / "nikkiConfig.cmake").read_text()
        version = re.search(r'^set\(nikki_RELEASE_VERSION "([^\"]*)"\)$', config, re.MULTILINE)[1]
        script = Path(__file__).resolve().parents[2] / "tools/package_wasm.py"
        result = subprocess.run([sys.executable, str(script), "--stage", str(args.stage),
                                 "--output", str(root / "release"), "--version", version,
                                 "--arch", "wasm" + other],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0 or "installed package architecture does not match release architecture" not in result.stdout:
            raise AssertionError(f"wrong-architecture packaging failed for an unexpected reason:\n{result.stdout}")
        assert not (root / "release").exists(), "mislabeled SDK must not be published"
    print(f"Validated {args.arch} SDK metadata, matching consumer, three ABI mismatches and mislabeled packaging")


if __name__ == "__main__":
    main()
