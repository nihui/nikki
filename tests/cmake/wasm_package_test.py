#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0
"""Check the bundled Emdawn package and its agreement with the source-build port."""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bootstrap_dawn import load_versions
from package_wasm import validate_emdawnwebgpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    versions = load_versions(ROOT / "CMakeLists.txt")
    remote = runpy.run_path(str(ROOT / "ports" / versions["NIKKI_EMDAWNWEBGPU_PORT_FILENAME"]))
    assert remote["EXTERNAL_PORT"] == versions["NIKKI_EMDAWNWEBGPU_PACKAGE_URL"]
    assert remote["SHA512"] == versions["NIKKI_EMDAWNWEBGPU_PACKAGE_SHA512"]
    validate_emdawnwebgpu(args.stage)
    relative = Path("share/nikki/ports/emdawnwebgpu_pkg")
    missing_inputs = ["emdawnwebgpu.port.py", "webgpu/src/library_webgpu.js",
                      "webgpu/src/library_webgpu_generated_sig_info.js",
                      "webgpu/include/webgpu/webgpu.h", "webgpu/src/LICENSE"]
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        shutil.copytree(args.stage / relative, stage / relative)
        for name in missing_inputs:
            path = stage / relative / name
            contents = path.read_bytes()
            path.unlink()
            try:
                validate_emdawnwebgpu(stage)
            except SystemExit as error:
                assert str(error) == f"missing Emdawnwebgpu package input: {name}", error
            else:
                raise AssertionError(f"accepted package without {name}")
            path.write_bytes(contents)
    print(f"Validated Emdawn pins, bundled runtime and {len(missing_inputs)} missing-input cases")


if __name__ == "__main__":
    main()
