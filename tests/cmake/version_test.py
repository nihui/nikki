#!/usr/bin/env python3
# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0
"""Check date validation without requiring Dawn or an Emscripten build."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temporary:
        script = Path(temporary) / "version.cmake"
        script.write_text(f'include("{root}/cmake/nikki_version.cmake")\n'
                          'message(STATUS "encoded=${NIKKI_DRIVER_VERSION_MAJOR}.${NIKKI_DRIVER_VERSION_MINOR}.${NIKKI_DRIVER_VERSION_PATCH}")\n')
        cases = {"20261024": "26.10.24", "20000229": "0.2.29",
                 "20240229": "24.2.29", "20991231": "99.12.31",
                 "20230229": None, "20260230": None, "20260431": None,
                 "20260001": None, "20261301": None, "20261000": None,
                 "19991231": None, "21000101": None, "2026-10-24": None,
                 "20261024junk": None, "": None}
        for date, encoded in cases.items():
            result = subprocess.run([args.cmake, f"-DNIKKI_RELEASE_VERSION={date}", "-P", str(script)],
                                    text=True, capture_output=True)
            if encoded is None:
                assert result.returncode != 0, f"accepted invalid date {date}"
            else:
                assert result.returncode == 0, result.stderr
                assert f"encoded={encoded}" in result.stdout, result.stdout
        print(f"Validated {len(cases)} release dates")


if __name__ == "__main__":
    main()
