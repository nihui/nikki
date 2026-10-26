# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("bootstrap_dawn", ROOT / "tools/bootstrap_dawn.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class DawnPatchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dawn = self.root / "third_party/dawn"
        self.source = self.dawn / BOOTSTRAP.DAWN_PATCH_SOURCE
        self.source.parent.mkdir(parents=True)
        self.original = subprocess.check_output([
            "git", "-C", str(ROOT / "third_party/dawn"), "show",
            BOOTSTRAP.DAWN_PATCH_COMMIT + ":" + BOOTSTRAP.DAWN_PATCH_SOURCE,
        ])
        self.source.write_bytes(self.original)
        self.patch = self.root / "patches/dawn-wgsl-matrix-i8.patch"
        self.patch.parent.mkdir()
        shutil.copyfile(ROOT / "patches/dawn-wgsl-matrix-i8.patch", self.patch)

    def apply(self):
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", return_value=BOOTSTRAP.DAWN_PATCH_COMMIT):
            BOOTSTRAP.apply_dawn_patch(self.dawn, self.patch, BOOTSTRAP.DAWN_PATCH_COMMIT)

    def test_apply_is_idempotent(self):
        self.apply()
        expected = self.source.read_bytes()
        self.assertEqual(hashlib.sha256(expected).hexdigest(), BOOTSTRAP.DAWN_PATCH_APPLIED_SHA256)
        self.apply()
        self.assertEqual(self.source.read_bytes(), expected)

    def test_refuse_changed_patch_after_application(self):
        self.apply()
        expected = self.source.read_bytes()
        self.patch.write_text(self.patch.read_text().replace("b.ty.i8()", "b.ty.i32()"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.apply()
        self.assertEqual(self.source.read_bytes(), expected)

    def test_refuse_unexpected_source(self):
        original = self.original + b"// local edit\n"
        self.source.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, "unexpected contents"):
            self.apply()
        self.assertEqual(self.source.read_bytes(), original)

    def test_refuse_wrong_revision(self):
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", return_value="0" * 40):
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                BOOTSTRAP.apply_dawn_patch(self.dawn, self.patch, BOOTSTRAP.DAWN_PATCH_COMMIT)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_dependency_revision_is_verified(self):
        scripts = self.dawn / "tools"
        scripts.mkdir()
        (scripts / "fetch_dawn_dependencies.py").write_text("required_submodules = ['third_party/test']")
        dependency = self.dawn / "third_party/test"
        dependency.mkdir(parents=True)
        (dependency / ".git").mkdir()
        entry = "160000 commit " + "a" * 40 + "\tthird_party/test\n"
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", side_effect=[entry, "b" * 40]):
            with self.assertRaisesRegex(RuntimeError, "dependency revision mismatch"):
                BOOTSTRAP.verify_dawn_dependencies(self.dawn)
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", side_effect=[entry, "a" * 40]):
            BOOTSTRAP.verify_dawn_dependencies(self.dawn)
        (dependency / ".git").rmdir()
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", return_value=entry):
            with self.assertRaisesRegex(RuntimeError, "dependency is missing"):
                BOOTSTRAP.verify_dawn_dependencies(self.dawn)

    def test_read_only_check_validates_patch_and_stamp(self):
        (self.dawn / "DEPS").write_text("test dependencies")
        versions = {"NIKKI_DAWN_COMMIT": BOOTSTRAP.DAWN_PATCH_COMMIT,
                    "NIKKI_DAWN_DEPS_SHA256": BOOTSTRAP.sha256_file(self.dawn / "DEPS")}
        self.apply()
        stamp = {"dawn_commit": BOOTSTRAP.DAWN_PATCH_COMMIT,
                 "deps_sha256": versions["NIKKI_DAWN_DEPS_SHA256"],
                 "matrix_writer_patch_sha256": BOOTSTRAP.sha256_file(self.patch), "schema": 2}
        stamp_path = self.dawn / "out/.nikki-deps-stamp"
        stamp_path.parent.mkdir()
        BOOTSTRAP.write_stamp(stamp_path, stamp)
        with mock.patch.object(BOOTSTRAP.subprocess, "check_output", return_value=BOOTSTRAP.DAWN_PATCH_COMMIT), \
             mock.patch.object(BOOTSTRAP, "verify_dawn_dependencies"):
            self.assertEqual(BOOTSTRAP.verify_dawn_state(self.dawn, self.root, versions),
                             stamp["matrix_writer_patch_sha256"])
            stamp_path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "stamp does not match"):
                BOOTSTRAP.verify_dawn_state(self.dawn, self.root, versions)
            BOOTSTRAP.write_stamp(stamp_path, stamp)
            self.source.write_bytes(self.original)
            with self.assertRaisesRegex(RuntimeError, "patch is missing"):
                BOOTSTRAP.verify_dawn_state(self.dawn, self.root, versions)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(BOOTSTRAP.load_stamp(stamp_path), stamp)

    def test_cache_hit_restores_missing_patch(self):
        # A dependency stamp must not make an unpatched checkout silently usable.
        (self.root / "tools").mkdir()
        (self.root / "ports").mkdir()
        (self.dawn / "CMakeLists.txt").write_text("")
        (self.dawn / "DEPS").write_text("test dependencies")
        (self.root / "ports/test.py").write_text("test port")
        deps_hash = BOOTSTRAP.sha256_file(self.dawn / "DEPS")
        port_hash = BOOTSTRAP.sha256_file(self.root / "ports/test.py")
        versions = {
            "NIKKI_DAWN_COMMIT": BOOTSTRAP.DAWN_PATCH_COMMIT,
            "NIKKI_DAWN_DEPS_SHA256": deps_hash,
            "NIKKI_EMDAWNWEBGPU_PORT_FILENAME": "test.py",
            "NIKKI_EMDAWNWEBGPU_PORT_SHA256": port_hash,
            "NIKKI_EMSDK_VERSION": "6.0.12",
            "NIKKI_ASYNCIFY_STACK_SIZE": "65536",
            "NIKKI_WASM_STACK_SIZE": "1048576",
        }
        (self.root / "CMakeLists.txt").write_text("\n".join('set(%s "%s")' % item for item in versions.items()))
        stamp = {"dawn_commit": BOOTSTRAP.DAWN_PATCH_COMMIT,
                 "deps_sha256": deps_hash,
                 "matrix_writer_patch_sha256": BOOTSTRAP.sha256_file(self.patch), "schema": 2}
        stamp_path = self.dawn / "out/.nikki-deps-stamp"
        stamp_path.parent.mkdir()
        BOOTSTRAP.write_stamp(stamp_path, stamp)
        with mock.patch.object(BOOTSTRAP, "__file__", str(self.root / "tools/bootstrap_dawn.py")), \
             mock.patch.object(BOOTSTRAP.subprocess, "check_output", return_value=BOOTSTRAP.DAWN_PATCH_COMMIT), \
             mock.patch.object(sys, "argv", ["bootstrap_dawn.py"]), \
             mock.patch.object(BOOTSTRAP, "verify_dawn_dependencies") as verify:
            self.addCleanup(lambda: self.assertEqual(verify.call_count, 2))
            self.assertEqual(BOOTSTRAP.main(), 0)
            self.source.write_bytes(self.original)
            self.assertEqual(BOOTSTRAP.main(), 0)
        self.assertEqual(BOOTSTRAP.sha256_file(self.source), BOOTSTRAP.DAWN_PATCH_APPLIED_SHA256)
        self.assertEqual(BOOTSTRAP.load_stamp(stamp_path), stamp)


if __name__ == "__main__":
    unittest.main()
