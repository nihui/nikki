# Contributing

Keep behavior changes separate from mechanical file moves. The modular files
under `src/` are authoritative; generate the single-source distribution in the
build tree when validating a modular change.

Use the checked-in `.clang-format`, which matches ncnn's style and deliberately
sets `ColumnLimit: 0`; do not manually wrap code to a fixed display width.
Do not format generated files, Dawn, port snapshots, or other third-party sources. New
project-owned source files need the `Copyright 2026 nihui` notice and
`SPDX-License-Identifier: Apache-2.0` header.

Before submitting a change, build both modular and generated forms and run the
native tests. Translator changes should include a
focused fixture; asynchronous runtime changes should exercise submission,
fence, and readback lifetime. Browser-specific changes also require the
Emscripten/browser lane described in `docs/testing.md`.
