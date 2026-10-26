// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

// Run the actual EM_JS bodies without a GPU or an Emscripten build. Small
// addresses use real typed arrays; sparse cases isolate address arithmetic.
// A real Memory64 heap also verifies a few writes above 4 GiB (not a full
// >4 GiB workload). CI requires that case with --require-memory64.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const arguments_ = process.argv.slice(2);
const source_paths = arguments_.filter((argument) => argument !== "--require-memory64");
assert.ok(source_paths.length <= 1, "usage: node wasm_pointer_bridge_test.mjs [source.cpp] [--require-memory64]");
const source = readFileSync(source_paths[0] || new URL("../../src/backend/device.cpp", import.meta.url), "utf8");
const feature_name = "chromium-experimental-subgroup-matrix";
const feature_value = 0x00050034;

function load_bridge(name, globals) {
    const matches = [...source.matchAll(new RegExp(`EM_JS\\(int, ${name},\\s*\\(([^)]*)\\),\\s*\\{([\\s\\S]*?)\\n\\}\\);`, "g"))];
    assert.equal(matches.length, 1, `expected one real ${name} EM_JS body`);
    const parameters = matches[0][1].split(",").map((parameter) => parameter.trim().match(/\w+$/)[0]);
    return runInNewContext(`(function(${parameters.join(",")}) {${matches[0][2]}\n})`, globals);
}

function make_fixture(base = 0, memory64 = false, backing_buffer = null) {
    const byte_length = base + 4096;
    const writes = new Map();
    function sparse_heap(element_size) {
        return new Proxy({ length: byte_length / element_size }, {
            get(target, key) {
                return key === "length" ? target.length : writes.get(`${element_size}:${key}`) || 0;
            },
            set(target, key, value) {
                const index = Number(key);
                assert.ok(Number.isSafeInteger(index) && index >= 0 && index < target.length, `invalid heap index ${key}`);
                writes.set(`${element_size}:${key}`, value);
                return true;
            },
        });
    }
    const buffer = backing_buffer || (base === 0 ? new ArrayBuffer(byte_length) : null);
    const HEAPU8 = buffer ? new Uint8Array(buffer) : sparse_heap(1);
    const HEAPU32 = buffer ? new Uint32Array(buffer) : sparse_heap(4);
    const pointer = (offset) => memory64 ? BigInt(base + offset) : (base + offset) | 0;
    const adapter = {
        info: {
            vendor: "vendor", architecture: "architecture", device: "device", description: "description",
            subgroupMinSize: 4, subgroupMaxSize: 32,
            subgroupMatrixConfigs: [
                { componentType: "f16", resultComponentType: "f32", M: 8, N: 16, K: 32 },
                { componentType: "i8", resultComponentType: "i32", M: 16, N: 8, K: 4 },
            ],
        },
        features: new Set([feature_name]),
    };
    const objects = { [base + 16]: adapter };
    const WebGPU = {
        // The upgraded Emdawn runtime supplies the matrix feature enum itself.
        // Reject writes to catch any obsolete bridge-side injection.
        FeatureName: new Proxy({ [feature_value]: feature_name }, {
            set() { throw new Error("bridge must not mutate upstream feature enums"); },
        }),
        // Match the pinned Emdawn implementation: wasm32 unsigned conversion,
        // but no conversion of the wasm64 pointer used as an object key.
        getJsObject(handle) {
            assert.equal(typeof handle, memory64 ? "bigint" : "number");
            if (!memory64) handle >>>= 0;
            return objects[handle];
        },
    };
    const globals = {
        WebGPU, HEAPU8, HEAPU32,
        stringToUTF8(value, address, capacity) {
            assert.equal(typeof address, "number", "stringToUTF8 needs a Number address");
            const bytes = Buffer.from(value, "utf8");
            const count = Math.min(bytes.length, capacity - 1);
            for (let i = 0; i < count; i++) HEAPU8[address + i] = bytes[i];
            HEAPU8[address + count] = 0;
        },
    };
    const info = load_bridge("nikki_read_adapter_info", globals);
    const matrix = load_bridge("nikki_read_subgroup_matrix_configs", globals);
    const info_arguments = [pointer(16), pointer(128), 64, pointer(256), 64, pointer(384), 64, pointer(512), 64, pointer(640), pointer(644)];
    return {
        adapter, WebGPU, pointer, info_arguments, writes, HEAPU8, HEAPU32,
        info: (...args) => info(...(args.length ? args : info_arguments)),
        matrix: (address = pointer(768), capacity = 2) => matrix(pointer(16), address, capacity),
        string(offset) {
            const bytes = [];
            for (let i = base + offset; HEAPU8[i]; i++) bytes.push(HEAPU8[i]);
            return Buffer.from(bytes).toString("utf8");
        },
        words(offset, count) {
            return Array.from({ length: count }, (_, i) => HEAPU32[(base + offset) / 4 + i]);
        },
        untouched() {
            assert.equal(writes.size, 0);
            if (buffer) assert.ok(HEAPU8.every((value) => value === 0));
        },
    };
}

const pointer_cases = [
    ["wasm32", 0, false],
    ["wasm64 BigInt", 0, true],
    ["wasm64 BigInt above 4 GiB (sparse)", 0x100000000, true],
    ["wasm32 signed i32 above 2 GiB (sparse)", 0x80000000, false],
];
try {
    const memory = new WebAssembly.Memory({ initial: 65537n, maximum: 65537n, address: "i64" });
    pointer_cases.push(["wasm64 BigInt above 4 GiB (real Memory64 heap)", 0x100000000, true, memory.buffer]);
} catch (error) {
    if (arguments_.includes("--require-memory64")) throw error;
    console.log(`SKIP real Memory64 heap: ${error.message}; use Node 24+ and --require-memory64 to require it`);
}
for (const [name, base, memory64, buffer] of pointer_cases) {
    const f = make_fixture(base, memory64, buffer);
    assert.equal(f.info(), 1, name);
    assert.equal(f.string(128), "vendor", name);
    assert.equal(f.string(256), "architecture", name);
    assert.equal(f.string(384), "device", name);
    assert.equal(f.string(512), "description", name);
    assert.deepEqual(f.words(640, 2), [4, 32], name);
    assert.equal(f.matrix(), 2, name);
    assert.deepEqual(f.words(768, 10), [1, 2, 8, 16, 32, 3, 5, 16, 8, 4], name);
    assert.equal(f.WebGPU.FeatureName[feature_value], feature_name);
    console.log(`PASS ${name}`);
}

{
    const f = make_fixture(0, true);
    f.adapter.info = { isFallbackAdapter: true, subgroupMinSize: -1, subgroupMaxSize: Infinity };
    assert.equal(f.info(), 2);
    assert.equal(f.string(128), "");
    assert.deepEqual(f.words(640, 2), [0, 0]);
    assert.equal(f.matrix(), 0);
}
for (const absent of ["info", "features"]) {
    const f = make_fixture(0, true);
    delete f.adapter[absent];
    if (absent === "info") assert.equal(f.info(), 0);
    assert.equal(f.matrix(), absent === "features" ? -1 : 0);
    f.untouched();
}
{
    const f = make_fixture(0, true);
    f.adapter.features.clear();
    assert.equal(f.matrix(), -1);
    assert.equal(f.WebGPU.FeatureName[feature_value], feature_name);
    f.untouched();
}
{
    const f = make_fixture(0, true);
    assert.equal(f.matrix(0n, 0), 2);
    f.untouched();
    assert.equal(f.matrix(f.pointer(768), 1), 2);
    assert.deepEqual(f.words(768, 10), [1, 2, 8, 16, 32, 0, 0, 0, 0, 0]);
    f.adapter.info.subgroupMatrixConfigs = [
        { componentType: "u8", resultComponentType: "u32", M: 2, N: 3, K: 4 },
        { componentType: "unknown", resultComponentType: "unknown", M: 5, N: 6, K: 7 },
    ];
    assert.equal(f.matrix(), 2);
    assert.deepEqual(f.words(768, 10), [4, 6, 2, 3, 4, 0, 0, 5, 6, 7]);
}
for (const address of [-1n, 9007199254740993n, 9007199254740992, Infinity, NaN, 1.5, -0x80000001, -0x100000000, -0x100000001, 4096n, 643n]) {
    const f = make_fixture(0, true);
    const args = [...f.info_arguments];
    args[10] = address; // Last output: an invalid address must not cause partial writes.
    assert.equal(f.info(...args), 0, `reject adapter info address ${address}`);
    f.untouched();
    assert.equal(f.matrix(address), 0, `reject matrix address ${address}`);
    f.untouched();
}
{
    const f = make_fixture(0, true);
    const args = [...f.info_arguments];
    args[1] = 4080n; // A string buffer, not just its first byte, must fit.
    assert.equal(f.info(...args), 0);
    f.untouched();
    assert.equal(f.matrix(4080n), 0);
    f.untouched();
    Object.defineProperty(f.adapter, "info", { get() { throw new Error("unavailable adapter info"); } });
    assert.equal(f.info(), 0);
    assert.equal(f.matrix(), 0);
    f.untouched();
}
console.log("PASS optional fields, fallback, absent feature/info, capacity, component types and rejected addresses");
