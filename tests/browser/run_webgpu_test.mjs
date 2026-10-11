// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import process from "node:process";

import puppeteer from "puppeteer-core";

const test_timeout_ms = 14400000;

function parse_arguments(argv) {
    const adapter = process.env.NIKKI_WEBGPU_TEST_ADAPTER || process.env.NIKKI_NCNN_TEST_ADAPTER || "";
    const options = {
        chrome: process.env.NIKKI_WEBGPU_TEST_CHROME || process.env.NIKKI_NCNN_TEST_CHROME || "",
        expected_adapter: adapter === "swiftshader" ? "swiftshader" : "",
        expected_version: process.env.NIKKI_WEBGPU_TEST_CHROME_VERSION || process.env.NIKKI_NCNN_TEST_CHROME_VERSION || "",
        hardware: adapter === "hardware",
        module: "",
        no_sandbox: adapter === "swiftshader",
        page: "",
        require_webgpu: false,
        swiftshader: adapter === "swiftshader",
        test_name: "",
    };

    for (let i = 0; i < argv.length; i++) {
        const argument = argv[i];
        if (!argument.startsWith("--")) {
            if (options.module) throw new Error(`unexpected positional argument ${argument}`);
            options.module = argument;
            continue;
        }
        if (argument === "--swiftshader") {
            options.swiftshader = true;
            continue;
        }
        if (argument === "--no-sandbox") {
            options.no_sandbox = true;
            continue;
        }
        if (argument === "--hardware") {
            options.hardware = true;
            continue;
        }
        if (argument === "--require-webgpu") {
            options.require_webgpu = true;
            continue;
        }
        const value = argv[++i];
        if (value === undefined) {
            throw new Error(`missing value after ${argument}`);
        }

        if (argument === "--chrome") options.chrome = value;
        else if (argument === "--expected-adapter") options.expected_adapter = value;
        else if (argument === "--expected-version") options.expected_version = value;
        else if (argument === "--module") options.module = value;
        else if (argument === "--page") options.page = value;
        else if (argument === "--test-name") options.test_name = value;
        else throw new Error(`unknown argument ${argument}`);
    }

    if (options.module) {
        const module_path = resolve(options.module);
        if (!options.page) options.page = resolve(dirname(module_path), "webgpu_test.html");
        options.module = module_path.split(sep).at(-1);
        if (!options.test_name) options.test_name = options.module.replace(/\.(?:m?js)$/, "");
        if (options.test_name === "test_nikki_compute" || options.test_name === "test_command" || options.test_name === "test_squeezenet" || options.test_name === "nikki_ncnn_model_test") {
            options.require_webgpu = true;
        }
    }

    for (const name of ["chrome", "module", "page", "test_name"]) {
        if (!options[name]) throw new Error(`--${name.replaceAll("_", "-")} is required`);
    }
    if (options.swiftshader && options.hardware) {
        throw new Error("--swiftshader and --hardware are mutually exclusive");
    }

    return options;
}

function content_type(path) {
    const types = {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".wasm": "application/wasm",
    };
    return types[extname(path)] || "application/octet-stream";
}

async function start_server(page_path) {
    const root = dirname(page_path);
    const root_prefix = root.endsWith(sep) ? root : root + sep;

    const server = createServer((request, response) => {
        const request_url = new URL(request.url, "http://127.0.0.1");
        const relative_path = decodeURIComponent(request_url.pathname).replace(/^\/+/, "");
        const requested_path = resolve(root, relative_path || page_path.split(sep).at(-1));

        if (requested_path !== root && !requested_path.startsWith(root_prefix)) {
            response.writeHead(403);
            response.end("forbidden\n");
            return;
        }

        if (!existsSync(requested_path)) {
            response.writeHead(404);
            response.end("not found\n");
            return;
        }
        const requested_stat = statSync(requested_path);
        if (!requested_stat.isFile()) {
            response.writeHead(404);
            response.end("not found\n");
            return;
        }

        response.writeHead(200, {
            "Cache-Control": "no-store",
            "Content-Length": requested_stat.size,
            "Content-Type": content_type(requested_path),
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
        });
        createReadStream(requested_path).pipe(response);
    });

    await new Promise((resolve_listen, reject_listen) => {
        server.once("error", reject_listen);
        server.listen(0, "127.0.0.1", resolve_listen);
    });

    return server;
}

function validate_result(payload, test_name, require_webgpu) {
    if (!payload || payload.schema !== 1) throw new Error("invalid or missing result schema");
    if (payload.name !== test_name) throw new Error(`result test name mismatch: ${payload.name}`);
    if (typeof payload.webgpu_used !== "boolean") throw new Error("webgpu_used is not a boolean");
    if (payload.status !== "pass") {
        throw new Error(`test returned status ${payload.status}: ${JSON.stringify(payload)}`);
    }
    if (!Array.isArray(payload.validation_errors)) throw new Error("validation_errors is not an array");
    if (payload.validation_errors.length !== 0) {
        throw new Error(`WebGPU validation errors: ${JSON.stringify(payload.validation_errors)}`);
    }
    if (payload.device_lost !== false) throw new Error("unexpected WebGPU device lost");
    if (require_webgpu && !payload.webgpu_used) {
        throw new Error("test did not use WebGPU");
    }
    if (payload.webgpu_used && payload.event_loop_ticks < 1) {
        throw new Error("Worker event loop did not advance while C++ waited");
    }
}

async function main() {
    const options = parse_arguments(process.argv.slice(2));
    const page_path = resolve(options.page);
    const server = await start_server(page_path);
    const address = server.address();
    const browser_arguments = [
        "--headless=new",
        "--enable-gpu",
        "--disable-dev-shm-usage",
    ];
    if (options.swiftshader) {
        browser_arguments.push(
            "--enable-unsafe-webgpu",
            "--use-webgpu-adapter=swiftshader",
            "--use-gl=angle",
            "--use-angle=swiftshader",
        );
    }
    if (options.hardware) {
        browser_arguments.push("--use-angle=vulkan");
    }
    if (options.no_sandbox) {
        browser_arguments.push("--no-sandbox");
    }

    let browser;
    const browser_errors = [];
    const browser_logs = [];
    const result_sentinels = [];
    try {
        browser = await puppeteer.launch({
            args: browser_arguments,
            executablePath: options.chrome,
            headless: true,
            protocolTimeout: test_timeout_ms,
        });

        const browser_version = await browser.version();
        if (options.expected_version && !browser_version.endsWith(`/${options.expected_version}`)) {
            throw new Error(`Chrome version mismatch: expected ${options.expected_version}, got ${browser_version}`);
        }

        const page = await browser.newPage();
        page.on("console", (message) => {
            const text = message.text();
            browser_logs.push(text);
            if (message.type() === "error") browser_errors.push(`console: ${text}`);
            else process.stderr.write(`[browser ${message.type()}] ${text}\n`);
        });
        page.on("pageerror", (error) => browser_errors.push(`pageerror: ${error.stack || error.message}`));
        page.on("requestfailed", (request) => {
            browser_errors.push(`requestfailed: ${request.url()} ${request.failure()?.errorText || ""}`);
        });

        const page_name = page_path.split(sep).at(-1);
        const test_url = new URL(`http://127.0.0.1:${address.port}/${encodeURIComponent(page_name)}`);
        test_url.searchParams.set("name", options.test_name);
        if (options.module) {
            const module_specifier = options.module.startsWith(".") || options.module.includes(":")
                ? options.module
                : `./${options.module}`;
            test_url.searchParams.set("module", module_specifier);
        }
        await page.goto(test_url.href, { waitUntil: "load", timeout: test_timeout_ms });

        const payload = await page.evaluate(
            (timeout_ms) => Promise.race([
                window.__nikkiWebGpuDone,
                new Promise((_, reject) => setTimeout(() => reject(new Error("browser test timeout")), timeout_ms)),
            ]),
            test_timeout_ms,
        );
        result_sentinels.push(payload);

        if (result_sentinels.length !== 1) throw new Error(`expected one result, got ${result_sentinels.length}`);
        if (payload.status === "pass") {
            for (let i = browser_errors.length - 1; i >= 0; i--) {
                // Chrome may report an Emscripten preload stream as aborted after
                // the worker has consumed it. A passing C++ test proves that the
                // model data was loaded; all other request failures remain fatal.
                if (/^requestfailed: .*\.data net::ERR_ABORTED$/.test(browser_errors[i])) {
                    browser_errors.splice(i, 1);
                }
            }
        }
        if (options.expected_adapter && payload.webgpu_used) {
            const adapter = options.expected_adapter.toLowerCase();
            const payload_adapter_matches =
                typeof payload.adapter_info === "string" && payload.adapter_info.toLowerCase().includes(adapter);
            const matching_logs = browser_logs.filter(
                (log) => (log.startsWith("[nikki]") || /^\[0 .+\]  queueC=/.test(log))
                    && log.toLowerCase().includes(adapter),
            );
            if (!payload_adapter_matches && matching_logs.length === 0) {
                throw new Error(`expected a ${options.expected_adapter} adapter log, got none`);
            }
        }
        if (browser_errors.length !== 0) throw new Error(browser_errors.join("\n"));
        validate_result(payload, options.test_name, options.require_webgpu);

        process.stdout.write(`NIKKI_TEST_RESULT:${JSON.stringify(payload)}\n`);
    } finally {
        if (browser) await browser.close();
        await new Promise((resolve_close) => server.close(resolve_close));
    }
}

main().catch((error) => {
    process.stderr.write(`run_webgpu_test.mjs: ${error.stack || error.message}\n`);
    process.exitCode = 1;
});
