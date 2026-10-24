// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

let wasm_call_pending = false;
let result_sent = false;

function send_result(payload) {
    if (result_sent) throw new Error("attempted to send more than one WebGPU result");
    result_sent = true;
    self.postMessage({ type: "nikki-webgpu-result", payload });
}

function send_log(stream, message) {
    self.postMessage({ type: "nikki-webgpu-log", stream, message });
}

self.addEventListener("message", async (event) => {
    if (event.data.type !== "nikki-webgpu-run") return;

    if (wasm_call_pending) {
        self.postMessage({ type: "nikki-webgpu-rejected", name: event.data.name });
        return;
    }

    wasm_call_pending = true;
    const start = performance.now();
    let event_loop_ticks = 0;
    let webgpu_used = false;
    let adapter_info = "";
    let device_lost = false;
    const validation_errors = [];
    const interval = setInterval(() => event_loop_ticks++, 10);

    try {
        if (typeof WorkerGlobalScope === "undefined" || !(self instanceof WorkerGlobalScope)) {
            throw new Error("nikki WebGPU tests must run in a Dedicated Worker");
        }
        if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");

        const request_adapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
        navigator.gpu.requestAdapter = (...arguments_) => {
            webgpu_used = true;
            return request_adapter(...arguments_).then((adapter) => {
                if (adapter) {
                    const info = adapter.info || {};
                    adapter_info = [info.vendor, info.architecture, info.device, info.description]
                        .filter((value) => value)
                        .join(" ");
                }
                return adapter;
            });
        };

        const imported_module = await import(event.data.module_url);
        const create_module = imported_module.default;
        if (typeof create_module !== "function") throw new Error("Emscripten module has no default factory export");

        const module = await create_module({
            noInitialRun: true,
            print: (message) => {
                if (message.startsWith("[nikki]") || /^\[0 .+\]  queueC=/.test(message)) webgpu_used = true;
                send_log("stdout", message);
            },
            printErr: (message) => {
                const nikki_validation_error = message.startsWith("[nikki]")
                    && (message.includes(" uncaptured error ")
                        || message.includes(" failed status="));
                const nikki_device_lost = message.startsWith("[nikki]")
                    && message.includes(" lost reason=");
                const webgpu_error = message.startsWith("WebGPU ")
                    || message.startsWith("no such shader module ")
                    || nikki_validation_error
                    || nikki_device_lost;
                if (webgpu_error) validation_errors.push(message);
                if (nikki_device_lost) device_lost = true;
                if (message.startsWith("[nikki]") || /^\[0 .+\]  queueC=/.test(message)) webgpu_used = true;
                send_log(webgpu_error ? "stderr" : "stdout", message);
            },
        });
        const return_code = await module.ccall("main", "number", [], [], { async: true });

        if (return_code !== 0) validation_errors.push(`C++ test returned ${return_code}`);

        send_result({
            schema: 1,
            name: event.data.name,
            status: return_code === 0 ? "pass" : "fail",
            assertions: 1,
            validation_errors,
            device_lost,
            duration_ms: Math.round(performance.now() - start),
            event_loop_ticks,
            webgpu_used,
            adapter_info,
        });
    } catch (error) {
        validation_errors.push(String(error.stack || error.message || error));
        send_result({
            schema: 1,
            name: event.data.name,
            status: "fail",
            assertions: 0,
            validation_errors,
            device_lost,
            duration_ms: Math.round(performance.now() - start),
            event_loop_ticks,
            webgpu_used,
            adapter_info,
        });
    } finally {
        clearInterval(interval);
        wasm_call_pending = false;
    }
});
