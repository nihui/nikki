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
    const adapter_device_names = [];
    const device_name_marker = "[nikki-test] physical-device-name: ";
    let physical_device_name = null;
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
                    // Match the public Vulkan name, including its 255-byte
                    // limit, rather than accepting the bridge's silent fallback.
                    const device_name = "nikki: "
                        + (info.device || info.architecture || "WebGPU");
                    adapter_device_names.push(new TextDecoder().decode(new TextEncoder().encode(device_name).slice(0, 255)));
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
                if (event.data.name === "test_nikki_compute" && message.startsWith(device_name_marker)) {
                    physical_device_name = message.slice(device_name_marker.length);
                }
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
        // This harness also runs ncnn tests, which do not emit this marker.
        if (event.data.name === "test_nikki_compute") {
            if (physical_device_name === null) {
                validation_errors.push("C++ compute test did not report its Vulkan physical device name");
            } else if (!adapter_device_names.includes(physical_device_name)) {
                validation_errors.push(`Vulkan device name ${JSON.stringify(physical_device_name)} does not match GPUAdapter.info: ${JSON.stringify(adapter_device_names)}`);
            }
        }

        send_result({
            schema: 1,
            name: event.data.name,
            status: return_code === 0 && validation_errors.length === 0 ? "pass" : "fail",
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
