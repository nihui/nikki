// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "datareader.h"
#include "net.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const char tiny_model_param[] = "7767517\n"
                                       "5 5\n"
                                       "Input data 0 1 data 0=8 1=8 2=3\n"
                                       "Convolution conv 1 1 data conv 0=4 1=3 2=1 3=1 4=1 5=1 6=108\n"
                                       "ReLU relu 1 1 conv relu 0=0\n"
                                       "Pooling pool 1 1 relu pool 0=1 1=2 2=2 3=0 4=0\n"
                                       "Softmax prob 1 1 pool prob 0=0\n";

static std::vector<unsigned char> make_model_data()
{
    const size_t weight_count = 108;
    const size_t bias_count = 4;
    std::vector<unsigned char> data(4 + (weight_count + bias_count) * sizeof(float));
    memset(data.data(), 0, 4); // raw-fp32 weight tag
    float* values = reinterpret_cast<float*>(data.data() + 4);
    for (size_t i = 0; i < weight_count; i++)
        values[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.015625);
    for (size_t i = 0; i < bias_count; i++)
        values[weight_count + i] = static_cast<float>((static_cast<int>(i) - 2) * 0.03125);
    return data;
}

static void set_baseline_options(ncnn::Net& net, bool use_vulkan)
{
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_packed = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_arithmetic = false;
    net.opt.use_shader_local_memory = false;
    net.opt.use_subgroup_ops = false;
    net.opt.use_packing_layout = false;
}

static int run_model(ncnn::Net& net, const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("data", input) != 0)
        return -1;
    return extractor.extract("prob", output);
}

static int load_model(ncnn::Net& net, const std::vector<unsigned char>& model_data)
{
    const unsigned char* model_ptr = model_data.data();
    ncnn::DataReaderFromMemory data_reader(model_ptr);
    return net.load_model(data_reader);
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 77;

    const std::vector<unsigned char> model_data = make_model_data();
    ncnn::Net cpu;
    ncnn::Net gpu;
    set_baseline_options(cpu, false);
    set_baseline_options(gpu, true);
    if (cpu.load_param_mem(tiny_model_param) != 0)
    {
        fprintf(stderr, "failed to load the CPU model parameters\n");
        return 1;
    }
    if (gpu.load_param_mem(tiny_model_param) != 0)
    {
        fprintf(stderr, "failed to load the Vulkan model parameters\n");
        return 1;
    }
    if (load_model(cpu, model_data) != 0)
    {
        fprintf(stderr, "failed to load the CPU model weights\n");
        return 1;
    }
    if (load_model(gpu, model_data) != 0)
    {
        fprintf(stderr, "failed to load the Vulkan model weights\n");
        return 1;
    }

    ncnn::Mat input(8, 8, 3);
    for (size_t i = 0; i < input.total(); i++)
        static_cast<float*>(input.data)[i] = static_cast<float>((static_cast<int>(i % 23) - 11) * 0.0625);

    ncnn::Mat expected;
    ncnn::Mat actual;
    if (run_model(cpu, input, expected) != 0)
    {
        fprintf(stderr, "failed to run the CPU model\n");
        return 1;
    }
    if (run_model(gpu, input, actual) != 0)
    {
        fprintf(stderr, "failed to run the Vulkan model\n");
        return 1;
    }
    if (expected.dims != actual.dims || expected.total() != actual.total())
    {
        fprintf(stderr, "tiny model output shape mismatch\n");
        return 1;
    }

    const float* expected_values = expected;
    const float* actual_values = actual;
    for (size_t i = 0; i < expected.total(); i++)
    {
        const float difference = std::fabs(expected_values[i] - actual_values[i]);
        if (difference > 1e-4f)
        {
            fprintf(stderr, "tiny model output mismatch at %zu: cpu=%g nikki=%g\n",
                    i, expected_values[i], actual_values[i]);
            return 1;
        }
    }
    return 0;
}
