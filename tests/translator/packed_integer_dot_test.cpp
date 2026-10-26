// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/translator/translator_stages.h"
#include <spirv-tools/libspirv.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>

namespace {
struct Case
{
    const char* opcode;
    bool unsigned_dot;
    bool mixed;
    bool saturating;
    bool dual = false;
    bool signed_operands = false;
    bool flip_result_signedness = false;
};

std::string fixture(const Case& test)
{
    const bool unsigned_result = test.unsigned_dot != test.flip_result_signedness;
    const char* operand = test.signed_operands ? "%i32" : "%u32";
    const char* operand_ptr = test.signed_operands ? "%iptr" : "%uptr";
    std::ostringstream s;
    s << "OpCapability Shader\nOpCapability DotProduct\nOpCapability DotProductInput4x8BitPacked\n"
      << "OpExtension \"SPV_KHR_integer_dot_product\"\nOpMemoryModel Logical GLSL450\n"
      << "OpEntryPoint GLCompute %main \"main\"\nOpExecutionMode %main LocalSize 1 1 1\n"
      << "OpDecorate %block Block\nOpMemberDecorate %block 0 Offset 0\n"
      << "OpMemberDecorate %block 1 Offset 4\nOpMemberDecorate %block 2 Offset 8\n"
      << "OpDecorate %buffer DescriptorSet 0\nOpDecorate %buffer Binding 0\n"
      << "%void = OpTypeVoid\n%u32 = OpTypeInt 32 0\n%i32 = OpTypeInt 32 1\n"
      << "%function = OpTypeFunction %void\n%zero = OpConstant %u32 0\n"
      << "%one = OpConstant %u32 1\n%two = OpConstant %u32 2\n"
      << "%block = OpTypeStruct " << operand << ' ' << operand << ' ' << (unsigned_result ? "%u32" : "%i32") << "\n"
      << "%blockptr = OpTypePointer StorageBuffer %block\n%uptr = OpTypePointer StorageBuffer %u32\n"
      << "%iptr = OpTypePointer StorageBuffer %i32\n%buffer = OpVariable %blockptr StorageBuffer\n"
      << "%main = OpFunction %void None %function\n%entry = OpLabel\n"
      << "%pa = OpAccessChain " << operand_ptr << " %buffer %zero\n%pb = OpAccessChain " << operand_ptr << " %buffer %one\n"
      << "%pc = OpAccessChain " << (unsigned_result ? "%uptr" : "%iptr") << " %buffer %two\n"
      << "%100 = OpLoad " << operand << " %pa\n%101 = OpLoad " << operand << " %pb\n"
      << "%102 = OpLoad " << (unsigned_result ? "%u32" : "%i32") << " %pc\n"
      << (test.dual ? "%199 = " : "%200 = ") << test.opcode << ' ' << (unsigned_result ? "%u32" : "%i32") << " %100 %101"
      << (test.saturating ? " %102" : "") << " PackedVectorFormat4x8Bit\n";
    if (test.dual)
        s << "%200 = " << test.opcode << ' ' << (unsigned_result ? "%u32" : "%i32")
          << " %100 %101 %199 PackedVectorFormat4x8Bit\n";
    s << "OpStore %pc %200\nOpReturn\nOpFunctionEnd\n";
    return s.str();
}

int64_t dot(uint32_t a, uint32_t b, bool unsigned_dot, bool mixed)
{
    int64_t total = 0;
    for (unsigned lane = 0; lane < 4; lane++)
    {
        int x = (a >> (lane * 8)) & 255, y = (b >> (lane * 8)) & 255;
        if (!unsigned_dot && x >= 128) x -= 256;
        if (!unsigned_dot && !mixed && y >= 128) y -= 256;
        total += x * y;
    }
    return total;
}

// Interpret the actual rewritten straight-line arithmetic, not a duplicated
// saturating-add formula. The two native dot markers model the WGSL builtins;
// all overflow detection, selection, casts and mixed-dot extraction come from
// the production lowering's SPIR-V instructions.
uint32_t evaluate(const std::vector<uint32_t>& words, uint32_t a, uint32_t b, uint32_t acc)
{
    std::map<uint32_t, uint32_t> values;
    std::map<uint32_t, uint32_t> reversals;
    std::map<uint32_t, bool> signed_types;
    for (size_t i = 5; i < words.size(); i += words[i] >> 16)
    {
        const auto op = static_cast<spv::Op>(words[i] & 65535);
        const uint32_t* w = words.data() + i;
        if (op == spv::Op::OpTypeInt) signed_types[w[1]] = w[3] != 0;
        if (op == spv::Op::OpConstant) values[w[2]] = w[3];
        if (op == spv::Op::OpLoad)
            values[w[2]] = w[2] == 100 ? a : w[2] == 101 ? b : acc;
        if ((words[i] >> 16) < 4) continue;
        uint32_t result = 0;
        bool has_result = true;
        switch (op)
        {
        case spv::Op::OpBitcast: case spv::Op::OpCopyObject: result = values[w[3]]; break;
        case spv::Op::OpBitReverse:
            reversals[w[2]] = w[3];
            for (unsigned k = 0; k < 32; k++) result |= ((values[w[3]] >> k) & 1) << (31 - k);
            break;
        case spv::Op::OpBitwiseXor:
            if (reversals.count(w[3]) && reversals.count(reversals[w[3]]))
                result = static_cast<uint32_t>(dot(values[reversals[reversals[w[3]]]], values[w[4]], !signed_types[w[1]], false));
            else result = values[w[3]] ^ values[w[4]];
            break;
        case spv::Op::OpBitwiseAnd: result = values[w[3]] & values[w[4]]; break;
        case spv::Op::OpIAdd: result = values[w[3]] + values[w[4]]; break;
        case spv::Op::OpIMul: result = values[w[3]] * values[w[4]]; break;
        case spv::Op::OpULessThan: result = values[w[3]] < values[w[4]]; break;
        case spv::Op::OpINotEqual: result = values[w[3]] != values[w[4]]; break;
        case spv::Op::OpSelect: result = values[w[3]] ? values[w[4]] : values[w[5]]; break;
        case spv::Op::OpBitFieldSExtract: case spv::Op::OpBitFieldUExtract:
            result = (values[w[3]] >> values[w[4]]) & 255u;
            if (op == spv::Op::OpBitFieldSExtract && result >= 128) result |= 0xffffff00u;
            break;
        default: has_result = false;
        }
        if (has_result) values[w[2]] = result;
    }
    return values[200];
}

bool check(const Case& test)
{
    spvtools::SpirvTools assembler(SPV_ENV_VULKAN_1_1);
    assembler.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        fprintf(stderr, "SPIR-V: %s\n", message);
    });
    std::vector<uint32_t> words;
    if (!assembler.Assemble(fixture(test), &words, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)
        || !assembler.Validate(words)) return false;
    // Mixed-only shaders have no dot4 marker after lowering. The public
    // translation pipeline must still enforce the feature, not confuse the
    // marker count with feature use.
    if (test.mixed)
    {
        nikki_detail::TranslationInput input{};
        input.spirv = words;
        input.entry_point = "main";
        input.shader_profile.max_uniform_buffer_binding_size = 65536;
        input.translator = std::make_shared<nikki_detail::TranslatorInstanceState>();
        input.translator->wgsl_allowed_features = tint::wgsl::AllowedFeatures::Everything();
        input.descriptor_layout.push_back({0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
        nikki_detail::TranslationResult translated;
        if (nikki_detail::translate_shader(input, translated) == 0)
        {
            fprintf(stderr, "Mixed packed dot accepted with the feature disabled\n");
            return false;
        }
        input.shader_profile.packed_integer_dot_enabled = true;
        if (nikki_detail::translate_shader(input, translated) != 0 || translated.wgsl.empty())
        {
            fprintf(stderr, "Mixed packed dot rejected with the feature enabled\n");
            return false;
        }
    }
    size_t count = 0;
    bool uses_integer_dot = false;
    if (nikki_detail::legalize_packed_integer_dot_spirv(words, count, &uses_integer_dot) != 0
        || !uses_integer_dot || !assembler.Validate(words)) return false;
    std::mt19937 random(71238);
    const uint32_t limits[] = {0, 1, 0x7ffffffeu, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu};
    const uint32_t packed[] = {0, 0x01010101u, 0x7f7f7f7fu, 0x80808080u, 0xffffffffu, 0x80ff017fu};
    auto numeric = [&](uint32_t a, uint32_t b, uint32_t acc) {
        int64_t expected = dot(a, b, test.unsigned_dot, test.mixed);
        if (test.saturating)
        {
            expected += test.unsigned_dot ? int64_t(acc) : int64_t(static_cast<int32_t>(acc));
            expected = std::max<int64_t>(test.unsigned_dot ? 0 : INT32_MIN,
                                        std::min<int64_t>(test.unsigned_dot ? int64_t(UINT32_MAX) : INT32_MAX, expected));
        }
        if (test.dual)
        {
            expected += dot(a, b, test.unsigned_dot, test.mixed);
            expected = std::max<int64_t>(test.unsigned_dot ? 0 : INT32_MIN,
                                        std::min<int64_t>(test.unsigned_dot ? int64_t(UINT32_MAX) : INT32_MAX, expected));
        }
        uint32_t actual = evaluate(words, a, b, acc);
        if (actual != static_cast<uint32_t>(expected))
        {
            fprintf(stderr, "%s: %08x . %08x + %08x = %08x, expected %08x\n", test.opcode, a, b, acc, actual, static_cast<uint32_t>(expected));
            return false;
        }
        return true;
    };
    for (uint32_t a : packed) for (uint32_t b : packed) for (uint32_t acc : limits)
        if (!numeric(a, b, acc)) return false;
    for (unsigned i = 0; i < 1000; i++) if (!numeric(random(), random(), random())) return false;
    auto ir = tint::spirv::reader::ReadIR(words);
    if (ir != tint::Success) { fprintf(stderr, "%s\n", ir.Failure().reason.c_str()); return false; }
    if (nikki_detail::restore_packed_integer_dot_ir(ir.Get(), count) != 0) return false;
    tint::wgsl::writer::Options options;
    auto program = tint::wgsl::writer::ProgramFromIR(ir.Get(), options);
    if (program != tint::Success) { fprintf(stderr, "%s\n", program.Failure().reason.c_str()); return false; }
    auto wgsl = tint::wgsl::writer::Generate(program.Get(), options);
    if (wgsl != tint::Success) { fprintf(stderr, "%s\n", wgsl.Failure().reason.c_str()); return false; }
    if (!test.mixed && wgsl->wgsl.find(test.unsigned_dot ? "dot4U8Packed" : "dot4I8Packed") == std::string::npos) return false;
    if (test.saturating && wgsl->wgsl.find("select(") == std::string::npos) return false;
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    tint::Initialize();
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            std::ifstream file(argv[i], std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (!file || bytes.empty() || bytes.size() % sizeof(uint32_t)) return 1;
            std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
            memcpy(words.data(), bytes.data(), bytes.size());
            nikki_detail::TranslationInput input{};
            input.spirv = std::move(words);
            input.entry_point = "main";
            input.shader_profile.packed_integer_dot_enabled = true;
            input.shader_profile.max_uniform_buffer_binding_size = 65536;
            input.translator = std::make_shared<nikki_detail::TranslatorInstanceState>();
            input.translator->wgsl_allowed_features = tint::wgsl::AllowedFeatures::Everything();
            input.descriptor_layout.push_back({0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
            input.descriptor_layout.push_back({0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
            nikki_detail::TranslationResult translated;
            if (nikki_detail::translate_shader(input, translated) != 0) return 1;
            printf("// %s\n%s\n", argv[i], translated.wgsl.c_str());
        }
        tint::Shutdown();
        return 0;
    }
    const Case cases[] = {{"OpSDot", false, false, false}, {"OpUDot", true, false, false},
        {"OpSUDot", false, true, false}, {"OpSDotAccSat", false, false, true},
        {"OpUDotAccSat", true, false, true}, {"OpSUDotAccSat", false, true, true},
        {"OpSDotAccSat", false, false, true, true}, {"OpUDotAccSat", true, false, true, true},
        {"OpSUDotAccSat", false, true, true, true},
        {"OpSDotAccSat", false, false, true, false, true, true},
        {"OpUDotAccSat", true, false, true, false, true, false},
        {"OpSUDotAccSat", false, true, true, false, true, true}};
    for (const Case& test : cases)
    {
        fprintf(stderr, "packed_integer_dot_test: %s%s\n", test.opcode, test.dual ? " dual" : "");
        if (!check(test)) { tint::Shutdown(); return 1; }
    }
    tint::Shutdown();
    return 0;
}
