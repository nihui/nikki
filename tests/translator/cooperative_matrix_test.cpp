// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/translator/translator_stages.h"
#include <spirv-tools/libspirv.hpp>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iterator>

namespace {

// Minimal versions of vkpeak's single/dual cooperative matrix kernels
// (nihui/vkpeak f37dab4c421781bf047face221dbaf6ae112a5e7). In
// particular, the FP16 result lives in float[] (two halfs per storage element),
// not half[]. Offset and stride therefore count 32-bit words, not halfs.
struct Fixture
{
    bool integer = false;
    bool integer_constructors = false;
    bool unsigned_constructor_chain = false;
    unsigned unsupported_scalar = 0;
    bool unsigned_backing = false;
    bool signed_backing = false;
    bool bounded = false;
    bool column_major = false;
    bool ordinary_access = false;
    bool array_length = false;
    bool two_buffers = false;
    bool nested = false;
    bool early_return = false;
    unsigned workgroup_size = 32;
    unsigned offset = 7;
    unsigned stride = 8;
    const char* binary = nullptr;

    std::string assembly() const
    {
        const char* backing = (integer || signed_backing) ? "%i32" : unsigned_backing ? "%u32" : "%f32";
        const char* component = integer ? "%i8" : "%f16";
        const char* accumulator = integer ? "%i32" : "%f16";
        std::ostringstream s;
        s << "OpCapability Shader\nOpCapability " << (integer ? "Int8" : "Float16") << "\n"
          << "OpCapability VulkanMemoryModel\nOpCapability CooperativeMatrixKHR\n"
          << "OpExtension \"SPV_KHR_cooperative_matrix\"\n"
          << "OpExtension \"SPV_KHR_vulkan_memory_model\"\n"
          << "OpMemoryModel Logical Vulkan\nOpEntryPoint GLCompute %main \"main\"\n"
          << "OpExecutionMode %main LocalSize " << workgroup_size << " 1 1\n"
          << "OpDecorate %array ArrayStride 4\nOpDecorate %block Block\n"
          << "OpMemberDecorate %block 0 Offset 0\n"
          << "OpDecorate %buffer DescriptorSet 0\nOpDecorate %buffer Binding 0\n"
          << (two_buffers ? "OpDecorate %block2 Block\nOpMemberDecorate %block2 0 Offset 0\nOpDecorate %buffer2 DescriptorSet 0\nOpDecorate %buffer2 Binding 1\n" : "")
          << "%void = OpTypeVoid\n%bool = OpTypeBool\n%function = OpTypeFunction %void\n"
          << "%u32 = OpTypeInt 32 0\n%i32 = OpTypeInt 32 1\n%f32 = OpTypeFloat 32\n"
          << (integer ? "%i8 = OpTypeInt 8 1\n" : "%f16 = OpTypeFloat 16\n")
          << (unsigned_constructor_chain ? "%u8 = OpTypeInt 8 0\n" : "")
          << "%zero = OpConstant %u32 0\n%one = OpConstant %u32 1\n%two = OpConstant %u32 2\n"
          << "%scope = OpConstant %u32 3\n%size = OpConstant %u32 16\n"
          << "%layout = OpConstant %u32 " << (column_major ? 1 : 0) << "\n"
          << "%offset = OpConstant %u32 " << offset << "\n"
          << "%stride = OpConstant %u32 " << stride << "\n"
          << "%store_stride = OpConstant %u32 " << (stride ? stride : 8) << "\n"
          << "%count = OpConstant %u32 256\n"
          << "%array = " << (bounded ? "OpTypeArray " : "OpTypeRuntimeArray ") << backing
          << (bounded ? " %count\n" : "\n")
          << "%block = OpTypeStruct %array\n%block_ptr = OpTypePointer StorageBuffer %block\n"
          << "%element_ptr = OpTypePointer StorageBuffer " << backing << "\n"
          << "%buffer = OpVariable %block_ptr StorageBuffer\n"
          << (two_buffers ? "%block2 = OpTypeStruct %array\n%block_ptr2 = OpTypePointer StorageBuffer %block2\n%buffer2 = OpVariable %block_ptr2 StorageBuffer\n" : "")
          << "%a_type = OpTypeCooperativeMatrixKHR " << component << " %scope %size %size %zero\n"
          << "%b_type = OpTypeCooperativeMatrixKHR " << component << " %scope %size %size %one\n"
          << "%c_type = OpTypeCooperativeMatrixKHR " << accumulator << " %scope %size %size %two\n"
          << "%f32_zero = OpConstant %f32 0\n"
          << "%initial = OpConstant " << accumulator << " 0\n"
          << "%main = OpFunction %void None %function\n%entry = OpLabel\n"
          << "%ptr = OpAccessChain %element_ptr %buffer %zero %offset\n";
        if (nested || early_return)
        {
            s << "%condition_value = OpLoad %f32 %ptr\n"
              << "%condition = OpFOrdEqual %bool %condition_value %f32_zero\n"
              << "OpSelectionMerge %merge None\nOpBranchConditional %condition %then %merge\n"
              << "%then = OpLabel\n";
            if (early_return)
                s << "OpReturn\n%merge = OpLabel\n";
        }
        if (integer_constructors)
        {
            if (unsigned_constructor_chain)
                s << "%input_u8 = OpUConvert %u8 %offset\n%input8 = OpBitcast %i8 %input_u8\n";
            else
                s << "%input32 = OpBitcast %i32 %offset\n%input8 = OpSConvert %i8 %input32\n";
            if (unsupported_scalar == 1)
                s << "%widened = OpSConvert %i32 %input8\n";
            if (unsupported_scalar == 2)
                s << "%added = OpIAdd %i8 %input8 %input8\n";
            if (unsupported_scalar == 3)
                s << "%float_value = OpConvertUToF %f32 %input8\n";
            s << "%a = OpCompositeConstruct %a_type %input8\n"
              << "%b = OpCompositeConstruct %b_type %input8\n";
        }
        else
            s << "%a = OpCooperativeMatrixLoadKHR %a_type %ptr %layout %stride\n"
              << "%b = OpCooperativeMatrixLoadKHR %b_type %ptr %layout %stride\n";
        s << "%c = OpCompositeConstruct %c_type %initial\n"
          << "%c0 = OpCooperativeMatrixMulAddKHR %c_type %a %b %c"
          << (integer ? " MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR\n" : "\n");
        if (binary)
            s << "%c1 = OpCooperativeMatrixMulAddKHR %c_type %a %b %c"
              << (integer ? " MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR\n" : "\n")
              << "%combined = " << binary << " %c_type %c0 %c1\n";
        s << "OpCooperativeMatrixStoreKHR %ptr " << (binary ? "%combined" : "%c0") << " %layout %store_stride\n";
        if (two_buffers)
            s << "%ptr2 = OpAccessChain %element_ptr %buffer2 %zero %offset\n"
              << "OpCooperativeMatrixStoreKHR %ptr2 %c0 %layout %store_stride\n";
        if (ordinary_access)
            s << "%alias = OpCopyObject %element_ptr %ptr\n"
              << "%ordinary = OpLoad " << backing << " %alias\n"
              << "%other = OpAccessChain %element_ptr %buffer %zero %one\n"
              << "OpStore %other %ordinary\n";
        if (array_length)
            s << "%length = OpArrayLength %u32 %buffer 0\n"
              << "%length_ptr = OpAccessChain %element_ptr %buffer %zero %two\n"
              << "OpStore %length_ptr %length\n";
        if (nested)
            s << "OpBranch %merge\n%merge = OpLabel\n";
        s << "OpReturn\nOpFunctionEnd\n";
        return s.str();
    }
};

bool constant_u32(tint::core::ir::Value* value, uint32_t& result)
{
    if (auto* constant = value->As<tint::core::ir::Constant>())
    {
        result = constant->Value()->ValueAs<uint32_t>();
        return true;
    }
    auto* instruction = value->As<tint::core::ir::InstructionResult>();
    if (instruction)
    {
        if (auto* call = instruction->Instruction()->As<tint::core::ir::CoreBuiltinCall>())
            if (call->Func() == tint::core::BuiltinFn::kBitcast)
                return constant_u32(call->Args()[0], result);
        if (auto* let = instruction->Instruction()->As<tint::core::ir::Let>())
            return constant_u32(let->Value(), result);
    }
    auto* binary = instruction ? instruction->Instruction()->As<tint::core::ir::CoreBinary>() : nullptr;
    uint32_t lhs, rhs;
    if (!binary || !constant_u32(binary->LHS(), lhs) || !constant_u32(binary->RHS(), rhs))
        return false;
    if (binary->Op() == tint::core::BinaryOp::kMultiply)
        result = lhs * rhs;
    else if (binary->Op() == tint::core::BinaryOp::kAdd)
        result = lhs + rhs;
    else if (binary->Op() == tint::core::BinaryOp::kShiftLeft && rhs < 32)
        result = lhs << rhs;
    else if (binary->Op() == tint::core::BinaryOp::kShiftRight && rhs < 32)
    {
        result = lhs >> rhs;
        if (rhs != 0 && binary->LHS()->Type()->Is<tint::core::type::I32>() && (lhs & 0x80000000u))
            result |= ~uint32_t(0) << (32 - rhs);
    }
    else
        return false;
    return true;
}

// Inspect the actual matrix builtin arguments, not unrelated numeric literals
// in WGSL: ordinary accesses also scale their indices and could hide a broken
// matrix offset/stride conversion in a whole-source substring assertion.
bool check_packed_addressing(tint::core::ir::Module& module, const Fixture& fixture)
{
    unsigned stores = 0;
    unsigned loads = 0;
    for (auto* instruction : module.Instructions())
    {
        auto* call = instruction->As<tint::core::ir::CoreBuiltinCall>();
        if (!call)
            continue;
        const bool store = call->Func() == tint::core::BuiltinFn::kSubgroupMatrixStore;
        if (!store && call->Func() != tint::core::BuiltinFn::kSubgroupMatrixLoad)
            continue;
        auto* pointer = call->Args()[0]->Type()->As<tint::core::type::Pointer>();
        if (!pointer || pointer->AddressSpace() != tint::core::AddressSpace::kStorage)
            continue;
        auto* layout = call->Args()[store ? 3 : 2]->As<tint::core::ir::Constant>();
        if (!layout || layout->Value()->ValueAs<bool>() != fixture.column_major)
            return false;
        uint32_t offset, stride;
        if (!constant_u32(call->Args()[1], offset)
            || !constant_u32(call->Args()[store ? 4 : 3], stride)
            || offset != fixture.offset * 2
            || stride != (store && !fixture.stride ? 8 : fixture.stride) * 2)
        {
            fprintf(stderr, "Packed matrix offset/stride do not preserve byte addressing\n");
            return false;
        }
        store ? stores++ : loads++;
    }
    return stores == (fixture.two_buffers ? 2u : 1u) && loads == (fixture.stride ? 2u : 0u);
}

int translate(std::vector<uint32_t> words, std::string& source, bool reject = false, const Fixture* fixture = nullptr)
{
    using namespace nikki_detail;
    if (normalize_spirv(words) != 0)
        return 1;
    CooperativeMatrixLegalization info;
    const int legalized = legalize_cooperative_matrix_spirv(words, info);
    if (fixture && fixture->unsupported_scalar)
    {
        if (legalized == 0)
            fprintf(stderr, "Unsupported scalar8 use was not rejected before Tint\n");
        return legalized == 0 ? 1 : 0;
    }
    if (legalized != 0)
        return 1;
    auto ir = tint::spirv::reader::ReadIR(words);
    if (ir != tint::Success)
    {
        fprintf(stderr, "Tint read: %s\n", ir.Failure().reason.c_str());
        return 1;
    }
    const int restored = restore_cooperative_matrix_ir(ir.Get(), info, 32);
    if (reject)
    {
        if (restored == 0)
            fprintf(stderr, "Expected unsafe divergent arithmetic to be rejected\n");
        return restored == 0 ? 1 : 0;
    }
    if (restored != 0)
        return 1;
    if (fixture && !fixture->integer && !check_packed_addressing(ir.Get(), *fixture))
        return 1;
    if (fixture && fixture->integer_constructors)
    {
        unsigned constructors = 0;
        const uint32_t byte = fixture->offset & 255u;
        const uint32_t expected = (byte & 128u) ? (byte | 0xffffff00u) : byte;
        for (auto* instruction : ir.Get().Instructions())
        {
            auto* construct = instruction->As<tint::core::ir::Construct>();
            auto* matrix = construct ? construct->Result()->Type()->As<tint::core::type::SubgroupMatrix>() : nullptr;
            if (!matrix || matrix->Kind() == tint::core::SubgroupMatrixKind::kResult)
                continue;
            uint32_t value;
            if (construct->Args().size() != 1 || !constant_u32(construct->Args()[0], value) || value != expected)
            {
                fprintf(stderr, "INT8 constructor does not preserve low-byte signed value for %u\n", fixture->offset);
                return 1;
            }
            constructors++;
        }
        if (constructors != 2)
            return 1;
    }
    if (fixture && fixture->binary)
    {
        const std::string operation = fixture->binary;
        const auto expected = operation.find("Add") != std::string::npos ? tint::core::BinaryOp::kAdd
                            : operation.find("Sub") != std::string::npos ? tint::core::BinaryOp::kSubtract
                            : tint::core::BinaryOp::kMultiply;
        unsigned scalar_operations = 0;
        for (auto* instruction : ir.Get().Instructions())
        {
            auto* binary = instruction->As<tint::core::ir::CoreBinary>();
            if (binary && binary->Op() == expected
                && (fixture->integer ? binary->Result()->Type()->Is<tint::core::type::I32>()
                                     : binary->Result()->Type()->Is<tint::core::type::F16>()))
                scalar_operations++;
        }
        if (scalar_operations != 8) // 16x16 elements / 32 lanes
        {
            fprintf(stderr, "Matrix binary fallback has %u scalar operations, expected 8\n", scalar_operations);
            return 1;
        }
    }
    std::vector<SubgroupBuiltinSubstitution> substitutions;
    bool linear_indexing = false;
    if (legalize_subgroup_id_builtins(ir.Get(), substitutions, linear_indexing) != 0)
        return 1;
    tint::wgsl::writer::Options options;
    options.allowed_features.extensions.insert(tint::wgsl::Extension::kChromiumExperimentalSubgroupMatrix);
    options.allowed_features.extensions.insert(tint::wgsl::Extension::kF16);
    options.allowed_features.extensions.insert(tint::wgsl::Extension::kSubgroups);
    auto program = tint::wgsl::writer::ProgramFromIR(ir.Get(), options);
    if (program != tint::Success)
    {
        fprintf(stderr, "Tint IR conversion: %s\n", program.Failure().reason.c_str());
        return 1;
    }
    auto wgsl = tint::wgsl::writer::Generate(program.Get(), options);
    if (wgsl != tint::Success)
    {
        fprintf(stderr, "WGSL generation: %s\n", wgsl.Failure().reason.c_str());
        return 1;
    }
    source = wgsl->wgsl;
    if (restore_subgroup_id_builtins(source, substitutions) != 0)
        return 1;
    return 0;
}

int check(const char* name, const Fixture& fixture, bool reject = false)
{
    fprintf(stderr, "cooperative_matrix_test: %s\n", name);
    spvtools::SpirvTools assembler(SPV_ENV_VULKAN_1_1);
    assembler.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "SPIR-V at %zu:%zu: %s\n", position.line, position.column, message);
    });
    std::vector<uint32_t> words;
    if (!assembler.Assemble(fixture.assembly(), &words) || !assembler.Validate(words))
        return 1;
    std::string source;
    if (translate(words, source, reject, &fixture) != 0)
        return 1;
    if (reject)
        return 0;
    const auto has = [&](const std::string& text) { return source.find(text) != std::string::npos; };
    bool valid = has("subgroupMatrixMultiplyAccumulate") && has("subgroupMatrixStore");
    if (!fixture.integer)
    {
        valid = valid && has(fixture.bounded ? "array<f16, 512u>" : "array<f16>");
        if (fixture.ordinary_access)
            valid = valid && has("bitcast<vec2<f16>>")
                    && has(fixture.unsigned_backing ? "bitcast<u32>" : fixture.signed_backing ? "bitcast<i32>" : "bitcast<f32>");
        if (fixture.stride == 0)
            valid = valid && has("var<workgroup> nikki_cooperative_matrix_broadcast") && has("workgroupBarrier()");
        if (fixture.array_length)
            valid = valid && has("arrayLength") && has("/ 2u");
    }
    if (fixture.binary)
    {
        const char* op = std::string(fixture.binary).find("Add") != std::string::npos ? " + "
                       : std::string(fixture.binary).find("Sub") != std::string::npos ? " - " : " * ";
        unsigned barriers = 0;
        for (size_t p = 0; (p = source.find("workgroupBarrier();", p)) != std::string::npos; p++)
            barriers++;
        valid = valid && has("var<workgroup> nikki_matrix_binary_lhs")
                && has("var<workgroup> nikki_matrix_binary_rhs") && has(op) && barriers >= 3
                && has("subgroup_invocation_id") && has("subgroup_id")
                && has(fixture.workgroup_size == 64 ? ", 512u>" : ", 256u>");
    }
    if (!valid)
    {
        fprintf(stderr, "Unexpected WGSL:\n%s\n", source.c_str());
        return 1;
    }
    return 0;
}

int translate_production(std::vector<uint32_t> words, const char* entry_point, nikki_detail::TranslationResult& result, bool matrix_features = true)
{
    nikki_detail::TranslationInput input{};
    input.spirv = std::move(words);
    input.entry_point = entry_point;
    input.shader_profile.shader_f16_enabled = true;
    input.shader_profile.subgroup_enabled = matrix_features;
    input.shader_profile.cooperative_matrix_enabled = matrix_features;
    input.shader_profile.packed_integer_dot_enabled = matrix_features;
    input.shader_profile.subgroup_default_size = 32;
    input.shader_profile.max_uniform_buffer_binding_size = 65536;
    input.translator = std::make_shared<nikki_detail::TranslatorInstanceState>();
    input.translator->wgsl_allowed_features = tint::wgsl::AllowedFeatures::Everything();
    input.descriptor_layout.push_back({0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    return nikki_detail::translate_shader(input, result) == 0 ? 0 : 1;
}

size_t occurrences(const std::string& source, const std::string& text)
{
    size_t count = 0;
    for (size_t p = 0; (p = source.find(text, p)) != std::string::npos; p += text.size())
        count++;
    return count;
}

bool replace_once(std::string& source, const std::string& from, const std::string& to)
{
    const size_t p = source.find(from);
    if (p == std::string::npos)
        return false;
    source.replace(p, from.size(), to);
    return true;
}

int production_fixture(const char* name, const std::string& assembly, const char* entry_point,
                       nikki_detail::TranslationResult& result, bool matrix_features = true)
{
    fprintf(stderr, "cooperative_matrix_test: %s\n", name);
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "SPIR-V at %zu:%zu: %s\n", position.line, position.column, message);
    });
    std::vector<uint32_t> words;
    if (!tools.Assemble(assembly, &words) || !tools.Validate(words))
        return 1;
    return translate_production(std::move(words), entry_point, result, matrix_features);
}

int check_packed_constants()
{
    int failures = 0;
    struct PackedCase { uint32_t bits; unsigned backing; };
    const PackedCase cases[] = {{0xffffffffu, 0}, {0x0000ffffu, 0}, {0x7c007c00u, 0}, {0u, 0},
                                {0xffffffffu, 1}, {0x0000ffffu, 2}};
    for (const PackedCase& test : cases)
    {
        const uint32_t bits = test.bits;
        Fixture fixture;
        fixture.unsigned_backing = test.backing == 0;
        fixture.signed_backing = test.backing == 1;
        fixture.ordinary_access = true;
        std::string assembly = fixture.assembly();
        const std::string literal = std::to_string(bits);
        const std::string stored = test.backing == 0 ? "OpStore %other %sentinel"
            : "%sentinel_cast = OpBitcast " + std::string(test.backing == 1 ? "%i32" : "%f32")
                  + " %sentinel\nOpStore %other %sentinel_cast";
        if (!replace_once(assembly, "%main = OpFunction", "%sentinel = OpConstant %u32 " + literal + "\n%main = OpFunction")
            || !replace_once(assembly, "OpStore %other %ordinary", stored))
            return 1;
        nikki_detail::TranslationResult result;
        if (production_fixture(("packed constant " + literal + " backing " + std::to_string(test.backing)).c_str(), assembly, "main", result) != 0)
        {
            failures++;
            continue;
        }
        // This must be a runtime bit reinterpretation, not a non-finite f16
        // constant or a numeric conversion that changes the NaN payload bits.
        const std::string& source = result.wgsl;
        const size_t helper = source.find("fn nikki_unpack_f16");
        const size_t body = helper == std::string::npos ? helper : source.find('{', helper);
        const size_t end = body == std::string::npos ? body : source.find('}', body);
        const std::string implementation = body == std::string::npos || end == std::string::npos
                                               ? "" : source.substr(body, end - body);
        const std::string argument = test.backing == 0 ? literal + "u"
            : "bitcast<u32>(bitcast<" + std::string(test.backing == 1 ? "i32" : "f32") + ">(" + literal + "u))";
        const std::string expected_call = "nikki_unpack_f16(" + argument + ")";
        const size_t call = source.find(expected_call);
        const size_t let = call == std::string::npos ? call : source.rfind("let ", call);
        const size_t equals = let == std::string::npos ? let : source.find(" = ", let);
        const std::string value = let == std::string::npos || equals == std::string::npos
                                     ? "" : source.substr(let + 4, equals - let - 4);
        if (value.empty() || source.find("= " + value + ".x;") == std::string::npos
            || source.find("= " + value + ".y;") == std::string::npos
            || implementation.find("return bitcast<vec2<f16>>(bits);") == std::string::npos
            || call == std::string::npos)
        {
            fprintf(stderr, "Packed constant bit-preservation path missing:\n%s\n", source.c_str());
            failures++;
        }
    }
    return failures;
}

int check_reused_scratch()
{
    Fixture fixture;
    fixture.integer = true;
    fixture.stride = 16;
    fixture.binary = "OpIAdd";
    fixture.workgroup_size = 128;
    std::string assembly = fixture.assembly();
    if (!replace_once(assembly, "OpCooperativeMatrixStoreKHR %ptr %combined",
        "%sum2 = OpIAdd %c_type %combined %c1\n"
        "%sum3 = OpIAdd %c_type %sum2 %c1\n"
        "%sum4 = OpIAdd %c_type %sum3 %c1\n"
        "%sum5 = OpIAdd %c_type %sum4 %c1\n"
        "OpCooperativeMatrixStoreKHR %ptr %sum5"))
        return 1;
    nikki_detail::TranslationResult result;
    if (production_fixture("five matrix additions reuse 8 KiB scratch", assembly, "main", result) != 0)
        return 1;
    const std::string& source = result.wgsl;
    // Four subgroups * 256 i32 elements * two arrays = 8192 bytes, not
    // five separate pairs (40960 bytes). Each reuse must complete its load
    // before the following operation overwrites either scratch array.
    if (occurrences(source, "var<workgroup>") != 2
        || occurrences(source, "array<i32, 1024u>") != 2
        || occurrences(source, "workgroupBarrier();") < 15
        || occurrences(source, "subgroupMatrixLoad<subgroup_matrix_result<i32") != 5)
    {
        fprintf(stderr, "Scratch capacity/reuse or synchronization incorrect:\n%s\n", source.c_str());
        return 1;
    }
    return 0;
}

int check_scratch_capacity()
{
    int failures = 0;
    for (bool small_first : {true, false})
    {
        Fixture fixture;
        fixture.integer = true;
        fixture.stride = 16;
        fixture.binary = "OpIAdd";
        fixture.workgroup_size = 128;
        std::string assembly = fixture.assembly();
        if (!replace_once(assembly, "%main = OpFunction",
            "%smallsize = OpConstant %u32 8\n"
            "%small_a_type = OpTypeCooperativeMatrixKHR %i8 %scope %smallsize %smallsize %zero\n"
            "%small_b_type = OpTypeCooperativeMatrixKHR %i8 %scope %smallsize %smallsize %one\n"
            "%small_c_type = OpTypeCooperativeMatrixKHR %i32 %scope %smallsize %smallsize %two\n"
            "%main = OpFunction"))
            return 1;
        const std::string small =
            "%small_a = OpCooperativeMatrixLoadKHR %small_a_type %ptr %layout %stride\n"
            "%small_b = OpCooperativeMatrixLoadKHR %small_b_type %ptr %layout %stride\n"
            "%small_c = OpCompositeConstruct %small_c_type %initial\n"
            "%small_c0 = OpCooperativeMatrixMulAddKHR %small_c_type %small_a %small_b %small_c MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR\n"
            "%small_sum = OpIAdd %small_c_type %small_c0 %small_c0\n"
            "OpCooperativeMatrixStoreKHR %ptr %small_sum %layout %store_stride\n";
        const std::string anchor = small_first ? "%a = OpCooperativeMatrixLoadKHR" : "OpReturn";
        if (!replace_once(assembly, anchor, small + anchor))
            return 1;
        nikki_detail::TranslationResult result;
        if (production_fixture(small_first ? "scratch grows to maximum capacity" : "scratch does not shrink", assembly, "main", result) != 0)
        {
            failures++;
            continue;
        }
        if (occurrences(result.wgsl, "var<workgroup>") != 2
            || occurrences(result.wgsl, "array<i32, 1024u>") != 2
            || occurrences(result.wgsl, "workgroupBarrier();") < 6)
        {
            fprintf(stderr, "Scratch must use max capacity, not sum or last size:\n%s\n", result.wgsl.c_str());
            failures++;
        }
    }
    return failures;
}

int check_selected_entries()
{
    Fixture fixture;
    fixture.binary = "OpFAdd";
    std::string assembly = fixture.assembly();
    // A live packed-dot helper is shared by both matrix entry call trees.
    // Entry selection must retain the helper for either matrix entry, but drop
    // its markers and capabilities when selecting the plain entry below.
    if (!replace_once(assembly, "OpCapability Shader\n",
            "OpCapability Shader\nOpCapability DotProduct\nOpCapability DotProductInput4x8BitPacked\n")
        || !replace_once(assembly, "OpMemoryModel", "OpExtension \"SPV_KHR_integer_dot_product\"\nOpMemoryModel")
        || !replace_once(assembly, "%main = OpFunction", "%shared_func_type = OpTypeFunction %u32 %u32\n%main = OpFunction")
        || !replace_once(assembly, "OpReturn",
            "%shared_loaded = OpLoad %f32 %ptr\n%shared_bits = OpBitcast %u32 %shared_loaded\n"
            "%shared_dot = OpFunctionCall %u32 %shared_helper %shared_bits\n"
            "%shared_float = OpBitcast %f32 %shared_dot\nOpStore %ptr %shared_float\nOpReturn"))
        return 1;
    const std::string helper =
        "%shared_helper = OpFunction %u32 None %shared_func_type\n"
        "%helper_arg = OpFunctionParameter %u32\n%helper_entry = OpLabel\n"
        "%helper_dot = OpUDot %u32 %helper_arg %helper_arg PackedVectorFormat4x8Bit\n"
        "OpReturnValue %helper_dot\nOpFunctionEnd\n";
    nikki_detail::TranslationResult result;
    int failures = production_fixture("single-entry control with shared dot helper", assembly + helper, "main", result);
    if (!replace_once(assembly, "OpExecutionMode %main",
        "OpEntryPoint GLCompute %other_main \"other\"\n"
        "OpExecutionMode %other_main LocalSize 64 1 1\nOpExecutionMode %main"))
        return 1;
    // Duplicate the body while renaming every local result and label. Both
    // entries contain matrix markers, so selecting either must discard the
    // other's markers as well as its compute entry point.
    std::string other = assembly.substr(assembly.find("%main = OpFunction"));
    std::unordered_set<std::string> locals;
    std::istringstream lines(other);
    std::string line;
    while (std::getline(lines, line))
    {
        const size_t equals = line.find(" = ");
        if (equals != std::string::npos && line[0] == '%')
            locals.insert(line.substr(0, equals));
    }
    std::string renamed;
    for (size_t p = 0; p < other.size();)
    {
        if (other[p] != '%')
        {
            renamed += other[p++];
            continue;
        }
        size_t end = other.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", p + 1);
        if (end == std::string::npos)
            end = other.size();
        const std::string id = other.substr(p, end - p);
        renamed += locals.count(id) ? "%other_" + id.substr(1) : id;
        p = end;
    }
    assembly += renamed + helper;
    for (const char* entry : {"main", "other"})
    {
        if (production_fixture((std::string("select matrix entry ") + entry).c_str(), assembly, entry, result) != 0)
        {
            failures++;
            continue;
        }
        const uint32_t expected_size = std::string(entry) == "main" ? 32 : 64;
        if (occurrences(result.wgsl, "@compute") != 1 || result.workgroup_size_x != expected_size
            || occurrences(result.wgsl, "var<workgroup>") != 2
            || result.wgsl.find("dot4U8Packed") == std::string::npos
            || occurrences(result.wgsl, expected_size == 32 ? "array<f16, 256u>" : "array<f16, 512u>") != 2)
        {
            fprintf(stderr, "Wrong entry point or resource footprint selected:\n%s\n", result.wgsl.c_str());
            failures++;
        }
    }
    if (!replace_once(assembly, "OpExecutionMode %other_main",
        "OpEntryPoint GLCompute %plain_main \"plain\"\n"
        "OpExecutionMode %plain_main LocalSize 1 1 1\nOpExecutionMode %other_main"))
        return 1;
    assembly += "%plain_main = OpFunction %void None %function\n%plain_entry = OpLabel\nOpReturn\nOpFunctionEnd\n";
    if (production_fixture("select plain entry without matrix/subgroup features", assembly, "plain", result, false) != 0)
        failures++;
    else if (occurrences(result.wgsl, "@compute") != 1 || result.uses_subgroup
             || result.wgsl.find("subgroupMatrix") != std::string::npos)
        failures++;
    if (production_fixture("reject missing entry", assembly, "missing", result) == 0)
    {
        fprintf(stderr, "Missing entry point was not rejected\n");
        failures++;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv)
{
    std::call_once(nikki_detail::g_tint_initialize_once, []() { tint::Initialize(); });
    // Optional specialized SPIR-V inputs exercise the full production pipeline for real
    // upstream shaders without adding a glslang build dependency to this test.
    // Inputs use Vulkan 1.1/SPIR-V 1.3, entry point main, binding 0 storage,
    // and 32-lane subgroups, matching the vkpeak compile-only test profile.
    if (argc > 1)
    {
        int failures = 0;
        for (int i = 1; i < argc; i++)
        {
            fprintf(stderr, "cooperative_matrix_test: %s\n", argv[i]);
            std::ifstream file(argv[i], std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (!file || bytes.empty() || bytes.size() % sizeof(uint32_t))
            {
                fprintf(stderr, "Cannot read SPIR-V input\n");
                failures++;
                continue;
            }
            std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
            memcpy(words.data(), bytes.data(), bytes.size());
            nikki_detail::TranslationResult result;
            failures += translate_production(std::move(words), "main", result);
            if (!result.wgsl.empty())
                printf("// %s\n%s\n", argv[i], result.wgsl.c_str());
        }
        return failures ? 1 : 0;
    }
    int failures = nikki_detail::cooperative_matrix_translation_probe() ? 0 : 1;
    failures += check_packed_constants();
    failures += check_reused_scratch();
    failures += check_scratch_capacity();
    failures += check_selected_entries();
    Fixture f;
    failures += check("vkpeak FP16 single, packed float buffer", f);
    f.column_major = true;
    failures += check("column-major packed float buffer", f);
    f.column_major = false;
    f.two_buffers = true;
    failures += check("distinct buffer structs sharing packed array type", f);
    f.two_buffers = false;
    f.binary = "OpFAdd";
    failures += check("vkpeak FP16 dual, matrix addition", f);
    f.workgroup_size = 64;
    failures += check("two independent subgroups", f);
    f.workgroup_size = 16;
    failures += check("reject partial subgroup", f, true);
    f.workgroup_size = 32;
    f.nested = true;
    failures += check("reject divergent matrix arithmetic", f, true);
    f.nested = false;
    f.early_return = true;
    failures += check("reject matrix arithmetic after conditional return", f, true);
    f.early_return = false;
    f.binary = "OpFSub";
    failures += check("FP16 matrix subtraction", f);
    f.binary = "OpFMul";
    failures += check("FP16 matrix component multiplication", f);
    f = Fixture();
    f.unsigned_backing = true;
    f.ordinary_access = true;
    failures += check("packed uint buffer with ordinary aliases", f);
    f.array_length = true;
    failures += check("packed uint buffer array length", f);
    f.array_length = false;
    f.unsigned_backing = false;
    failures += check("packed float buffer with ordinary aliases", f);
    f.signed_backing = true;
    failures += check("packed signed int buffer with ordinary aliases", f);
    f.signed_backing = false;
    f.bounded = true;
    failures += check("bounded packed float buffer", f);
    f = Fixture();
    f.stride = 0;
    failures += check("zero stride packed float buffer", f);
    f = Fixture();
    f.integer = true;
    f.stride = 16;
    f.binary = "OpIAdd";
    failures += check("vkpeak INT8 dual, matrix addition", f);
    f.integer_constructors = true;
    failures += check("vkpeak INT8 scalar constructor inputs", f);
    f.unsigned_constructor_chain = true;
    f.offset = 127;
    failures += check("INT8 constructor, 127 preserved", f);
    f.offset = 255;
    failures += check("vkpeak unsigned-to-signed INT8 constructor, 255 becomes -1", f);
    f.offset = 128;
    failures += check("INT8 constructor, 128 becomes -128", f);
    f.offset = 256;
    failures += check("INT8 constructor, 256 becomes zero", f);
    f.unsupported_scalar = 1;
    failures += check("reject scalar8 widening", f, true);
    f.unsupported_scalar = 2;
    failures += check("reject scalar8 wrapping arithmetic", f, true);
    f.unsupported_scalar = 3;
    failures += check("reject scalar8 unsigned-to-float consumer", f, true);
    f.unsupported_scalar = 0;
    f.unsigned_constructor_chain = false;
    f.integer_constructors = false;
    f.offset = 7;
    f.binary = "OpISub";
    failures += check("INT8 matrix subtraction", f);
    f.binary = "OpIMul";
    failures += check("INT8 matrix component multiplication", f);
    return failures ? 1 : 0;
}
