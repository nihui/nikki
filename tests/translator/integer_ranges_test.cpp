// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/translator/translator_stages.h"

#include <spirv-tools/libspirv.hpp>

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <map>
#include <sstream>

struct Fixture
{
    bool signed_type = true;
    std::string initial = "0";
    std::string bound = "9";
    std::string divisor = "9";
    std::string step = "1";
    std::string decorations;
    std::string variables;
    std::string initialization;
    std::string condition;
    std::string body;
    std::string continuation;

    std::string assembly() const
    {
        const char* divide = signed_type ? "OpSDiv" : "OpUDiv";
        const char* modulo = signed_type ? "OpSMod" : "OpUMod";
        std::ostringstream text;
        text << "OpCapability Shader\n"
             << "OpMemoryModel Logical GLSL450\n"
             << "OpEntryPoint GLCompute %main \"main\"\n"
             << "OpExecutionMode %main LocalSize 1 1 1\n"
             << decorations
             << "%void = OpTypeVoid\n"
             << "%bool = OpTypeBool\n"
             << "%integer = OpTypeInt 32 " << signed_type << "\n"
             << "%pointer = OpTypePointer Function %integer\n"
             << "%function = OpTypeFunction %void\n"
             << "%initial = OpConstant %integer " << initial << "\n"
             << "%bound = OpConstant %integer " << bound << "\n"
             << "%divisor = OpConstant %integer " << divisor << "\n"
             << "%step = OpConstant %integer " << step << "\n"
             << "%main = OpFunction %void None %function\n"
             << "%entry = OpLabel\n"
             << "%counter = OpVariable %pointer Function\n"
             << variables
             << "OpStore %counter %initial\n"
             << initialization
             << "OpBranch %header\n"
             << "%header = OpLabel\n"
             << "OpLoopMerge %merge %continue None\n"
             << "OpBranch %condition\n"
             << "%condition = OpLabel\n"
             << "%current = OpLoad %integer %counter\n"
             << condition
             << "%less = " << (signed_type ? "OpSLessThan" : "OpULessThan") << " %bool %current %bound\n"
             << "OpBranchConditional %less %body %merge\n"
             << "%body = OpLabel\n"
             << body
             << "%value = OpLoad %integer %counter\n"
             << "%quotient = " << divide << " %integer %value %divisor\n"
             << "%modulo = " << modulo << " %integer %value %divisor\n";
        if (signed_type)
            text << "%remainder = OpSRem %integer %value %divisor\n";
        text << "OpBranch %continue\n"
             << "%continue = OpLabel\n"
             << "%prior = OpLoad %integer %counter\n"
             << "%next = OpIAdd %integer %prior %step\n"
             << "OpStore %counter %next\n"
             << continuation
             << "%advanced = OpLoad %integer %counter\n"
             << "%continue_quotient = " << divide << " %integer %advanced %divisor\n"
             << "OpBranch %header\n"
             << "%merge = OpLabel\n"
             << "%final = OpLoad %integer %counter\n"
             << "%final_quotient = " << divide << " %integer %final %divisor\n"
             << "OpReturn\n"
             << "OpFunctionEnd\n";
        return text.str();
    }
};

static std::string replace(std::string text, const std::string& from, const std::string& to)
{
    const size_t position = text.find(from);
    if (position == std::string::npos)
    {
        fprintf(stderr, "integer_ranges_test: missing fixture fragment: %s\n", from.c_str());
        abort();
    }
    text.replace(position, from.size(), to);
    return text;
}

static unsigned count_divisions(const std::vector<uint32_t>& words)
{
    unsigned remaining = 0;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16)
    {
        const spv::Op opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        remaining += opcode == spv::Op::OpSDiv || opcode == spv::Op::OpUDiv || opcode == spv::Op::OpSMod || opcode == spv::Op::OpSRem || opcode == spv::Op::OpUMod;
    }
    return remaining;
}

static bool check_replacements(const std::vector<uint32_t>& original, const std::vector<uint32_t>& rewritten)
{
    std::map<uint32_t, size_t> definitions;
    for (size_t offset = 5; offset < rewritten.size(); offset += rewritten[offset] >> 16)
    {
        bool has_result = false;
        bool has_type = false;
        spv::HasResultAndType(static_cast<spv::Op>(rewritten[offset] & 0xffffu), &has_result, &has_type);
        if (has_result)
            definitions[rewritten[offset + (has_type ? 2 : 1)]] = offset;
    }
    for (size_t offset = 5; offset < original.size(); offset += original[offset] >> 16)
    {
        const spv::Op opcode = static_cast<spv::Op>(original[offset] & 0xffffu);
        const bool division = opcode == spv::Op::OpSDiv || opcode == spv::Op::OpUDiv;
        const bool remainder = opcode == spv::Op::OpSMod || opcode == spv::Op::OpSRem || opcode == spv::Op::OpUMod;
        if (!division && !remainder)
            continue;
        const auto result = definitions.find(original[offset + 2]);
        if (result == definitions.end())
            return false;
        const size_t replacement = result->second;
        if (rewritten[replacement] == original[offset])
        {
            if (!std::equal(original.begin() + offset, original.begin() + offset + 5, rewritten.begin() + replacement))
                return false;
            continue;
        }
        if (rewritten[replacement] != ((4u << 16) | (uint32_t)spv::Op::OpCopyObject) || rewritten[replacement + 1] != original[offset + 1])
            return false;
        if (remainder)
        {
            if (rewritten[replacement + 3] != original[offset + 3])
                return false;
            continue;
        }
        const auto zero = definitions.find(rewritten[replacement + 3]);
        if (zero == definitions.end() || rewritten[zero->second + 1] != original[offset + 1])
            return false;
        const uint32_t declaration = rewritten[zero->second];
        if (declaration != ((3u << 16) | (uint32_t)spv::Op::OpConstantNull)
            && (declaration != ((4u << 16) | (uint32_t)spv::Op::OpConstant) || rewritten[zero->second + 3] != 0))
            return false;
    }
    return true;
}

static int check(const char* name, const std::string& assembly, unsigned expected)
{
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([name](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "integer_ranges_test: %s SPIR-V at %zu:%zu: %s\n", name, position.line, position.column, message);
    });
    std::vector<uint32_t> words;
    if (!tools.Assemble(assembly, &words) || !tools.Validate(words))
        return 1;
    const std::vector<uint32_t> original = words;
    nikki_detail::simplify_bounded_integer_loops(words);
    if (!tools.Validate(words) || !check_replacements(original, words))
        return 1;
    const unsigned remaining = count_divisions(words);
    if (remaining != expected)
    {
        fprintf(stderr, "integer_ranges_test: %s remaining divisions/remainders=%u expected=%u\n", name, remaining, expected);
        return 1;
    }
    // A second invocation must preserve the already rewritten, valid module.
    const std::vector<uint32_t> first = words;
    nikki_detail::simplify_bounded_integer_loops(words);
    if (words != first)
    {
        fprintf(stderr, "integer_ranges_test: %s is not idempotent\n", name);
        return 1;
    }
    return 0;
}

static std::string live_assembly(const Fixture& fixture)
{
    std::string source = fixture.assembly();
    source = replace(source, "%void = OpTypeVoid",
                     "OpDecorate %output_type Block\n"
                     "OpMemberDecorate %output_type 0 Offset 0\n"
                     "OpMemberDecorate %output_type 1 Offset 4\n"
                     "OpMemberDecorate %output_type 2 Offset 8\n"
                     "OpMemberDecorate %output_type 3 Offset 12\n"
                     "OpMemberDecorate %output_type 4 Offset 16\n"
                     "OpDecorate %output DescriptorSet 0\n"
                     "OpDecorate %output Binding 0\n"
                     "%void = OpTypeVoid");
    source = replace(source, "%main = OpFunction",
                     "%index0 = OpConstant %integer 0\n"
                     "%index1 = OpConstant %integer 1\n"
                     "%index2 = OpConstant %integer 2\n"
                     "%index3 = OpConstant %integer 3\n"
                     "%index4 = OpConstant %integer 4\n"
                     "%output_type = OpTypeStruct %integer %integer %integer %integer %integer\n"
                     "%output_pointer = OpTypePointer StorageBuffer %output_type\n"
                     "%result_pointer = OpTypePointer StorageBuffer %integer\n"
                     "%output = OpVariable %output_pointer StorageBuffer\n"
                     "%main = OpFunction");
    source = replace(source, "OpBranch %continue",
                     "%result0 = OpAccessChain %result_pointer %output %index0\n"
                     "OpStore %result0 %quotient\n"
                     "%result1 = OpAccessChain %result_pointer %output %index1\n"
                     "OpStore %result1 %modulo\n"
                     "OpBranch %continue");
    if (fixture.signed_type)
        source = replace(source, "OpBranch %continue",
                         "%result2 = OpAccessChain %result_pointer %output %index2\n"
                         "OpStore %result2 %remainder\n"
                         "OpBranch %continue");
    source = replace(source, "%integer %advanced %divisor\n",
                     "%integer %advanced %divisor\n"
                     "%result3 = OpAccessChain %result_pointer %output %index3\n"
                     "OpStore %result3 %continue_quotient\n");
    source = replace(source, "OpReturn\n",
                     "%result4 = OpAccessChain %result_pointer %output %index4\n"
                     "OpStore %result4 %final_quotient\n"
                     "OpReturn\n");
    return source;
}

static int check_specialization(bool signed_type, uint32_t divisor, unsigned expected)
{
    // Keep every result observable so DCE cannot hide a missing or incorrect pass.
    Fixture fixture;
    fixture.signed_type = signed_type;
    std::string source = live_assembly(fixture);
    source = replace(source, "%void = OpTypeVoid", "OpDecorate %divisor SpecId 7\n%void = OpTypeVoid");
    source = replace(source, "%divisor = OpConstant", "%divisor = OpSpecConstant");
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "integer_ranges_test: specialization SPIR-V at %zu:%zu: %s\n", position.line, position.column, message);
    });
    std::vector<uint32_t> words;
    if (!tools.Assemble(source, &words) || !tools.Validate(words))
        return 1;
    const VkSpecializationMapEntry entry = {7, 0, sizeof(divisor)};
    const VkSpecializationInfo info = {1, &entry, sizeof(divisor), &divisor};
    if (nikki_detail::specialize_spirv(words, &info) != 0 || !tools.Validate(words))
        return 1;
    if (count_divisions(words) != expected)
    {
        fprintf(stderr, "integer_ranges_test: specialization signed=%d divisor=%u remaining=%u expected=%u\n", signed_type, divisor, count_divisions(words), expected);
        return 1;
    }
    return 0;
}

int main()
{
    int failures = 0;
    Fixture f;
    failures += check("signed, preserve continue and exit", f.assembly(), 2);
    failures += check("constant-null initial", replace(f.assembly(), "%initial = OpConstant %integer 0", "%initial = OpConstantNull %integer"), 2);
    failures += check("unspecialized bound", replace(f.assembly(), "%bound = OpConstant", "%bound = OpSpecConstant"), 5);
    failures += check("mismatched comparison signedness", replace(f.assembly(), "OpSLessThan", "OpULessThan"), 5);
    std::string wide = replace(f.assembly(), "OpCapability Shader", "OpCapability Shader\nOpCapability Int64");
    wide = replace(wide, "%integer = OpTypeInt 32", "%integer = OpTypeInt 64");
    failures += check("64-bit counter", wide, 5);
    f.signed_type = false;
    failures += check("unsigned", f.assembly(), 2);
    f = Fixture();
    f.initial = "3";
    f.bound = "11";
    f.divisor = "15";
    failures += check("nonzero initial and larger divisor, allocate zero", f.assembly(), 2);
    f.initial = "2147483645";
    f.bound = f.divisor = "2147483647";
    failures += check("signed maximum", f.assembly(), 2);
    f.signed_type = false;
    f.initial = "4294967292";
    f.bound = f.divisor = "4294967295";
    failures += check("unsigned maximum", f.assembly(), 2);
    f = Fixture();
    f.initial = "-12";
    failures += check("negative initial", f.assembly(), 5);
    f = Fixture();
    f.divisor = "8";
    failures += check("smaller divisor", f.assembly(), 5);
    f.divisor = "-9";
    failures += check("negative divisor", f.assembly(), 5);
    f.divisor = "0";
    failures += check("zero divisor", f.assembly(), 5);
    f = Fixture();
    f.step = "2";
    failures += check("nonunit step", f.assembly(), 5);
    f = Fixture();
    f.initial = "9";
    failures += check("empty loop", f.assembly(), 5);
    f = Fixture();
    f.body = "OpStore %counter %divisor\n";
    failures += check("counter modified in body", f.assembly(), 5);
    f = Fixture();
    f.body = "%alias = OpCopyObject %pointer %counter\n";
    failures += check("counter alias", f.assembly(), 5);
    f = Fixture();
    f.decorations = "OpDecorate %counter Volatile\n";
    failures += check("volatile variable", f.assembly(), 5);
    f.decorations = "%group = OpDecorationGroup\nOpDecorate %group Volatile\nOpGroupDecorate %group %counter\n";
    failures += check("group-applied volatile", f.assembly(), 5);
    f = Fixture();
    failures += check("volatile access", replace(f.assembly(), "%value = OpLoad %integer %counter", "%value = OpLoad %integer %counter Volatile"), 5);
    failures += check("condition value copied in body", replace(f.assembly(), "%value = OpLoad %integer %counter", "%value = OpCopyObject %integer %current"), 5);
    std::string source = replace(f.assembly(), "%integer %value %divisor", "%integer %current %divisor");
    failures += check("dominating condition load", source, 2);
    source = replace(f.assembly(), "%next = OpIAdd %integer %prior %step", "%next = OpIAdd %integer %step %prior");
    failures += check("commuted increment", source, 2);
    source = replace(f.assembly(), "%next = OpIAdd %integer %prior %step", "%next = OpIAdd %integer %initial %step");
    failures += check("increment unrelated to counter", source, 5);
    f.variables = "%local_bound = OpVariable %pointer Function\n";
    f.initialization = "%product = OpIMul %integer %bound %step\nOpStore %local_bound %product\n";
    f.condition = "%limit = OpLoad %integer %local_bound\n";
    source = replace(f.assembly(), "%bool %current %bound", "%bool %current %limit");
    failures += check("single-store constant bound", source, 2);
    f.continuation = "OpStore %local_bound %divisor\n";
    source = replace(f.assembly(), "%bool %current %bound", "%bool %current %limit");
    failures += check("bound has additional store", source, 5);
    f = Fixture();
    source = replace(f.assembly(), "OpLoopMerge %merge %continue None\nOpBranch %condition\n%condition = OpLabel\n", "");
    source = replace(source, "OpBranchConditional %less %body %merge", "OpLoopMerge %merge %continue None\nOpBranchConditional %less %body %merge");
    failures += check("condition in loop header", source, 2);
    failures += check_specialization(true, 9, 2);
    failures += check_specialization(false, 9, 2);
    failures += check_specialization(true, 8, 5);
    failures += check_specialization(false, 8, 4);
    return failures ? 1 : 0;
}
