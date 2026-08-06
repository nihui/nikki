// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

struct CooperativeMatrixHelper
{
    CooperativeMatrixMarker marker;
    uint32_t return_type;
    std::vector<uint32_t> argument_types;
    uint32_t function_type;
    uint32_t function;
    uint32_t undef_value;
};

static void append_spirv_string_instruction(std::vector<uint32_t>& spirv, spv::Op opcode,
                                            uint32_t target, const std::string& value)
{
    const size_t string_word_count = (value.size() + 1 + 3) / 4;
    spirv.push_back(((uint32_t)(2 + string_word_count) << 16) | (uint32_t)opcode);
    spirv.push_back(target);
    const size_t string_begin = spirv.size();
    spirv.resize(string_begin + string_word_count, 0);
    memcpy(&spirv[string_begin], value.c_str(), value.size());
}

static bool spirv_constant_u32(const std::unordered_map<uint32_t, uint32_t>& constants,
                               uint32_t id, uint32_t& value)
{
    std::unordered_map<uint32_t, uint32_t>::const_iterator it = constants.find(id);
    if (it == constants.end())
        return false;

    value = it->second;
    return true;
}

static bool cooperative_matrix_helper_matches(const CooperativeMatrixHelper& helper,
                                              CooperativeMatrixMarker marker, uint32_t return_type,
                                              const std::vector<uint32_t>& argument_types)
{
    return helper.marker == marker && helper.return_type == return_type
           && helper.argument_types == argument_types;
}

static uint32_t find_or_create_cooperative_matrix_helper(
    std::vector<CooperativeMatrixHelper>& helpers, CooperativeMatrixMarker marker,
    uint32_t return_type, const std::vector<uint32_t>& argument_types, uint32_t& next_id)
{
    for (size_t i = 0; i < helpers.size(); i++)
    {
        if (cooperative_matrix_helper_matches(helpers[i], marker, return_type, argument_types))
            return helpers[i].function;
    }

    CooperativeMatrixHelper helper;
    helper.marker = marker;
    helper.return_type = return_type;
    helper.argument_types = argument_types;
    helper.function_type = next_id++;
    helper.function = next_id++;
    helper.undef_value = marker == CooperativeMatrixMarkerLoad
                                 || marker == CooperativeMatrixMarkerStore
                                 || marker == CooperativeMatrixMarkerConvert
                             ? next_id++
                             : 0;
    helpers.push_back(helper);
    return helper.function;
}

// Tint's SPIR-V reader does not support SPV_KHR_cooperative_matrix, while its
// core IR and WGSL writer support chromium_experimental_subgroup_matrix. Replace
// cooperative matrix types with named one-member structures and matrix builtins
// with named helper calls. The original types and builtins are restored in IR.
int legalize_cooperative_matrix_spirv(std::vector<uint32_t>& spirv,
                                      CooperativeMatrixLegalization& info)
{
    info = CooperativeMatrixLegalization();
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t original_bound = spirv[3];
    std::vector<uint32_t> result_types(original_bound, 0);
    std::vector<uint32_t> pointer_pointee_types(original_bound, 0);
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, bool> signed_integer_types;
    std::unordered_map<uint32_t, uint32_t> integer_type_widths;
    uint32_t integer32_types[2] = {0, 0};
    std::unordered_map<uint32_t, std::string> cooperative_matrix_names;
    std::unordered_map<uint32_t, uint32_t> cooperative_matrix_component_types;
    std::unordered_set<uint32_t> cooperative_matrix_types;
    std::unordered_set<uint32_t> cooperative_matrix_8bit_component_types;
    std::vector<CooperativeMatrixLegalization::ComponentTypeSubstitution> cooperative_matrix_32bit_integer_types;
    uint32_t void_type = 0;

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (spirv_instruction_has_result_type(opcode) && word_count >= 3 && spirv[i + 2] < original_bound)
            result_types[spirv[i + 2]] = spirv[i + 1];

        if (opcode == spv::Op::OpTypeVoid && word_count == 2)
            void_type = spirv[i + 1];
        else if (opcode == spv::Op::OpTypeInt && word_count == 4)
        {
            integer_type_widths[spirv[i + 1]] = spirv[i + 2];
            signed_integer_types[spirv[i + 1]] = spirv[i + 3] != 0;
            if (spirv[i + 2] == 32)
                integer32_types[spirv[i + 3] != 0 ? 1 : 0] = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpTypePointer && word_count == 4 && spirv[i + 1] < original_bound)
            pointer_pointee_types[spirv[i + 1]] = spirv[i + 3];
        else if (opcode == spv::Op::OpConstant && word_count == 4)
            constants[spirv[i + 2]] = spirv[i + 3];

        i += word_count;
    }

    bool uses_cooperative_matrix = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (opcode == spv::Op::OpTypeCooperativeMatrixKHR)
        {
            if (word_count != 7)
                return -1;

            uint32_t scope = 0;
            uint32_t rows = 0;
            uint32_t columns = 0;
            uint32_t use = 0;
            if (!spirv_constant_u32(constants, spirv[i + 3], scope)
                || !spirv_constant_u32(constants, spirv[i + 4], rows)
                || !spirv_constant_u32(constants, spirv[i + 5], columns)
                || !spirv_constant_u32(constants, spirv[i + 6], use)
                || scope != 3 || rows == 0 || columns == 0 || use > 2)
            {
                log_error("only subgroup cooperative matrix types with constant dimensions are supported");
                return -1;
            }

            uint32_t integer_width = 0;
            uint32_t integer_signed = 0;
            const std::unordered_map<uint32_t, uint32_t>::const_iterator width_it = integer_type_widths.find(spirv[i + 2]);
            if (width_it != integer_type_widths.end())
            {
                integer_width = width_it->second;
                integer_signed = signed_integer_types[spirv[i + 2]] ? 1 : 0;
            }

            char name[112];
            snprintf(name, sizeof(name), "nikki_coopmat_%u_%u_%u_%u_%u_%u",
                     use, rows, columns, spirv[i + 1], integer_width, integer_signed);
            cooperative_matrix_names[spirv[i + 1]] = name;
            cooperative_matrix_component_types[spirv[i + 1]] = spirv[i + 2];
            cooperative_matrix_types.insert(spirv[i + 1]);
            if (integer_width == 8)
            {
                cooperative_matrix_8bit_component_types.insert(spirv[i + 2]);
                CooperativeMatrixLegalization::ComponentTypeSubstitution substitution;
                substitution.use = use;
                substitution.rows = rows;
                substitution.columns = columns;
                substitution.is_signed = integer_signed != 0;
                bool duplicate = false;
                for (size_t j = 0; j < info.component_type_substitutions.size(); j++)
                {
                    const CooperativeMatrixLegalization::ComponentTypeSubstitution& existing = info.component_type_substitutions[j];
                    if (existing.use == substitution.use && existing.rows == substitution.rows
                        && existing.columns == substitution.columns && existing.is_signed == substitution.is_signed)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                    info.component_type_substitutions.push_back(substitution);
            }
            else if (integer_width == 32)
            {
                CooperativeMatrixLegalization::ComponentTypeSubstitution type;
                type.use = use;
                type.rows = rows;
                type.columns = columns;
                type.is_signed = integer_signed != 0;
                cooperative_matrix_32bit_integer_types.push_back(type);
            }
            uses_cooperative_matrix = true;
        }

        i += word_count;
    }

    if (!uses_cooperative_matrix)
        return 0;

    for (size_t i = 0; i < info.component_type_substitutions.size(); i++)
    {
        const CooperativeMatrixLegalization::ComponentTypeSubstitution& substitution = info.component_type_substitutions[i];
        for (size_t j = 0; j < cooperative_matrix_32bit_integer_types.size(); j++)
        {
            const CooperativeMatrixLegalization::ComponentTypeSubstitution& type = cooperative_matrix_32bit_integer_types[j];
            if (type.use == substitution.use && type.rows == substitution.rows
                && type.columns == substitution.columns && type.is_signed == substitution.is_signed)
            {
                log_error("8-bit and 32-bit cooperative matrix types have an ambiguous WGSL representation");
                return -1;
            }
        }
    }

    uint32_t next_id = original_bound;
    if (void_type == 0)
        void_type = next_id++;

    std::unordered_map<uint32_t, uint32_t> cooperative_matrix_component_placeholders;
    std::vector<uint32_t> created_integer32_types;
    for (std::unordered_set<uint32_t>::const_iterator it = cooperative_matrix_8bit_component_types.begin();
         it != cooperative_matrix_8bit_component_types.end(); ++it)
    {
        const bool is_signed = signed_integer_types[*it];
        uint32_t& placeholder = integer32_types[is_signed ? 1 : 0];
        if (placeholder == 0)
        {
            placeholder = next_id++;
            created_integer32_types.push_back(placeholder);
        }
        cooperative_matrix_component_placeholders[*it] = placeholder;
    }

    std::vector<CooperativeMatrixHelper> helpers;
    std::vector<uint32_t> instructions;
    instructions.reserve(spirv.size());

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);

        const bool cooperative_matrix_extension = opcode == spv::Op::OpExtension
                                                  && spirv_literal_string(&spirv[i + 1], word_count - 1) == "SPV_KHR_cooperative_matrix";
        const bool cooperative_matrix_capability = opcode == spv::Op::OpCapability && word_count == 2
                                                   && spirv[i + 1] == (uint32_t)spv::Capability::CooperativeMatrixKHR;
        const bool cooperative_matrix_int8_capability = opcode == spv::Op::OpCapability && word_count == 2
                                                        && spirv[i + 1] == (uint32_t)spv::Capability::Int8
                                                        && !cooperative_matrix_8bit_component_types.empty();

        if (cooperative_matrix_extension || cooperative_matrix_capability || cooperative_matrix_int8_capability)
        {
            i += word_count;
            continue;
        }

        if (opcode == spv::Op::OpTypeInt && word_count == 4
            && cooperative_matrix_8bit_component_types.count(spirv[i + 1]) != 0)
        {
            i += word_count;
            continue;
        }
        else if (opcode == spv::Op::OpName && word_count >= 3
                 && cooperative_matrix_8bit_component_types.count(spirv[i + 1]) != 0)
        {
            i += word_count;
            continue;
        }
        else if (opcode == spv::Op::OpTypeCooperativeMatrixKHR)
        {
            instructions.push_back((3u << 16) | (uint32_t)spv::Op::OpTypeStruct);
            instructions.push_back(spirv[i + 1]);
            const std::unordered_map<uint32_t, uint32_t>::const_iterator placeholder = cooperative_matrix_component_placeholders.find(spirv[i + 2]);
            instructions.push_back(placeholder == cooperative_matrix_component_placeholders.end()
                                       ? spirv[i + 2]
                                       : placeholder->second);
        }
        else if ((opcode == spv::Op::OpSConvert || opcode == spv::Op::OpUConvert)
                 && word_count == 4
                 && cooperative_matrix_component_placeholders.count(spirv[i + 1]) != 0)
        {
            const uint32_t placeholder = cooperative_matrix_component_placeholders[spirv[i + 1]];
            const uint32_t source_type = spirv[i + 3] < result_types.size() ? result_types[spirv[i + 3]] : 0;
            if (source_type == 0 || integer_type_widths[source_type] != 32)
                return -1;

            const spv::Op replacement_opcode = source_type == placeholder
                                                   ? spv::Op::OpCopyObject
                                                   : spv::Op::OpBitcast;
            instructions.push_back((4u << 16) | (uint32_t)replacement_opcode);
            instructions.push_back(placeholder);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(spirv[i + 3]);
        }
        else if ((opcode == spv::Op::OpConstant || opcode == spv::Op::OpSpecConstant)
                 && word_count == 4
                 && cooperative_matrix_component_placeholders.count(spirv[i + 1]) != 0)
        {
            const bool is_signed = signed_integer_types[spirv[i + 1]];
            uint32_t value = spirv[i + 3] & 0xffu;
            if (is_signed && (value & 0x80u) != 0)
                value |= 0xffffff00u;
            instructions.insert(instructions.end(), spirv.begin() + i, spirv.begin() + i + word_count);
            instructions[instructions.size() - 3] = cooperative_matrix_component_placeholders[spirv[i + 1]];
            instructions[instructions.size() - 1] = value;
        }
        else if ((opcode == spv::Op::OpConstantNull || opcode == spv::Op::OpUndef)
                 && word_count == 3
                 && cooperative_matrix_component_placeholders.count(spirv[i + 1]) != 0)
        {
            instructions.insert(instructions.end(), spirv.begin() + i, spirv.begin() + i + word_count);
            instructions[instructions.size() - 2] = cooperative_matrix_component_placeholders[spirv[i + 1]];
        }
        else if (opcode == spv::Op::OpCooperativeMatrixLoadKHR)
        {
            if (word_count < 6 || spirv[i + 3] >= result_types.size()
                || spirv[i + 4] >= result_types.size() || spirv[i + 5] >= result_types.size())
                return -1;

            const uint32_t pointer_type = result_types[spirv[i + 3]];
            const uint32_t component_type = pointer_type < pointer_pointee_types.size()
                                                ? pointer_pointee_types[pointer_type]
                                                : 0;
            const std::vector<uint32_t> argument_types = {
                component_type, result_types[spirv[i + 4]], result_types[spirv[i + 5]]};
            if (argument_types[0] == 0 || argument_types[1] == 0 || argument_types[2] == 0)
                return -1;
            const uint32_t helper = find_or_create_cooperative_matrix_helper(
                helpers, CooperativeMatrixMarkerLoad, spirv[i + 1], argument_types, next_id);
            const uint32_t loaded_component = next_id++;
            instructions.push_back((4u << 16) | (uint32_t)spv::Op::OpLoad);
            instructions.push_back(component_type);
            instructions.push_back(loaded_component);
            instructions.push_back(spirv[i + 3]);
            instructions.push_back((7u << 16) | (uint32_t)spv::Op::OpFunctionCall);
            instructions.push_back(spirv[i + 1]);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(helper);
            instructions.push_back(loaded_component);
            instructions.push_back(spirv[i + 4]);
            instructions.push_back(spirv[i + 5]);
            info.load_count++;
        }
        else if (opcode == spv::Op::OpCooperativeMatrixStoreKHR)
        {
            if (word_count < 5 || spirv[i + 1] >= result_types.size()
                || spirv[i + 2] >= result_types.size() || spirv[i + 3] >= result_types.size()
                || spirv[i + 4] >= result_types.size())
                return -1;

            const uint32_t pointer_type = result_types[spirv[i + 1]];
            const uint32_t component_type = pointer_type < pointer_pointee_types.size()
                                                ? pointer_pointee_types[pointer_type]
                                                : 0;
            const std::vector<uint32_t> argument_types = {
                result_types[spirv[i + 2]], result_types[spirv[i + 3]], result_types[spirv[i + 4]]};
            if (component_type == 0 || argument_types[0] == 0
                || argument_types[1] == 0 || argument_types[2] == 0)
                return -1;
            const uint32_t helper = find_or_create_cooperative_matrix_helper(
                helpers, CooperativeMatrixMarkerStore, component_type, argument_types, next_id);
            const uint32_t stored_component = next_id++;
            instructions.push_back((7u << 16) | (uint32_t)spv::Op::OpFunctionCall);
            instructions.push_back(component_type);
            instructions.push_back(stored_component);
            instructions.push_back(helper);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(spirv[i + 3]);
            instructions.push_back(spirv[i + 4]);
            instructions.push_back((3u << 16) | (uint32_t)spv::Op::OpStore);
            instructions.push_back(spirv[i + 1]);
            instructions.push_back(stored_component);
            info.store_count++;
        }
        else if (opcode == spv::Op::OpCooperativeMatrixMulAddKHR)
        {
            if ((word_count != 6 && word_count != 7) || spirv[i + 3] >= result_types.size()
                || spirv[i + 4] >= result_types.size() || spirv[i + 5] >= result_types.size())
                return -1;

            const uint32_t operand_flags = word_count == 7 ? spirv[i + 6] : 0;
            const uint32_t saturating_accumulation = 0x10;
            if ((operand_flags & saturating_accumulation) != 0 || (operand_flags & ~0x1fu) != 0)
            {
                log_error("cooperative matrix saturating or unknown mul-add operand flags are not supported");
                return -1;
            }

            const std::vector<uint32_t> argument_types = {
                result_types[spirv[i + 3]], result_types[spirv[i + 4]], result_types[spirv[i + 5]]};
            if (argument_types[0] == 0 || argument_types[1] == 0 || argument_types[2] == 0)
                return -1;

            const uint32_t matrix_types[4] = {
                argument_types[0], argument_types[1], argument_types[2], spirv[i + 1]};
            uint32_t expected_signed_flags = 0;
            for (uint32_t operand = 0; operand < 4; operand++)
            {
                const std::unordered_map<uint32_t, uint32_t>::const_iterator component_it = cooperative_matrix_component_types.find(matrix_types[operand]);
                if (component_it == cooperative_matrix_component_types.end())
                    return -1;

                const std::unordered_map<uint32_t, bool>::const_iterator signed_it = signed_integer_types.find(component_it->second);
                if (signed_it != signed_integer_types.end() && signed_it->second)
                    expected_signed_flags |= 1u << operand;
            }
            if ((operand_flags & 0xfu) != expected_signed_flags)
            {
                log_error("cooperative matrix mul-add signed operand flags do not match component types");
                return -1;
            }

            const uint32_t helper = find_or_create_cooperative_matrix_helper(
                helpers, CooperativeMatrixMarkerMulAdd, spirv[i + 1], argument_types, next_id);
            instructions.push_back((7u << 16) | (uint32_t)spv::Op::OpFunctionCall);
            instructions.push_back(spirv[i + 1]);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(helper);
            instructions.push_back(spirv[i + 3]);
            instructions.push_back(spirv[i + 4]);
            instructions.push_back(spirv[i + 5]);
            info.muladd_count++;
        }
        else if ((opcode == spv::Op::OpFAdd || opcode == spv::Op::OpIAdd
                  || opcode == spv::Op::OpFSub || opcode == spv::Op::OpISub
                  || opcode == spv::Op::OpFMul || opcode == spv::Op::OpIMul
                  || opcode == spv::Op::OpMatrixTimesScalar)
                 && word_count == 5
                 && cooperative_matrix_types.count(spirv[i + 1]) != 0)
        {
            if (spirv[i + 3] >= result_types.size() || spirv[i + 4] >= result_types.size())
                return -1;

            const bool lhs_is_matrix = cooperative_matrix_types.count(result_types[spirv[i + 3]]) != 0;
            const bool rhs_is_matrix = cooperative_matrix_types.count(result_types[spirv[i + 4]]) != 0;
            if (lhs_is_matrix == rhs_is_matrix)
            {
                log_error("cooperative matrix binary operation requires one matrix and one scalar operand");
                return -1;
            }
            if (!lhs_is_matrix && (opcode == spv::Op::OpFSub || opcode == spv::Op::OpISub))
            {
                log_error("scalar minus cooperative matrix is not supported by WebGPU");
                return -1;
            }

            CooperativeMatrixMarker marker;
            if (opcode == spv::Op::OpFAdd || opcode == spv::Op::OpIAdd)
                marker = CooperativeMatrixMarkerScalarAdd;
            else if (opcode == spv::Op::OpFSub || opcode == spv::Op::OpISub)
                marker = CooperativeMatrixMarkerScalarSubtract;
            else
                marker = CooperativeMatrixMarkerScalarMultiply;

            const uint32_t matrix_value = lhs_is_matrix ? spirv[i + 3] : spirv[i + 4];
            const uint32_t scalar_value = lhs_is_matrix ? spirv[i + 4] : spirv[i + 3];
            const std::vector<uint32_t> argument_types = {
                result_types[matrix_value], result_types[scalar_value]};
            if (argument_types[0] == 0 || argument_types[1] == 0)
                return -1;

            const uint32_t helper = find_or_create_cooperative_matrix_helper(
                helpers, marker, spirv[i + 1], argument_types, next_id);
            instructions.push_back((6u << 16) | (uint32_t)spv::Op::OpFunctionCall);
            instructions.push_back(spirv[i + 1]);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(helper);
            instructions.push_back(matrix_value);
            instructions.push_back(scalar_value);

            if (marker == CooperativeMatrixMarkerScalarAdd)
                info.scalar_add_count++;
            else if (marker == CooperativeMatrixMarkerScalarSubtract)
                info.scalar_subtract_count++;
            else
                info.scalar_multiply_count++;
        }
        else if (opcode == spv::Op::OpFConvert && word_count == 4
                 && cooperative_matrix_types.count(spirv[i + 1]) != 0)
        {
            if (spirv[i + 3] >= result_types.size()
                || cooperative_matrix_types.count(result_types[spirv[i + 3]]) == 0)
                return -1;

            const std::vector<uint32_t> argument_types = {result_types[spirv[i + 3]]};
            const uint32_t helper = find_or_create_cooperative_matrix_helper(
                helpers, CooperativeMatrixMarkerConvert, spirv[i + 1], argument_types, next_id);
            instructions.push_back((5u << 16) | (uint32_t)spv::Op::OpFunctionCall);
            instructions.push_back(spirv[i + 1]);
            instructions.push_back(spirv[i + 2]);
            instructions.push_back(helper);
            instructions.push_back(spirv[i + 3]);
            info.convert_count++;
        }
        else
        {
            instructions.insert(instructions.end(), spirv.begin() + i, spirv.begin() + i + word_count);
        }

        i += word_count;
    }

    size_t debug_insert = instructions.size();
    size_t declaration_insert = instructions.size();
    for (size_t i = 0; i < instructions.size();)
    {
        const uint32_t word_count = instructions[i] >> 16;
        const spv::Op opcode = (spv::Op)(instructions[i] & 0xffff);
        if (debug_insert == instructions.size()
            && (opcode == spv::Op::OpDecorate || opcode == spv::Op::OpMemberDecorate
                || (opcode >= spv::Op::OpTypeVoid && opcode <= spv::Op::OpTypeForwardPointer)))
            debug_insert = i;
        if (opcode == spv::Op::OpFunction)
        {
            declaration_insert = i;
            break;
        }
        i += word_count;
    }
    if (debug_insert == instructions.size())
        debug_insert = declaration_insert;

    std::vector<uint32_t> names;
    for (std::unordered_map<uint32_t, std::string>::const_iterator it = cooperative_matrix_names.begin();
         it != cooperative_matrix_names.end(); ++it)
        append_spirv_string_instruction(names, spv::Op::OpName, it->first, it->second);
    for (size_t i = 0; i < helpers.size(); i++)
    {
        const char* marker_name = "muladd";
        if (helpers[i].marker == CooperativeMatrixMarkerLoad)
            marker_name = "load";
        else if (helpers[i].marker == CooperativeMatrixMarkerStore)
            marker_name = "store";
        else if (helpers[i].marker == CooperativeMatrixMarkerConvert)
            marker_name = "convert";
        else if (helpers[i].marker == CooperativeMatrixMarkerScalarAdd)
            marker_name = "scalar_add";
        else if (helpers[i].marker == CooperativeMatrixMarkerScalarSubtract)
            marker_name = "scalar_subtract";
        else if (helpers[i].marker == CooperativeMatrixMarkerScalarMultiply)
            marker_name = "scalar_multiply";
        char name[96];
        snprintf(name, sizeof(name), "nikki_coop_%s_%u", marker_name, helpers[i].function);
        append_spirv_string_instruction(names, spv::Op::OpName, helpers[i].function, name);
    }

    std::vector<uint32_t> declarations;
    if (void_type >= original_bound)
    {
        declarations.push_back((2u << 16) | (uint32_t)spv::Op::OpTypeVoid);
        declarations.push_back(void_type);
    }
    for (size_t i = 0; i < created_integer32_types.size(); i++)
    {
        const uint32_t type = created_integer32_types[i];
        declarations.push_back((4u << 16) | (uint32_t)spv::Op::OpTypeInt);
        declarations.push_back(type);
        declarations.push_back(32);
        declarations.push_back(type == integer32_types[1] ? 1 : 0);
    }
    for (size_t i = 0; i < helpers.size(); i++)
    {
        CooperativeMatrixHelper& helper = helpers[i];
        declarations.push_back(((uint32_t)(3 + helper.argument_types.size()) << 16)
                               | (uint32_t)spv::Op::OpTypeFunction);
        declarations.push_back(helper.function_type);
        declarations.push_back(helper.return_type);
        declarations.insert(declarations.end(), helper.argument_types.begin(), helper.argument_types.end());
        if (helper.undef_value != 0)
        {
            declarations.push_back((3u << 16) | (uint32_t)spv::Op::OpUndef);
            declarations.push_back(helper.return_type);
            declarations.push_back(helper.undef_value);
        }
    }

    std::vector<uint32_t> helper_functions;
    for (size_t i = 0; i < helpers.size(); i++)
    {
        CooperativeMatrixHelper& helper = helpers[i];
        helper_functions.push_back((5u << 16) | (uint32_t)spv::Op::OpFunction);
        helper_functions.push_back(helper.return_type);
        helper_functions.push_back(helper.function);
        helper_functions.push_back((uint32_t)spv::FunctionControlMask::MaskNone);
        helper_functions.push_back(helper.function_type);

        std::vector<uint32_t> parameters;
        for (size_t j = 0; j < helper.argument_types.size(); j++)
        {
            const uint32_t parameter = next_id++;
            parameters.push_back(parameter);
            helper_functions.push_back((3u << 16) | (uint32_t)spv::Op::OpFunctionParameter);
            helper_functions.push_back(helper.argument_types[j]);
            helper_functions.push_back(parameter);
        }
        helper_functions.push_back((2u << 16) | (uint32_t)spv::Op::OpLabel);
        helper_functions.push_back(next_id++);
        helper_functions.push_back((2u << 16) | (uint32_t)spv::Op::OpReturnValue);
        if (helper.undef_value != 0)
            helper_functions.push_back(helper.undef_value);
        else if (helper.marker == CooperativeMatrixMarkerMulAdd)
            helper_functions.push_back(parameters.back());
        else
            helper_functions.push_back(parameters.front());
        helper_functions.push_back((1u << 16) | (uint32_t)spv::Op::OpFunctionEnd);
    }

    std::vector<uint32_t> legalized;
    legalized.reserve(5 + instructions.size() + names.size() + declarations.size() + helper_functions.size());
    legalized.insert(legalized.end(), spirv.begin(), spirv.begin() + 5);
    legalized.insert(legalized.end(), instructions.begin(), instructions.begin() + debug_insert);
    legalized.insert(legalized.end(), names.begin(), names.end());
    legalized.insert(legalized.end(), instructions.begin() + debug_insert, instructions.begin() + declaration_insert);
    legalized.insert(legalized.end(), declarations.begin(), declarations.end());
    legalized.insert(legalized.end(), instructions.begin() + declaration_insert, instructions.end());
    legalized.insert(legalized.end(), helper_functions.begin(), helper_functions.end());
    legalized[3] = next_id;

    spirv.swap(legalized);
    return 0;
}

} // namespace nikki_detail
