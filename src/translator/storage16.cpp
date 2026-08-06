// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

namespace {

enum SpirvTypeKind
{
    SPIRV_TYPE_UNKNOWN,
    SPIRV_TYPE_INT,
    SPIRV_TYPE_VECTOR,
    SPIRV_TYPE_RUNTIME_ARRAY,
    SPIRV_TYPE_STRUCT,
    SPIRV_TYPE_POINTER
};

struct SpirvType
{
    SpirvTypeKind kind;
    uint32_t width;
    bool is_signed;
    uint32_t element_type;
    uint32_t element_count;
    spv::StorageClass storage_class;
    std::vector<uint32_t> members;

    SpirvType()
        : kind(SPIRV_TYPE_UNKNOWN), width(0), is_signed(false), element_type(0), element_count(0), storage_class(spv::StorageClass::Max)
    {
    }
};

struct Storage16Buffer
{
    uint32_t variable_id;
    uint32_t array_type;
    uint32_t element_type;
    uint32_t scalar_type;
    uint32_t component_count;
    bool is_signed;
    bool writable;

    Storage16Buffer()
        : variable_id(0), array_type(0), element_type(0), scalar_type(0), component_count(0), is_signed(false), writable(false)
    {
    }
};

struct Storage16Access
{
    size_t buffer_index;
    uint32_t index_id;
    uint32_t index_type;
    uint32_t normalized_index_id;

    Storage16Access()
        : buffer_index(0), index_id(0), index_type(0), normalized_index_id(0)
    {
    }
};

static void append_instruction(std::vector<uint32_t>& output, spv::Op opcode,
                               std::initializer_list<uint32_t> operands)
{
    output.push_back(((uint32_t)(operands.size() + 1) << 16) | (uint32_t)opcode);
    output.insert(output.end(), operands.begin(), operands.end());
}

static bool is_type_instruction(spv::Op opcode)
{
    return opcode >= spv::Op::OpTypeVoid && opcode <= spv::Op::OpTypeForwardPointer;
}

static bool is_integer_type(const std::vector<SpirvType>& types, uint32_t type_id,
                            uint32_t width)
{
    return type_id < types.size() && types[type_id].kind == SPIRV_TYPE_INT
           && types[type_id].width == width;
}

static bool integer_element_type(const std::vector<SpirvType>& types, uint32_t type_id,
                                 uint32_t& scalar_type, uint32_t& component_count)
{
    if (is_integer_type(types, type_id, 16))
    {
        scalar_type = type_id;
        component_count = 1;
        return true;
    }

    if (type_id >= types.size() || types[type_id].kind != SPIRV_TYPE_VECTOR)
        return false;

    const SpirvType& vector_type = types[type_id];
    if (!is_integer_type(types, vector_type.element_type, 16)
        || vector_type.element_count < 2 || vector_type.element_count > 4)
        return false;

    scalar_type = vector_type.element_type;
    component_count = vector_type.element_count;
    return true;
}

static uint32_t find_constant_u32(const std::unordered_map<uint32_t, uint32_t>& constants,
                                  uint32_t id, bool& found)
{
    std::unordered_map<uint32_t, uint32_t>::const_iterator it = constants.find(id);
    found = it != constants.end();
    return found ? it->second : 0;
}

struct Storage16Legalizer
{
    std::vector<uint32_t>& spirv;
    uint32_t original_bound;
    uint32_t next_id;
    std::vector<SpirvType> types;
    std::vector<size_t> type_offsets;
    std::vector<uint32_t> value_types;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_set<uint32_t> buffer_block_types;
    std::vector<Storage16Buffer> buffers;
    std::unordered_map<uint32_t, size_t> variable_to_buffer;
    std::unordered_map<uint32_t, Storage16Access> accesses;
    std::unordered_map<uint32_t, uint32_t> packed_element_types;
    std::unordered_set<uint32_t> moved_helper_types;

    uint32_t helper_u32;
    uint32_t helper_i32;
    uint32_t helper_v2u32;
    uint32_t helper_v2i32;
    uint32_t helper_v3u32;
    uint32_t helper_v3i32;
    uint32_t helper_v4u32;
    uint32_t helper_v4i32;
    uint32_t constant_u32_0;
    uint32_t constant_u32_1;
    uint32_t constant_u32_4;
    uint32_t constant_u32_16;
    uint32_t constant_u32_ffff;
    uint32_t constant_i32_16;
    size_t type_insert_offset;
    size_t function_insert_offset;

    explicit Storage16Legalizer(std::vector<uint32_t>& words)
        : spirv(words), original_bound(words.size() >= 4 ? words[3] : 0), next_id(original_bound), types(original_bound), type_offsets(original_bound, std::numeric_limits<size_t>::max()), value_types(original_bound, 0), helper_u32(0), helper_i32(0), helper_v2u32(0), helper_v2i32(0), helper_v3u32(0), helper_v3i32(0), helper_v4u32(0), helper_v4i32(0), constant_u32_0(0), constant_u32_1(0), constant_u32_4(0), constant_u32_16(0), constant_u32_ffff(0), constant_i32_16(0), type_insert_offset(0), function_insert_offset(0)
    {
    }

    uint32_t allocate_id()
    {
        return next_id++;
    }

    int parse()
    {
        if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber || original_bound == 0)
            return -1;

        function_insert_offset = spirv.size();
        for (size_t i = 5; i < spirv.size();)
        {
            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            if (word_count == 0 || i + word_count > spirv.size())
                return -1;

            if (opcode == spv::Op::OpFunction && function_insert_offset == spirv.size())
                function_insert_offset = i;

            if (opcode == spv::Op::OpTypeInt && word_count == 4 && spirv[i + 1] < original_bound)
            {
                SpirvType& type = types[spirv[i + 1]];
                type_offsets[spirv[i + 1]] = i;
                type.kind = SPIRV_TYPE_INT;
                type.width = spirv[i + 2];
                type.is_signed = spirv[i + 3] != 0;
            }
            else if (opcode == spv::Op::OpTypeVector && word_count == 4 && spirv[i + 1] < original_bound)
            {
                SpirvType& type = types[spirv[i + 1]];
                type_offsets[spirv[i + 1]] = i;
                type.kind = SPIRV_TYPE_VECTOR;
                type.element_type = spirv[i + 2];
                type.element_count = spirv[i + 3];
            }
            else if (opcode == spv::Op::OpTypeRuntimeArray && word_count == 3 && spirv[i + 1] < original_bound)
            {
                SpirvType& type = types[spirv[i + 1]];
                type_offsets[spirv[i + 1]] = i;
                type.kind = SPIRV_TYPE_RUNTIME_ARRAY;
                type.element_type = spirv[i + 2];
            }
            else if (opcode == spv::Op::OpTypeStruct && word_count >= 2 && spirv[i + 1] < original_bound)
            {
                SpirvType& type = types[spirv[i + 1]];
                type_offsets[spirv[i + 1]] = i;
                type.kind = SPIRV_TYPE_STRUCT;
                type.members.assign(spirv.begin() + i + 2, spirv.begin() + i + word_count);
            }
            else if (opcode == spv::Op::OpTypePointer && word_count == 4 && spirv[i + 1] < original_bound)
            {
                SpirvType& type = types[spirv[i + 1]];
                type_offsets[spirv[i + 1]] = i;
                type.kind = SPIRV_TYPE_POINTER;
                type.storage_class = (spv::StorageClass)spirv[i + 2];
                type.element_type = spirv[i + 3];
            }
            else if (opcode == spv::Op::OpDecorate && word_count == 3
                     && spirv[i + 2] == (uint32_t)spv::Decoration::BufferBlock)
            {
                buffer_block_types.insert(spirv[i + 1]);
            }

            const bool function_result_type = i >= function_insert_offset && word_count >= 3
                                              && spirv[i + 1] < original_bound
                                              && types[spirv[i + 1]].kind != SPIRV_TYPE_UNKNOWN;
            if ((spirv_instruction_has_result_type(opcode) || function_result_type)
                && word_count >= 3 && spirv[i + 2] < original_bound)
            {
                value_types[spirv[i + 2]] = spirv[i + 1];
            }

            if (opcode == spv::Op::OpConstant && word_count == 4
                && spirv[i + 2] < original_bound)
            {
                value_types[spirv[i + 2]] = spirv[i + 1];
                if (is_integer_type(types, spirv[i + 1], 32))
                    constants[spirv[i + 2]] = spirv[i + 3];
            }

            i += word_count;
        }

        if (function_insert_offset == spirv.size())
            return -1;

        return find_buffers();
    }

    int find_buffers()
    {
        for (size_t i = 5; i < function_insert_offset;)
        {
            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            if (opcode == spv::Op::OpVariable && word_count >= 4
                && spirv[i + 2] < original_bound
                && (spirv[i + 3] == (uint32_t)spv::StorageClass::StorageBuffer
                    || spirv[i + 3] == (uint32_t)spv::StorageClass::Uniform))
            {
                const uint32_t pointer_type_id = spirv[i + 1];
                if (pointer_type_id >= types.size()
                    || types[pointer_type_id].kind != SPIRV_TYPE_POINTER)
                    return -1;

                const uint32_t struct_type_id = types[pointer_type_id].element_type;
                if (struct_type_id >= types.size()
                    || types[struct_type_id].kind != SPIRV_TYPE_STRUCT)
                    return -1;

                if (spirv[i + 3] == (uint32_t)spv::StorageClass::Uniform
                    && buffer_block_types.count(struct_type_id) == 0)
                {
                    i += word_count;
                    continue;
                }

                const SpirvType& struct_type = types[struct_type_id];
                if (struct_type.members.size() == 1)
                {
                    const uint32_t array_type_id = struct_type.members[0];
                    if (array_type_id < types.size()
                        && types[array_type_id].kind == SPIRV_TYPE_RUNTIME_ARRAY)
                    {
                        const uint32_t element_type_id = types[array_type_id].element_type;
                        uint32_t scalar_type_id = 0;
                        uint32_t component_count = 0;
                        if (integer_element_type(types, element_type_id, scalar_type_id, component_count))
                        {
                            Storage16Buffer buffer;
                            buffer.variable_id = spirv[i + 2];
                            buffer.array_type = array_type_id;
                            buffer.element_type = element_type_id;
                            buffer.scalar_type = scalar_type_id;
                            buffer.component_count = component_count;
                            buffer.is_signed = types[scalar_type_id].is_signed;
                            variable_to_buffer[buffer.variable_id] = buffers.size();
                            buffers.push_back(buffer);
                        }
                    }
                }
            }

            i += word_count;
        }

        if (buffers.empty())
            return 0;

        type_insert_offset = function_insert_offset;
        for (size_t i = 5; i < function_insert_offset;)
        {
            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            if (opcode == spv::Op::OpTypeRuntimeArray)
            {
                for (size_t j = 0; j < buffers.size(); j++)
                {
                    if (spirv[i + 1] == buffers[j].array_type)
                        type_insert_offset = std::min(type_insert_offset, i);
                }
            }
            i += word_count;
        }
        if (type_insert_offset == function_insert_offset)
            return -1;

        for (size_t i = 5; i < spirv.size();)
        {
            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
                && word_count >= 5)
            {
                std::unordered_map<uint32_t, size_t>::const_iterator it = variable_to_buffer.find(spirv[i + 3]);
                if (it != variable_to_buffer.end())
                {
                    if (word_count != 6)
                    {
                        log_error("16-bit storage buffer %u uses an unsupported access-chain shape", spirv[i + 3]);
                        return -1;
                    }

                    bool found = false;
                    const uint32_t member = find_constant_u32(constants, spirv[i + 4], found);
                    if (!found || member != 0 || spirv[i + 2] >= original_bound
                        || spirv[i + 5] >= original_bound)
                    {
                        log_error("16-bit storage buffer %u must access runtime-array member zero", spirv[i + 3]);
                        return -1;
                    }

                    Storage16Access access;
                    access.buffer_index = it->second;
                    access.index_id = spirv[i + 5];
                    access.index_type = value_types[access.index_id];
                    if (!is_integer_type(types, access.index_type, 32))
                    {
                        log_error("16-bit storage buffer %u has a non-32-bit array index", spirv[i + 3]);
                        return -1;
                    }
                    accesses[spirv[i + 2]] = access;
                }
            }
            else if (opcode == spv::Op::OpStore && word_count == 3)
            {
                std::unordered_map<uint32_t, Storage16Access>::const_iterator it = accesses.find(spirv[i + 1]);
                if (it != accesses.end())
                    buffers[it->second.buffer_index].writable = true;
            }

            i += word_count;
        }

        return validate_access_uses();
    }

    int validate_access_uses()
    {
        for (size_t i = function_insert_offset; i < spirv.size();)
        {
            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            for (uint32_t j = 1; j < word_count; j++)
            {
                std::unordered_map<uint32_t, Storage16Access>::const_iterator it = accesses.find(spirv[i + j]);
                if (it == accesses.end())
                    continue;

                const bool access_result = (opcode == spv::Op::OpAccessChain
                                            || opcode == spv::Op::OpInBoundsAccessChain)
                                           && j == 2;
                const bool load_pointer = opcode == spv::Op::OpLoad && j == 3;
                const bool store_pointer = opcode == spv::Op::OpStore && j == 1;
                if (!access_result && !load_pointer && !store_pointer)
                {
                    log_error("16-bit storage access %u is used by unsupported SPIR-V opcode %u",
                              spirv[i + j], (uint32_t)opcode);
                    return -1;
                }
            }
            i += word_count;
        }

        return 0;
    }

    uint32_t mapped_type(uint32_t type_id) const
    {
        if (type_id >= types.size())
            return type_id;

        uint32_t scalar_type = 0;
        uint32_t component_count = 0;
        if (!integer_element_type(types, type_id, scalar_type, component_count))
            return type_id;

        const bool is_signed = types[scalar_type].is_signed;
        if (component_count == 1)
            return is_signed ? helper_i32 : helper_u32;
        if (component_count == 2)
            return is_signed ? helper_v2i32 : helper_v2u32;
        if (component_count == 3)
            return is_signed ? helper_v3i32 : helper_v3u32;
        return is_signed ? helper_v4i32 : helper_v4u32;
    }

    uint32_t packed_type(uint32_t component_count) const
    {
        return component_count <= 2 ? helper_u32 : helper_v2u32;
    }

    uint32_t scalar_mapped_type(uint32_t original_type) const
    {
        return types[original_type].is_signed ? helper_i32 : helper_u32;
    }

    uint32_t component_count(uint32_t type_id) const
    {
        if (type_id < types.size() && types[type_id].kind == SPIRV_TYPE_VECTOR)
            return types[type_id].element_count;
        return 1;
    }

    uint32_t scalar_original_type(uint32_t type_id) const
    {
        if (type_id < types.size() && types[type_id].kind == SPIRV_TYPE_VECTOR)
            return types[type_id].element_type;
        return type_id;
    }

    uint32_t emit_bitcast_u32(std::vector<uint32_t>& output, uint32_t value,
                              uint32_t original_scalar_type)
    {
        if (!types[original_scalar_type].is_signed)
            return value;

        const uint32_t result = allocate_id();
        append_instruction(output, spv::Op::OpBitcast, {helper_u32, result, value});
        return result;
    }

    uint32_t emit_low_16(std::vector<uint32_t>& output, uint32_t bits,
                         bool signed_result, uint32_t result_id)
    {
        if (!signed_result)
        {
            append_instruction(output, spv::Op::OpBitwiseAnd,
                               {helper_u32, result_id, bits, constant_u32_ffff});
            return result_id;
        }

        const uint32_t shifted = allocate_id();
        const uint32_t signed_bits = allocate_id();
        append_instruction(output, spv::Op::OpShiftLeftLogical,
                           {helper_u32, shifted, bits, constant_u32_16});
        append_instruction(output, spv::Op::OpBitcast,
                           {helper_i32, signed_bits, shifted});
        append_instruction(output, spv::Op::OpShiftRightArithmetic,
                           {helper_i32, result_id, signed_bits, constant_i32_16});
        return result_id;
    }

    uint32_t emit_component(std::vector<uint32_t>& output, uint32_t vector_value,
                            uint32_t original_vector_type, uint32_t component)
    {
        const uint32_t scalar_type = scalar_original_type(original_vector_type);
        const uint32_t result = allocate_id();
        append_instruction(output, spv::Op::OpCompositeExtract,
                           {scalar_mapped_type(scalar_type), result, vector_value, component});
        return result;
    }

    uint32_t emit_packed_word(std::vector<uint32_t>& output, uint32_t vector_value,
                              uint32_t original_vector_type, uint32_t first_component)
    {
        const uint32_t scalar_type = scalar_original_type(original_vector_type);
        const uint32_t count = component_count(original_vector_type);
        uint32_t low = emit_component(output, vector_value, original_vector_type, first_component);
        low = emit_bitcast_u32(output, low, scalar_type);
        const uint32_t low_masked = allocate_id();
        append_instruction(output, spv::Op::OpBitwiseAnd,
                           {helper_u32, low_masked, low, constant_u32_ffff});
        if (first_component + 1 >= count)
            return low_masked;

        uint32_t high = emit_component(output, vector_value, original_vector_type, first_component + 1);
        high = emit_bitcast_u32(output, high, scalar_type);
        const uint32_t high_masked = allocate_id();
        const uint32_t high_shifted = allocate_id();
        const uint32_t packed = allocate_id();
        append_instruction(output, spv::Op::OpBitwiseAnd,
                           {helper_u32, high_masked, high, constant_u32_ffff});
        append_instruction(output, spv::Op::OpShiftLeftLogical,
                           {helper_u32, high_shifted, high_masked, constant_u32_16});
        append_instruction(output, spv::Op::OpBitwiseOr,
                           {helper_u32, packed, low_masked, high_shifted});
        return packed;
    }

    uint32_t emit_unpacked_component(std::vector<uint32_t>& output, uint32_t word,
                                     uint32_t half, bool is_signed, uint32_t result_id)
    {
        uint32_t bits = word;
        if (half != 0)
        {
            bits = allocate_id();
            append_instruction(output, spv::Op::OpShiftRightLogical,
                               {helper_u32, bits, word, constant_u32_16});
        }
        return emit_low_16(output, bits, is_signed, result_id);
    }

    bool integer_32_type(uint32_t type_id, uint32_t& scalar_type,
                         uint32_t& component_count_value) const
    {
        if (is_integer_type(types, type_id, 32))
        {
            scalar_type = type_id;
            component_count_value = 1;
            return true;
        }

        if (type_id >= types.size() || types[type_id].kind != SPIRV_TYPE_VECTOR)
            return false;

        const SpirvType& vector_type = types[type_id];
        if (!is_integer_type(types, vector_type.element_type, 32))
            return false;

        scalar_type = vector_type.element_type;
        component_count_value = vector_type.element_count;
        return true;
    }

    uint32_t emit_word_component(std::vector<uint32_t>& output, uint32_t value,
                                 uint32_t type_id, uint32_t component)
    {
        if (types[type_id].kind != SPIRV_TYPE_VECTOR)
        {
            if (types[type_id].is_signed)
            {
                const uint32_t result = allocate_id();
                append_instruction(output, spv::Op::OpBitcast,
                                   {helper_u32, result, value});
                return result;
            }
            return value;
        }

        const uint32_t extracted = allocate_id();
        append_instruction(output, spv::Op::OpCompositeExtract,
                           {types[type_id].element_type, extracted, value, component});
        if (!types[types[type_id].element_type].is_signed)
            return extracted;

        const uint32_t result = allocate_id();
        append_instruction(output, spv::Op::OpBitcast,
                           {helper_u32, result, extracted});
        return result;
    }

    int emit_bitcast(std::vector<uint32_t>& output, const uint32_t* instruction)
    {
        const uint32_t destination_type = instruction[1];
        const uint32_t result_id = instruction[2];
        const uint32_t source = instruction[3];
        const uint32_t source_type = source < value_types.size() ? value_types[source] : 0;

        uint32_t source_16_scalar = 0;
        uint32_t source_16_count = 0;
        uint32_t destination_16_scalar = 0;
        uint32_t destination_16_count = 0;
        const bool source_is_16 = integer_element_type(types, source_type,
                                                       source_16_scalar, source_16_count);
        const bool destination_is_16 = integer_element_type(types, destination_type,
                                                            destination_16_scalar, destination_16_count);

        if (source_is_16 == destination_is_16)
            return -1;

        if (destination_is_16)
        {
            uint32_t source_32_scalar = 0;
            uint32_t source_32_count = 0;
            if (!integer_32_type(source_type, source_32_scalar, source_32_count)
                || source_32_count * 2 != destination_16_count)
                return -1;

            std::vector<uint32_t> components;
            components.reserve(destination_16_count);
            for (uint32_t i = 0; i < destination_16_count; i++)
            {
                const uint32_t word = emit_word_component(output, source, source_type, i / 2);
                const uint32_t component = allocate_id();
                emit_unpacked_component(output, word, i % 2,
                                        types[destination_16_scalar].is_signed, component);
                components.push_back(component);
            }

            output.push_back(((uint32_t)(components.size() + 3) << 16)
                             | (uint32_t)spv::Op::OpCompositeConstruct);
            output.push_back(mapped_type(destination_type));
            output.push_back(result_id);
            output.insert(output.end(), components.begin(), components.end());
            return 0;
        }

        uint32_t destination_32_scalar = 0;
        uint32_t destination_32_count = 0;
        if (!integer_32_type(destination_type, destination_32_scalar, destination_32_count)
            || source_16_count != destination_32_count * 2)
            return -1;

        if (destination_32_count == 1)
        {
            const uint32_t packed = emit_packed_word(output, source, source_type, 0);
            if (types[destination_32_scalar].is_signed)
                append_instruction(output, spv::Op::OpBitcast,
                                   {destination_type, result_id, packed});
            else
                append_instruction(output, spv::Op::OpCopyObject,
                                   {destination_type, result_id, packed});
            return 0;
        }

        std::vector<uint32_t> words;
        words.reserve(destination_32_count);
        for (uint32_t i = 0; i < destination_32_count; i++)
            words.push_back(emit_packed_word(output, source, source_type, i * 2));

        const uint32_t packed_vector = types[destination_32_scalar].is_signed
                                           ? allocate_id()
                                           : result_id;
        output.push_back(((uint32_t)(words.size() + 3) << 16)
                         | (uint32_t)spv::Op::OpCompositeConstruct);
        output.push_back(helper_v2u32);
        output.push_back(packed_vector);
        output.insert(output.end(), words.begin(), words.end());
        if (types[destination_32_scalar].is_signed)
            append_instruction(output, spv::Op::OpBitcast,
                               {destination_type, result_id, packed_vector});
        return 0;
    }

    int emit_load(std::vector<uint32_t>& output, const uint32_t* instruction,
                  const Storage16Access& access)
    {
        const Storage16Buffer& buffer = buffers[access.buffer_index];
        const uint32_t result_id = instruction[2];
        const uint32_t pointer_id = instruction[3];

        if (buffer.component_count == 1)
        {
            const uint32_t word = allocate_id();
            if (buffer.writable)
            {
                append_instruction(output, spv::Op::OpAtomicLoad,
                                   {helper_u32, word, pointer_id, constant_u32_1, constant_u32_0});
            }
            else
            {
                append_instruction(output, spv::Op::OpLoad, {helper_u32, word, pointer_id});
            }

            const uint32_t parity = allocate_id();
            const uint32_t shift = allocate_id();
            const uint32_t shifted = allocate_id();
            append_instruction(output, spv::Op::OpBitwiseAnd,
                               {helper_u32, parity, access.normalized_index_id, constant_u32_1});
            append_instruction(output, spv::Op::OpShiftLeftLogical,
                               {helper_u32, shift, parity, constant_u32_4});
            append_instruction(output, spv::Op::OpShiftRightLogical,
                               {helper_u32, shifted, word, shift});
            emit_low_16(output, shifted, buffer.is_signed, result_id);
            return 0;
        }

        const uint32_t packed = allocate_id();
        append_instruction(output, spv::Op::OpLoad,
                           {packed_type(buffer.component_count), packed, pointer_id});

        std::vector<uint32_t> components;
        components.reserve(buffer.component_count);
        for (uint32_t i = 0; i < buffer.component_count; i++)
        {
            uint32_t word = packed;
            if (buffer.component_count > 2)
            {
                word = allocate_id();
                append_instruction(output, spv::Op::OpCompositeExtract,
                                   {helper_u32, word, packed, i / 2});
            }

            const uint32_t component = allocate_id();
            emit_unpacked_component(output, word, i % 2, buffer.is_signed, component);
            components.push_back(component);
        }

        output.push_back(((uint32_t)(components.size() + 3) << 16)
                         | (uint32_t)spv::Op::OpCompositeConstruct);
        output.push_back(mapped_type(buffer.element_type));
        output.push_back(result_id);
        output.insert(output.end(), components.begin(), components.end());
        return 0;
    }

    int emit_store(std::vector<uint32_t>& output, const uint32_t* instruction,
                   const Storage16Access& access)
    {
        const Storage16Buffer& buffer = buffers[access.buffer_index];
        const uint32_t pointer_id = instruction[1];
        const uint32_t value_id = instruction[2];

        if (buffer.component_count == 1)
        {
            uint32_t bits = emit_bitcast_u32(output, value_id, buffer.scalar_type);
            const uint32_t masked = allocate_id();
            const uint32_t parity = allocate_id();
            const uint32_t shift = allocate_id();
            const uint32_t shifted_mask = allocate_id();
            const uint32_t shifted_value = allocate_id();
            const uint32_t preserve_mask = allocate_id();
            const uint32_t discard_and = allocate_id();
            const uint32_t discard_or = allocate_id();
            append_instruction(output, spv::Op::OpBitwiseAnd,
                               {helper_u32, masked, bits, constant_u32_ffff});
            append_instruction(output, spv::Op::OpBitwiseAnd,
                               {helper_u32, parity, access.normalized_index_id, constant_u32_1});
            append_instruction(output, spv::Op::OpShiftLeftLogical,
                               {helper_u32, shift, parity, constant_u32_4});
            append_instruction(output, spv::Op::OpShiftLeftLogical,
                               {helper_u32, shifted_mask, constant_u32_ffff, shift});
            append_instruction(output, spv::Op::OpShiftLeftLogical,
                               {helper_u32, shifted_value, masked, shift});
            append_instruction(output, spv::Op::OpNot,
                               {helper_u32, preserve_mask, shifted_mask});
            append_instruction(output, spv::Op::OpAtomicAnd,
                               {helper_u32, discard_and, pointer_id, constant_u32_1,
                                constant_u32_0, preserve_mask});
            append_instruction(output, spv::Op::OpAtomicOr,
                               {helper_u32, discard_or, pointer_id, constant_u32_1,
                                constant_u32_0, shifted_value});
            return 0;
        }

        const uint32_t first_word = emit_packed_word(output, value_id, buffer.element_type, 0);
        if (buffer.component_count == 2)
        {
            append_instruction(output, spv::Op::OpStore, {pointer_id, first_word});
            return 0;
        }

        const uint32_t second_word = emit_packed_word(output, value_id, buffer.element_type, 2);
        const uint32_t packed = allocate_id();
        append_instruction(output, spv::Op::OpCompositeConstruct,
                           {helper_v2u32, packed, first_word, second_word});
        append_instruction(output, spv::Op::OpStore, {pointer_id, packed});
        return 0;
    }

    uint32_t emit_scalar_conversion(std::vector<uint32_t>& output, spv::Op opcode,
                                    uint32_t source, uint32_t source_type,
                                    uint32_t destination_type, uint32_t result_id)
    {
        const bool source_is_16 = is_integer_type(types, source_type, 16);
        const bool destination_is_16 = is_integer_type(types, destination_type, 16);
        uint32_t bits = source;
        if (types[source_type].is_signed)
            bits = emit_bitcast_u32(output, source, source_type);

        if (destination_is_16)
            return emit_low_16(output, bits, types[destination_type].is_signed, result_id);

        uint32_t converted = allocate_id();
        if (opcode == spv::Op::OpSConvert && source_is_16)
            emit_low_16(output, bits, true, converted);
        else
            emit_low_16(output, bits, false, converted);

        const bool converted_is_signed = opcode == spv::Op::OpSConvert && source_is_16;
        if (converted_is_signed == types[destination_type].is_signed)
        {
            append_instruction(output, spv::Op::OpCopyObject,
                               {destination_type, result_id, converted});
        }
        else
        {
            append_instruction(output, spv::Op::OpBitcast,
                               {destination_type, result_id, converted});
        }
        return result_id;
    }

    int emit_conversion(std::vector<uint32_t>& output, spv::Op opcode,
                        const uint32_t* instruction)
    {
        const uint32_t destination_type = instruction[1];
        const uint32_t result_id = instruction[2];
        const uint32_t source = instruction[3];
        const uint32_t source_type = source < value_types.size() ? value_types[source] : 0;
        if (source_type == 0)
            return -1;

        const uint32_t source_count = component_count(source_type);
        const uint32_t destination_count = component_count(destination_type);
        if (source_count != destination_count)
        {
            log_error("16-bit integer conversion changes vector width");
            return -1;
        }

        if (source_count == 1)
        {
            emit_scalar_conversion(output, opcode, source, source_type,
                                   destination_type, result_id);
            return 0;
        }

        std::vector<uint32_t> converted;
        converted.reserve(source_count);
        const uint32_t source_scalar_type = scalar_original_type(source_type);
        const uint32_t destination_scalar_type = scalar_original_type(destination_type);
        for (uint32_t i = 0; i < source_count; i++)
        {
            const uint32_t component = allocate_id();
            append_instruction(output, spv::Op::OpCompositeExtract,
                               {mapped_type(source_scalar_type), component, source, i});
            const uint32_t converted_component = allocate_id();
            emit_scalar_conversion(output, opcode, component, source_scalar_type,
                                   destination_scalar_type, converted_component);
            converted.push_back(converted_component);
        }

        output.push_back(((uint32_t)(converted.size() + 3) << 16)
                         | (uint32_t)spv::Op::OpCompositeConstruct);
        output.push_back(mapped_type(destination_type));
        output.push_back(result_id);
        output.insert(output.end(), converted.begin(), converted.end());
        return 0;
    }

    void allocate_helpers()
    {
        for (uint32_t i = 1; i < original_bound; i++)
        {
            if (types[i].kind == SPIRV_TYPE_INT && types[i].width == 32)
            {
                if (types[i].is_signed && helper_i32 == 0)
                    helper_i32 = i;
                if (!types[i].is_signed && helper_u32 == 0)
                    helper_u32 = i;
            }
        }
        if (helper_u32 == 0)
            helper_u32 = allocate_id();
        if (helper_i32 == 0)
            helper_i32 = allocate_id();

        for (uint32_t i = 1; i < original_bound; i++)
        {
            if (types[i].kind != SPIRV_TYPE_VECTOR
                || types[i].element_type >= types.size()
                || types[types[i].element_type].kind != SPIRV_TYPE_INT
                || types[types[i].element_type].width != 32)
                continue;

            const bool is_signed = types[types[i].element_type].is_signed;
            if (types[i].element_count == 2 && is_signed && helper_v2i32 == 0)
                helper_v2i32 = i;
            else if (types[i].element_count == 2 && !is_signed && helper_v2u32 == 0)
                helper_v2u32 = i;
            else if (types[i].element_count == 3 && is_signed && helper_v3i32 == 0)
                helper_v3i32 = i;
            else if (types[i].element_count == 3 && !is_signed && helper_v3u32 == 0)
                helper_v3u32 = i;
            else if (types[i].element_count == 4 && is_signed && helper_v4i32 == 0)
                helper_v4i32 = i;
            else if (types[i].element_count == 4 && !is_signed && helper_v4u32 == 0)
                helper_v4u32 = i;
        }

        if (helper_v2u32 == 0)
            helper_v2u32 = allocate_id();
        if (helper_v2i32 == 0)
            helper_v2i32 = allocate_id();
        if (helper_v3u32 == 0)
            helper_v3u32 = allocate_id();
        if (helper_v3i32 == 0)
            helper_v3i32 = allocate_id();
        if (helper_v4u32 == 0)
            helper_v4u32 = allocate_id();
        if (helper_v4i32 == 0)
            helper_v4i32 = allocate_id();

        const uint32_t helper_types[] = {
            helper_u32, helper_i32, helper_v2u32, helper_v2i32,
            helper_v3u32, helper_v3i32, helper_v4u32, helper_v4i32};
        for (size_t i = 0; i < sizeof(helper_types) / sizeof(helper_types[0]); i++)
        {
            const uint32_t type_id = helper_types[i];
            if (type_id >= original_bound || type_offsets[type_id] >= type_insert_offset)
                moved_helper_types.insert(type_id);
        }

        constant_u32_0 = allocate_id();
        constant_u32_1 = allocate_id();
        constant_u32_4 = allocate_id();
        constant_u32_16 = allocate_id();
        constant_u32_ffff = allocate_id();
        constant_i32_16 = allocate_id();

        for (size_t i = 0; i < buffers.size(); i++)
            packed_element_types[buffers[i].element_type] = packed_type(buffers[i].component_count);
    }

    void append_helper_types(std::vector<uint32_t>& output)
    {
        if (moved_helper_types.count(helper_u32) != 0)
            append_instruction(output, spv::Op::OpTypeInt, {helper_u32, 32, 0});
        if (moved_helper_types.count(helper_i32) != 0)
            append_instruction(output, spv::Op::OpTypeInt, {helper_i32, 32, 1});
        if (moved_helper_types.count(helper_v2u32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v2u32, helper_u32, 2});
        if (moved_helper_types.count(helper_v2i32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v2i32, helper_i32, 2});
        if (moved_helper_types.count(helper_v3u32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v3u32, helper_u32, 3});
        if (moved_helper_types.count(helper_v3i32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v3i32, helper_i32, 3});
        if (moved_helper_types.count(helper_v4u32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v4u32, helper_u32, 4});
        if (moved_helper_types.count(helper_v4i32) != 0)
            append_instruction(output, spv::Op::OpTypeVector, {helper_v4i32, helper_i32, 4});
    }

    void append_helper_constants(std::vector<uint32_t>& output)
    {
        append_instruction(output, spv::Op::OpConstant, {helper_u32, constant_u32_0, 0});
        append_instruction(output, spv::Op::OpConstant, {helper_u32, constant_u32_1, 1});
        append_instruction(output, spv::Op::OpConstant, {helper_u32, constant_u32_4, 4});
        append_instruction(output, spv::Op::OpConstant, {helper_u32, constant_u32_16, 16});
        append_instruction(output, spv::Op::OpConstant, {helper_u32, constant_u32_ffff, 0xffff});
        append_instruction(output, spv::Op::OpConstant, {helper_i32, constant_i32_16, 16});
    }

    int rewrite_access(std::vector<uint32_t>& output, const uint32_t* instruction,
                       uint32_t word_count, Storage16Access& access)
    {
        const Storage16Buffer& buffer = buffers[access.buffer_index];
        uint32_t normalized_index = access.index_id;
        if (types[access.index_type].is_signed)
        {
            normalized_index = allocate_id();
            append_instruction(output, spv::Op::OpBitcast,
                               {helper_u32, normalized_index, access.index_id});
        }
        access.normalized_index_id = normalized_index;

        uint32_t packed_index = normalized_index;
        if (buffer.component_count == 1)
        {
            packed_index = allocate_id();
            append_instruction(output, spv::Op::OpShiftRightLogical,
                               {helper_u32, packed_index, normalized_index, constant_u32_1});
        }

        output.insert(output.end(), instruction, instruction + word_count);
        output[output.size() - word_count + 5] = packed_index;
        return 0;
    }

    int rewrite()
    {
        if (buffers.empty())
        {
            std::vector<uint32_t> output;
            output.reserve(spirv.size());
            output.insert(output.end(), spirv.begin(), spirv.begin() + 5);
            for (size_t i = 5; i < spirv.size();)
            {
                const uint32_t word_count = spirv[i] >> 16;
                const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
                if (opcode != spv::Op::OpCapability || word_count != 2
                    || spirv[i + 1] != (uint32_t)spv::Capability::Int16)
                {
                    output.insert(output.end(), spirv.begin() + i,
                                  spirv.begin() + i + word_count);
                }
                i += word_count;
            }
            spirv.swap(output);
            return 0;
        }

        allocate_helpers();
        std::vector<uint32_t> output;
        output.reserve(spirv.size() + 512);
        output.insert(output.end(), spirv.begin(), spirv.begin() + 5);

        bool inserted_types = false;
        bool inserted_constants = false;
        for (size_t i = 5; i < spirv.size();)
        {
            if (!inserted_types && i == type_insert_offset)
            {
                append_helper_types(output);
                inserted_types = true;
            }
            if (!inserted_constants && i == function_insert_offset)
            {
                append_helper_constants(output);
                inserted_constants = true;
            }

            const uint32_t word_count = spirv[i] >> 16;
            const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
            const uint32_t* instruction = &spirv[i];

            if (opcode == spv::Op::OpCapability && word_count == 2
                && spirv[i + 1] == (uint32_t)spv::Capability::Int16)
            {
                i += word_count;
                continue;
            }

            if (is_type_instruction(opcode) && word_count >= 2
                && moved_helper_types.count(spirv[i + 1]) != 0
                && i != type_insert_offset)
            {
                i += word_count;
                continue;
            }

            if (opcode == spv::Op::OpDecorate && word_count >= 4
                && spirv[i + 2] == (uint32_t)spv::Decoration::ArrayStride)
            {
                bool rewritten = false;
                for (size_t j = 0; j < buffers.size(); j++)
                {
                    if (spirv[i + 1] == buffers[j].array_type)
                    {
                        output.insert(output.end(), instruction, instruction + word_count);
                        output.back() = buffers[j].component_count == 1   ? 4
                                        : buffers[j].component_count == 2 ? 4
                                                                          : 8;
                        rewritten = true;
                        break;
                    }
                }
                if (rewritten)
                {
                    i += word_count;
                    continue;
                }
            }

            if (opcode == spv::Op::OpTypeRuntimeArray && word_count == 3)
            {
                bool rewritten = false;
                for (size_t j = 0; j < buffers.size(); j++)
                {
                    if (spirv[i + 1] == buffers[j].array_type)
                    {
                        append_instruction(output, opcode,
                                           {spirv[i + 1], packed_type(buffers[j].component_count)});
                        rewritten = true;
                        break;
                    }
                }
                if (rewritten)
                {
                    i += word_count;
                    continue;
                }
            }

            if (opcode == spv::Op::OpTypePointer && word_count == 4
                && (spirv[i + 2] == (uint32_t)spv::StorageClass::StorageBuffer
                    || spirv[i + 2] == (uint32_t)spv::StorageClass::Uniform))
            {
                std::unordered_map<uint32_t, uint32_t>::const_iterator it = packed_element_types.find(spirv[i + 3]);
                if (it != packed_element_types.end())
                {
                    append_instruction(output, opcode,
                                       {spirv[i + 1], spirv[i + 2], it->second});
                    i += word_count;
                    continue;
                }
            }

            if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
                && word_count == 6)
            {
                std::unordered_map<uint32_t, Storage16Access>::iterator it = accesses.find(spirv[i + 2]);
                if (it != accesses.end())
                {
                    rewrite_access(output, instruction, word_count, it->second);
                    i += word_count;
                    continue;
                }
            }

            if (opcode == spv::Op::OpLoad && word_count >= 4)
            {
                std::unordered_map<uint32_t, Storage16Access>::const_iterator it = accesses.find(spirv[i + 3]);
                if (it != accesses.end())
                {
                    emit_load(output, instruction, it->second);
                    i += word_count;
                    continue;
                }
            }

            if (opcode == spv::Op::OpStore && word_count == 3)
            {
                std::unordered_map<uint32_t, Storage16Access>::const_iterator it = accesses.find(spirv[i + 1]);
                if (it != accesses.end())
                {
                    emit_store(output, instruction, it->second);
                    i += word_count;
                    continue;
                }
            }

            if ((opcode == spv::Op::OpSConvert || opcode == spv::Op::OpUConvert)
                && word_count == 4)
            {
                const uint32_t source = spirv[i + 3];
                const uint32_t source_type = source < value_types.size() ? value_types[source] : 0;
                uint32_t source_scalar = 0;
                uint32_t source_count = 0;
                uint32_t destination_scalar = 0;
                uint32_t destination_count = 0;
                const bool source_is_16 = integer_element_type(types, source_type, source_scalar, source_count);
                const bool destination_is_16 = integer_element_type(types, spirv[i + 1], destination_scalar, destination_count);
                if (source_is_16 || destination_is_16)
                {
                    if (emit_conversion(output, opcode, instruction) != 0)
                        return -1;
                    i += word_count;
                    continue;
                }
            }

            if (opcode == spv::Op::OpBitcast && word_count == 4)
            {
                const uint32_t source = spirv[i + 3];
                const uint32_t source_type = source < value_types.size() ? value_types[source] : 0;
                uint32_t source_scalar = 0;
                uint32_t source_count = 0;
                uint32_t destination_scalar = 0;
                uint32_t destination_count = 0;
                const bool source_is_16 = integer_element_type(types, source_type,
                                                               source_scalar, source_count);
                const bool destination_is_16 = integer_element_type(types, spirv[i + 1],
                                                                    destination_scalar, destination_count);
                if (source_is_16 || destination_is_16)
                {
                    if (emit_bitcast(output, instruction) != 0)
                    {
                        log_error("unsupported 16-bit integer SPIR-V bitcast source-type=%u destination-type=%u",
                                  source_type, spirv[i + 1]);
                        return -1;
                    }
                    i += word_count;
                    continue;
                }
            }

            if ((opcode == spv::Op::OpCompositeConstruct
                 || opcode == spv::Op::OpCompositeExtract
                 || opcode == spv::Op::OpVectorShuffle
                 || opcode == spv::Op::OpCopyObject)
                && word_count >= 3 && mapped_type(spirv[i + 1]) != spirv[i + 1])
            {
                output.insert(output.end(), instruction, instruction + word_count);
                output[output.size() - word_count + 1] = mapped_type(spirv[i + 1]);
                i += word_count;
                continue;
            }

            if ((opcode == spv::Op::OpConstant || opcode == spv::Op::OpSpecConstant)
                && word_count == 4 && is_integer_type(types, spirv[i + 1], 16))
            {
                uint32_t value = spirv[i + 3] & 0xffff;
                if (types[spirv[i + 1]].is_signed && (value & 0x8000) != 0)
                    value |= 0xffff0000u;
                append_instruction(output, opcode,
                                   {mapped_type(spirv[i + 1]), spirv[i + 2], value});
                i += word_count;
                continue;
            }

            if ((opcode == spv::Op::OpConstantComposite || opcode == spv::Op::OpSpecConstantComposite)
                && word_count >= 4 && mapped_type(spirv[i + 1]) != spirv[i + 1])
            {
                output.insert(output.end(), instruction, instruction + word_count);
                output[output.size() - word_count + 1] = mapped_type(spirv[i + 1]);
                i += word_count;
                continue;
            }

            if ((opcode == spv::Op::OpConstantNull || opcode == spv::Op::OpUndef)
                && word_count == 3 && mapped_type(spirv[i + 1]) != spirv[i + 1])
            {
                append_instruction(output, opcode,
                                   {mapped_type(spirv[i + 1]), spirv[i + 2]});
                i += word_count;
                continue;
            }

            if (spirv_instruction_has_result_type(opcode) && word_count >= 3
                && mapped_type(spirv[i + 1]) != spirv[i + 1])
            {
                log_error("unsupported 16-bit integer SPIR-V opcode %u", (uint32_t)opcode);
                return -1;
            }

            output.insert(output.end(), instruction, instruction + word_count);
            i += word_count;
        }

        if (!inserted_types || !inserted_constants)
            return -1;

        output[3] = next_id;
        spirv.swap(output);
        return 0;
    }
};

} // namespace

int legalize_storage_buffer_16bit_integer_spirv(std::vector<uint32_t>& spirv)
{
    Storage16Legalizer legalizer(spirv);
    if (legalizer.parse() != 0)
        return -1;
    return legalizer.rewrite();
}

} // namespace nikki_detail
