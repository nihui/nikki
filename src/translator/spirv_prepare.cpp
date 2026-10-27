// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

#include <initializer_list>

namespace nikki_detail {

int normalize_spirv(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t bound = spirv[3];
    uint32_t local_size_ids[3] = {0, 0, 0};
    uint32_t workgroup_size_builtin_id = 0;
    std::vector<int32_t> spec_ids(bound, -1);
    std::vector<uint32_t> workgroup_size_members;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpExecutionModeId && word_count >= 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_ids[0] = spirv[i + 3];
            local_size_ids[1] = spirv[i + 4];
            local_size_ids[2] = spirv[i + 5];
        }
        else if (opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 1] < bound)
        {
            if (spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
                spec_ids[spirv[i + 1]] = spirv[i + 3];
            else if (spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn && spirv[i + 3] == (uint32_t)spv::BuiltIn::WorkgroupSize)
                workgroup_size_builtin_id = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpSpecConstantComposite && word_count == 6 && spirv[i + 2] == workgroup_size_builtin_id)
        {
            workgroup_size_members.assign(spirv.begin() + i + 3, spirv.begin() + i + 6);
        }

        i += word_count;
    }

    bool has_workgroup_size_builtin = workgroup_size_members.size() == 3;
    for (int i = 0; i < 3 && has_workgroup_size_builtin; i++)
    {
        const uint32_t id = workgroup_size_members[i];
        if (id >= bound || spec_ids[id] != 233 + i)
            has_workgroup_size_builtin = false;
    }

    const bool has_local_size_id_mode = local_size_ids[0] != 0 && local_size_ids[1] != 0 && local_size_ids[2] != 0;

    std::vector<uint32_t> normalized;
    normalized.reserve(spirv.size());
    normalized.insert(normalized.end(), spirv.begin(), spirv.begin() + 5);

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        const bool non_readable_decorate = opcode == spv::Op::OpDecorate && word_count >= 3 && spirv[i + 2] == (uint32_t)spv::Decoration::NonReadable;
        const bool non_readable_member_decorate = opcode == spv::Op::OpMemberDecorate && word_count >= 4 && spirv[i + 3] == (uint32_t)spv::Decoration::NonReadable;
        const bool literal_local_size = has_workgroup_size_builtin && opcode == spv::Op::OpExecutionMode && word_count == 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSize;
        bool duplicate_local_size_spec_id = false;
        if (has_local_size_id_mode && opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
        {
            const uint32_t spec_id = spirv[i + 3];
            if (spec_id >= 233 && spec_id <= 235 && spirv[i + 1] != local_size_ids[spec_id - 233])
                duplicate_local_size_spec_id = true;
        }

        if (!non_readable_decorate && !non_readable_member_decorate && !literal_local_size && !duplicate_local_size_spec_id)
            normalized.insert(normalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);

        i += word_count;
    }

    spirv.swap(normalized);
    return 0;
}

int specialize_spirv(std::vector<uint32_t>& spirv, const VkSpecializationInfo* info)
{
    if (info && info->dataSize != 0 && !info->pData)
        return -1;
    if (info && info->mapEntryCount != 0 && !info->pMapEntries)
        return -1;

    std::unordered_map<uint32_t, std::vector<uint32_t> > values;
    if (info)
    {
        for (uint32_t i = 0; i < info->mapEntryCount; i++)
        {
            const VkSpecializationMapEntry& entry = info->pMapEntries[i];
            if (entry.size != 4 || entry.offset > info->dataSize
                || entry.size > info->dataSize - entry.offset)
            {
                log_error("specialization id %u has invalid offset=%u size=%llu data-size=%llu",
                          entry.constantID, entry.offset, (unsigned long long)entry.size,
                          (unsigned long long)info->dataSize);
                return -1;
            }

            if (values.find(entry.constantID) != values.end())
            {
                log_error("specialization id %u is specified more than once", entry.constantID);
                return -1;
            }

            std::vector<uint32_t>& value = values[entry.constantID];
            value.resize(1);
            memcpy(value.data(), (const unsigned char*)info->pData + entry.offset, entry.size);
        }
    }

    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
    optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        log_error("SPIR-V optimizer word=%zu: %s", position.index, message ? message : "");
    });
    // Entry-point selection has removed other roots. Remove their private call
    // trees before any legalization counts operations or examines scalar types.
    optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
    if (!values.empty())
        optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(values));
    optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
    optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass(false));
    optimizer.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());

    std::vector<uint32_t> specialized;
    if (!optimizer.Run(spirv.data(), spirv.size(), &specialized))
    {
        log_error("SPIR-V specialization failed");
        return -1;
    }

    simplify_bounded_integer_loops(specialized);
    spirv.swap(specialized);
    return 0;
}

int legalize_non_finite_fp16_constants(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    std::unordered_set<uint32_t> fp16_types;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpTypeFloat && word_count == 3 && spirv[i + 2] == 16)
            fp16_types.insert(spirv[i + 1]);

        i += word_count;
    }

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if ((opcode == spv::Op::OpConstant || opcode == spv::Op::OpSpecConstant)
            && word_count == 4 && fp16_types.count(spirv[i + 1]) != 0)
        {
            const uint16_t bits = (uint16_t)spirv[i + 3];
            if ((bits & 0x7c00) == 0x7c00)
            {
                if ((bits & 0x03ff) != 0)
                {
                    log_error("non-finite f16 NaN constants are not supported by WGSL");
                    return -1;
                }

                // WGSL shader-creation constants must be finite. Match its
                // runtime f32-to-f16 conversion by clamping overflow.
                spirv[i + 3] = (uint32_t)((bits & 0x8000) | 0x7bff);
            }
        }

        i += word_count;
    }

    return 0;
}

std::string spirv_literal_string(const uint32_t* words, size_t word_count)
{
    std::string value;
    value.reserve(word_count * 4);
    for (size_t i = 0; i < word_count; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            const char ch = (char)((words[i] >> (j * 8)) & 0xff);
            if (ch == 0)
                return value;
            value.push_back(ch);
        }
    }

    return value;
}

// Tint's SPIR-V reader does not accept SPV_KHR_integer_dot_product yet. Packed
// same-signed dots use a reverseBits(reverseBits(x)) ^ y marker, restored to
// dot4I8Packed/dot4U8Packed in IR. Mixed dots and saturation use exact integer
// operations; in particular, saturation must not rely on signed add overflow.
int legalize_packed_integer_dot_spirv(std::vector<uint32_t>& spirv,
                                      size_t& packed_integer_dot_count, bool* uses_integer_dot)
{
    packed_integer_dot_count = 0;
    if (uses_integer_dot)
        *uses_integer_dot = false;
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    auto is_dot = [](spv::Op op) {
        return op == spv::Op::OpSDot || op == spv::Op::OpUDot || op == spv::Op::OpSUDot
            || op == spv::Op::OpSDotAccSat || op == spv::Op::OpUDotAccSat || op == spv::Op::OpSUDotAccSat;
    };
    auto is_acc = [](spv::Op op) {
        return op == spv::Op::OpSDotAccSat || op == spv::Op::OpUDotAccSat || op == spv::Op::OpSUDotAccSat;
    };
    std::unordered_map<uint32_t, uint32_t> widths;
    std::unordered_map<uint32_t, uint32_t> value_types;
    uint32_t u32_type = 0, i32_type = 0, bool_type = 0;
    size_t dot_count = 0;
    bool has_declaration = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t wc = spirv[i] >> 16;
        const spv::Op op = (spv::Op)(spirv[i] & 0xffff);
        if (wc == 0 || i + wc > spirv.size())
            return -1;
        if (op == spv::Op::OpTypeInt && wc == 4)
        {
            widths[spirv[i + 1]] = spirv[i + 2];
            if (spirv[i + 2] == 32)
                (spirv[i + 3] ? i32_type : u32_type) = spirv[i + 1];
        }
        if (op == spv::Op::OpTypeBool && wc == 2)
            bool_type = spirv[i + 1];
        bool result = false, type = false;
        spv::HasResultAndType(op, &result, &type);
        if (result && type && wc >= 3)
            value_types[spirv[i + 2]] = spirv[i + 1];
        if (is_dot(op))
        {
            const unsigned expected = is_acc(op) ? 7 : 6;
            if (wc != expected || spirv[i + wc - 1] != (uint32_t)spv::PackedVectorFormat::PackedVectorFormat4x8Bit)
            {
                log_error("only packed 4x8 integer dot products are supported");
                return -1;
            }
            dot_count++;
        }
        if ((op == spv::Op::OpExtension && spirv_literal_string(&spirv[i + 1], wc - 1) == "SPV_KHR_integer_dot_product")
            || (op == spv::Op::OpCapability && wc == 2
                && (spirv[i + 1] == (uint32_t)spv::Capability::DotProductInput4x8BitPacked
                    || spirv[i + 1] == (uint32_t)spv::Capability::DotProduct)))
            has_declaration = true;
        i += wc;
    }
    if (uses_integer_dot)
        *uses_integer_dot = dot_count != 0;
    if (!dot_count && !has_declaration)
        return 0;
    // Each supported instruction needs fewer than 64 fresh IDs (including the
    // shared declarations). Check before emitting, rather than wrapping IDs.
    if (spirv[3] > UINT32_MAX - 16u || dot_count > (UINT32_MAX - spirv[3] - 16ull) / 64ull)
        return -1;
    uint32_t next_id = spirv[3];
    std::vector<uint32_t> declarations;
    auto emit = [](std::vector<uint32_t>& out, spv::Op op, std::initializer_list<uint32_t> args) {
        out.push_back(((uint32_t)args.size() + 1) << 16 | (uint32_t)op);
        out.insert(out.end(), args.begin(), args.end());
    };
    if (!u32_type) { u32_type = next_id++; emit(declarations, spv::Op::OpTypeInt, {u32_type, 32, 0}); }
    if (!i32_type) { i32_type = next_id++; emit(declarations, spv::Op::OpTypeInt, {i32_type, 32, 1}); }
    if (!bool_type) { bool_type = next_id++; emit(declarations, spv::Op::OpTypeBool, {bool_type}); }
    auto constant = [&](uint32_t value) {
        const uint32_t id = next_id++;
        emit(declarations, spv::Op::OpConstant, {u32_type, id, value});
        return id;
    };
    const uint32_t zero = constant(0), sign = constant(0x80000000u);
    const uint32_t maximum = constant(0x7fffffffu), umax = constant(0xffffffffu), eight = constant(8);
    const uint32_t offsets[] = {zero, eight, constant(16), constant(24)};
    // Keep the marker's input non-constant through Tint's SPIR-V reader.
    // Its builder folds reverseBits(constant), including frozen specialization
    // constants. A real function call preserves the value without depending on
    // its bits or on which operand happens to be constant.
    uint32_t identity_type = 0;
    for (size_t i = 5; i < spirv.size(); i += spirv[i] >> 16)
        if ((spirv[i] & 0xffff) == (uint32_t)spv::Op::OpTypeFunction
            && (spirv[i] >> 16) == 4 && spirv[i + 2] == u32_type && spirv[i + 3] == u32_type)
            identity_type = spirv[i + 1];
    if (!identity_type)
    {
        identity_type = next_id++;
        emit(declarations, spv::Op::OpTypeFunction, {identity_type, u32_type, u32_type});
    }
    const uint32_t identity = next_id++, identity_arg = next_id++, identity_label = next_id++;
    std::vector<uint32_t> body;
    auto unary = [&](spv::Op op, uint32_t type, uint32_t a) {
        uint32_t id = next_id++; emit(body, op, {type, id, a}); return id;
    };
    auto binary = [&](spv::Op op, uint32_t type, uint32_t a, uint32_t b) {
        uint32_t id = next_id++; emit(body, op, {type, id, a, b}); return id;
    };
    auto select = [&](uint32_t condition, uint32_t yes, uint32_t no) {
        uint32_t id = next_id++; emit(body, spv::Op::OpSelect, {u32_type, id, condition, yes, no}); return id;
    };
    auto cast = [&](uint32_t type, uint32_t id, uint32_t from_type) {
        return type == from_type ? id : unary(spv::Op::OpBitcast, type, id);
    };
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t wc = spirv[i] >> 16;
        const spv::Op op = (spv::Op)(spirv[i] & 0xffff);
        if (is_dot(op))
        {
            const uint32_t result_type = spirv[i + 1], lhs_type = value_types[spirv[i + 3]], rhs_type = value_types[spirv[i + 4]];
            if (widths[result_type] != 32 || widths[lhs_type] != 32 || widths[rhs_type] != 32
                || (is_acc(op) && value_types[spirv[i + 5]] != result_type))
            {
                log_error("packed integer dot products require scalar 32-bit operands and result");
                return -1;
            }
            const bool unsign = op == spv::Op::OpUDot || op == spv::Op::OpUDotAccSat;
            const bool mixed = op == spv::Op::OpSUDot || op == spv::Op::OpSUDotAccSat;
            const uint32_t type = unsign ? u32_type : i32_type;
            const uint32_t lhs = cast(type, spirv[i + 3], lhs_type);
            const uint32_t rhs = cast(mixed ? u32_type : type, spirv[i + 4], rhs_type);
            uint32_t dot = 0;
            if (mixed)
            {
                // The signed x unsigned 4x8 sum lies in [-130560,129540],
                // so every product and partial sum fits exactly in i32.
                for (unsigned lane = 0; lane < 4; lane++)
                {
                    uint32_t a = next_id++, b = next_id++;
                    emit(body, spv::Op::OpBitFieldSExtract, {i32_type, a, lhs, offsets[lane], eight});
                    emit(body, spv::Op::OpBitFieldUExtract, {u32_type, b, rhs, offsets[lane], eight});
                    b = unary(spv::Op::OpBitcast, i32_type, b);
                    uint32_t product = binary(spv::Op::OpIMul, i32_type, a, b);
                    dot = lane ? binary(spv::Op::OpIAdd, i32_type, dot, product) : product;
                }
            }
            else
            {
                const uint32_t input = cast(u32_type, lhs, type);
                const uint32_t guarded = next_id++;
                emit(body, spv::Op::OpFunctionCall, {u32_type, guarded, identity, input});
                const uint32_t guarded_lhs = cast(type, guarded, u32_type);
                uint32_t reversed = unary(spv::Op::OpBitReverse, type, guarded_lhs);
                reversed = unary(spv::Op::OpBitReverse, type, reversed);
                dot = binary(spv::Op::OpBitwiseXor, type, reversed, rhs);
                packed_integer_dot_count++;
            }
            if (is_acc(op))
            {
                const uint32_t acc = cast(u32_type, spirv[i + 5], result_type);
                const uint32_t udot = cast(u32_type, dot, type);
                const uint32_t sum = binary(spv::Op::OpIAdd, u32_type, acc, udot);
                uint32_t saturated;
                if (unsign)
                {
                    const uint32_t overflow = binary(spv::Op::OpULessThan, bool_type, sum, acc);
                    saturated = select(overflow, umax, sum);
                }
                else
                {
                    const uint32_t changed_acc = binary(spv::Op::OpBitwiseXor, u32_type, acc, sum);
                    const uint32_t changed_dot = binary(spv::Op::OpBitwiseXor, u32_type, udot, sum);
                    uint32_t overflow = binary(spv::Op::OpBitwiseAnd, u32_type, changed_acc, changed_dot);
                    overflow = binary(spv::Op::OpBitwiseAnd, u32_type, overflow, sign);
                    overflow = binary(spv::Op::OpINotEqual, bool_type, overflow, zero);
                    uint32_t negative = binary(spv::Op::OpBitwiseAnd, u32_type, acc, sign);
                    negative = binary(spv::Op::OpINotEqual, bool_type, negative, zero);
                    const uint32_t limit = select(negative, sign, maximum);
                    saturated = select(overflow, limit, sum);
                }
                emit(body, result_type == u32_type ? spv::Op::OpCopyObject : spv::Op::OpBitcast,
                     {result_type, spirv[i + 2], saturated});
            }
            else
                emit(body, result_type == type ? spv::Op::OpCopyObject : spv::Op::OpBitcast,
                     {result_type, spirv[i + 2], dot});
        }
        else
        {
            const bool extension = op == spv::Op::OpExtension
                && spirv_literal_string(&spirv[i + 1], wc - 1) == "SPV_KHR_integer_dot_product";
            const bool capability = op == spv::Op::OpCapability && wc == 2
                && (spirv[i + 1] == (uint32_t)spv::Capability::DotProductInput4x8BitPacked
                    || spirv[i + 1] == (uint32_t)spv::Capability::DotProduct);
            if (!extension && !capability)
                body.insert(body.end(), spirv.begin() + i, spirv.begin() + i + wc);
        }
        i += wc;
    }
    std::vector<uint32_t> legalized(spirv.begin(), spirv.begin() + 5);
    legalized[3] = next_id;
    bool inserted = false;
    for (size_t i = 0; i < body.size();)
    {
        const uint32_t wc = body[i] >> 16;
        if (!inserted && (spv::Op)(body[i] & 0xffff) == spv::Op::OpFunction)
        {
            legalized.insert(legalized.end(), declarations.begin(), declarations.end());
            inserted = true;
        }
        legalized.insert(legalized.end(), body.begin() + i, body.begin() + i + wc);
        i += wc;
    }
    if (packed_integer_dot_count)
    {
        emit(legalized, spv::Op::OpFunction, {u32_type, identity, 0, identity_type});
        emit(legalized, spv::Op::OpFunctionParameter, {u32_type, identity_arg});
        emit(legalized, spv::Op::OpLabel, {identity_label});
        emit(legalized, spv::Op::OpReturnValue, {identity_arg});
        emit(legalized, spv::Op::OpFunctionEnd, {});
    }
    spirv.swap(legalized);
    return 0;
}

bool spirv_instruction_has_result_type(spv::Op opcode)
{
    switch (opcode)
    {
    case spv::Op::OpUndef:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantNull:
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
    case spv::Op::OpVariable:
    case spv::Op::OpLoad:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpCopyObject:
    case spv::Op::OpTranspose:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpBitcast:
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpPhi:
    case spv::Op::OpSelect:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpCooperativeMatrixLoadKHR:
    case spv::Op::OpCooperativeMatrixMulAddKHR:
        return true;
    default:
        return false;
    }
}

} // namespace nikki_detail
