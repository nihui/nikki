// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

using namespace tint::core::number_suffixes;
using namespace tint::core::fluent_types;

// SPIR-V OpAtomicCompareExchange has strong compare-exchange semantics, while
// WGSL only exposes atomicCompareExchangeWeak(). Retry spurious failures so the
// value returned to the Vulkan shader preserves the SPIR-V contract.
int strengthen_atomic_compare_exchange_ir(tint::core::ir::Module& module)
{
    std::vector<tint::core::ir::CoreBuiltinCall*> worklist;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::CoreBuiltinCall* call = instruction->As<tint::core::ir::CoreBuiltinCall>();
        if (call && call->Func() == tint::core::BuiltinFn::kAtomicCompareExchangeWeak)
            worklist.push_back(call);
    }

    tint::core::ir::Builder builder(module);
    for (size_t i = 0; i < worklist.size(); i++)
    {
        tint::core::ir::CoreBuiltinCall* call = worklist[i];
        if (call->Args().size() != 3 || !call->Result())
            return -1;

        const tint::Vector<tint::core::ir::Usage, 4> usages = call->Result()->UsagesSorted();
        tint::core::ir::Access* old_value_access = usages.Length() == 1
                                                       ? usages[0].instruction->As<tint::core::ir::Access>()
                                                       : 0;
        if (!old_value_access || old_value_access->Object() != call->Result())
            return -1;

        tint::core::ir::Value* pointer = call->Args()[0];
        tint::core::ir::Value* compare = call->Args()[1];
        tint::core::ir::Value* value = call->Args()[2];
        const tint::core::type::Type* result_type = call->Result()->Type();
        if (old_value_access->Result()->Type() != compare->Type())
            return -1;
        tint::core::ir::InstructionResult* result = old_value_access->DetachResult();

        builder.InsertBefore(call, [&]() {
            tint::core::ir::Var* strong_result = builder.Var(module.Types().ptr<function>(compare->Type()));
            tint::core::ir::Loop* loop = builder.Loop();
            builder.Append(loop->Body(), [&]() {
                tint::core::ir::Value* attempt = builder.Call(
                    result_type, tint::core::BuiltinFn::kAtomicCompareExchangeWeak,
                    pointer, compare, value);
                tint::core::ir::Value* old_value = builder.Access(compare->Type(), attempt, 0_u);
                tint::core::ir::Value* exchanged = builder.Access(module.Types().bool_(), attempt, 1_u);
                tint::core::ir::Value* value_changed = builder.NotEqual(old_value, compare);
                tint::core::ir::Value* complete = builder.Or(exchanged, value_changed);
                tint::core::ir::If* branch = builder.If(complete);
                builder.Append(branch->True(), [&]() {
                    builder.Store(strong_result, old_value);
                    builder.ExitLoop(loop);
                });
                builder.Continue(loop);
            });
            builder.LoadWithResult(result, strong_result);
        });
        old_value_access->Destroy();
        call->Destroy();
    }

    return 0;
}

int restore_packed_integer_dot_ir(tint::core::ir::Module& module, size_t expected_count)
{
    if (expected_count == 0)
        return 0;

    struct WorkItem
    {
        tint::core::ir::CoreBinary* binary;
        bool is_signed;
    };

    std::vector<WorkItem> worklist;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::CoreBinary* binary = instruction->As<tint::core::ir::CoreBinary>();
        if (!binary || binary->Op() != tint::core::BinaryOp::kXor)
            continue;

        tint::core::ir::InstructionResult* outer_reverse_result = binary->LHS()->As<tint::core::ir::InstructionResult>();
        if (!outer_reverse_result)
            continue;
        tint::core::ir::CoreBuiltinCall* outer_reverse = outer_reverse_result->Instruction()->As<tint::core::ir::CoreBuiltinCall>();
        if (!outer_reverse || outer_reverse->Func() != tint::core::BuiltinFn::kReverseBits
            || outer_reverse->Args().size() != 1)
            continue;
        tint::core::ir::InstructionResult* inner_reverse_result = outer_reverse->Args()[0]->As<tint::core::ir::InstructionResult>();
        if (!inner_reverse_result)
            continue;
        tint::core::ir::CoreBuiltinCall* inner_reverse = inner_reverse_result->Instruction()->As<tint::core::ir::CoreBuiltinCall>();
        if (!inner_reverse || inner_reverse->Func() != tint::core::BuiltinFn::kReverseBits
            || inner_reverse->Args().size() != 1)
            continue;

        const bool is_signed = binary->Result()->Type()->Is<tint::core::type::I32>();
        if (!is_signed && !binary->Result()->Type()->Is<tint::core::type::U32>())
            continue;
        worklist.push_back(WorkItem{binary, is_signed});
    }

    if (worklist.size() != expected_count)
    {
        log_error("restored %llu of %llu packed integer dot markers",
                  (unsigned long long)worklist.size(), (unsigned long long)expected_count);
        return -1;
    }

    tint::core::ir::Builder builder(module);
    for (size_t i = 0; i < worklist.size(); i++)
    {
        tint::core::ir::CoreBinary* binary = worklist[i].binary;
        const bool is_signed = worklist[i].is_signed;
        tint::core::ir::CoreBuiltinCall* outer_reverse = binary->LHS()->As<tint::core::ir::InstructionResult>()->Instruction()->As<tint::core::ir::CoreBuiltinCall>();
        tint::core::ir::CoreBuiltinCall* inner_reverse = outer_reverse->Args()[0]->As<tint::core::ir::InstructionResult>()->Instruction()->As<tint::core::ir::CoreBuiltinCall>();

        tint::core::ir::InstructionResult* result = binary->DetachResult();
        builder.InsertBefore(binary, [&]() {
            tint::core::ir::Value* lhs = inner_reverse->Args()[0];
            tint::core::ir::Value* rhs = binary->RHS();
            if (is_signed)
            {
                lhs = builder.Bitcast(module.Types().u32(), lhs);
                rhs = builder.Bitcast(module.Types().u32(), rhs);
            }
            builder.CallReplaceResult(result,
                                   is_signed ? tint::core::BuiltinFn::kDot4I8Packed
                                             : tint::core::BuiltinFn::kDot4U8Packed,
                                   lhs, rhs);
        });
        binary->Destroy();
        outer_reverse->Destroy();
        inner_reverse->Destroy();
    }

    return 0;
}

} // namespace nikki_detail
