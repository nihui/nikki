// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

// Matrix memory legalization uses bool layout operands internally. Convert them
// to the current core IR overloads only after offset/stride and scratch lowering.
// The final matrix types are already genuine f16/f32/i8/u8/i32/u32, so Tint's
// validator, inspector and WGSL writer agree on their memory requirements.
int normalize_cooperative_matrix_overloads(tint::core::ir::Module& module)
{
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::CoreBuiltinCall* call = instruction->As<tint::core::ir::CoreBuiltinCall>();
        if (!call)
            continue;
        const bool load = call->Func() == tint::core::BuiltinFn::kSubgroupMatrixLoad;
        if (!load && call->Func() != tint::core::BuiltinFn::kSubgroupMatrixStore)
            continue;
        const size_t layout_index = load ? 2 : 3;
        if (call->Args().size() != layout_index + 2)
        {
            log_error("unexpected cooperative matrix memory overload");
            return -1;
        }
        const tint::core::ir::Constant* layout = call->Args()[layout_index]->As<tint::core::ir::Constant>();
        if (!layout || !layout->Type()->Is<tint::core::type::Bool>())
        {
            log_error("cooperative matrix layout must be a constant bool");
            return -1;
        }
        const tint::core::Majorness majorness = layout->Value()->ValueAs<bool>()
                                                  ? tint::core::Majorness::kColMajor
                                                  : tint::core::Majorness::kRowMajor;
        tint::Vector<tint::core::ir::TemplateParameter, 2> templates;
        if (load)
            templates.Push(call->Result()->Type());
        templates.Push(majorness);
        tint::Vector<tint::core::ir::Value*, 4> arguments;
        for (size_t i = 0; i < call->Args().size(); i++)
        {
            if (i != layout_index)
                arguments.Push(call->Args()[i]);
        }
        call->SetExplicitTemplateParams(templates);
        call->SetOperands(arguments);
    }
    return 0;
}

} // namespace nikki_detail
