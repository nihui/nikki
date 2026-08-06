// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

bool uses_push_constants(const tint::core::ir::Module& module)
{
    for (tint::core::ir::Instruction* instruction : *module.root_block)
    {
        const tint::core::ir::Var* var = instruction->As<tint::core::ir::Var>();
        const tint::core::type::Pointer* pointer = var ? var->Result()->Type()->As<tint::core::type::Pointer>() : 0;
        if (pointer && pointer->AddressSpace() == tint::core::AddressSpace::kImmediate)
            return true;
    }

    return false;
}

} // namespace nikki_detail
