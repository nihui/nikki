// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

int restore_cooperative_matrix_wgsl_types(
    std::string& wgsl, const CooperativeMatrixLegalization& info)
{
    for (size_t i = 0; i < info.component_type_substitutions.size(); i++)
    {
        const CooperativeMatrixLegalization::ComponentTypeSubstitution& substitution = info.component_type_substitutions[i];
        const char* kind = substitution.use == 0   ? "left"
                           : substitution.use == 1 ? "right"
                                                   : "result";
        const char* placeholder = substitution.is_signed ? "i32" : "u32";
        const char* component = substitution.is_signed ? "i8" : "u8";

        char source[96];
        char target[96];
        snprintf(source, sizeof(source), "subgroup_matrix_%s<%s, %u, %u>",
                 kind, placeholder, substitution.columns, substitution.rows);
        snprintf(target, sizeof(target), "subgroup_matrix_%s<%s, %u, %u>",
                 kind, component, substitution.columns, substitution.rows);

        bool replaced = false;
        for (size_t offset = wgsl.find(source); offset != std::string::npos;
             offset = wgsl.find(source, offset + strlen(target)))
        {
            wgsl.replace(offset, strlen(source), target);
            replaced = true;
        }
        if (!replaced)
        {
            log_error("WGSL cooperative matrix component type placeholder was not found");
            return -1;
        }
    }

    return 0;
}

} // namespace nikki_detail
