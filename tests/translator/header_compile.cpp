// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/translator/translator_internal.h"

int nikki_translator_header_compile()
{
    nikki_detail::TranslationInput input;
    nikki_detail::TranslationResult result;
    return input.spirv.empty() && result.wgsl.empty() ? 0 : 1;
}
