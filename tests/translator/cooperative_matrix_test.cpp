// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

namespace nikki_detail {

bool cooperative_matrix_translation_probe();

} // namespace nikki_detail

int main()
{
    return nikki_detail::cooperative_matrix_translation_probe() ? 0 : 1;
}
