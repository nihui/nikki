// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_TRANSLATOR_STAGES_H
#define NIKKI_TRANSLATOR_STAGES_H

#include "translator_internal.h"

#include "spirv-tools/optimizer.hpp"
#ifndef SPV_ENABLE_UTILITY_CODE
#define SPV_ENABLE_UTILITY_CODE
#endif
#include "spirv/unified1/spirv.hpp11"
#include "src/tint/api/common/binding_point.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/ir/access.h"
#include "src/tint/lang/core/ir/block.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/ir/constant.h"
#include "src/tint/lang/core/ir/construct.h"
#include "src/tint/lang/core/ir/core_binary.h"
#include "src/tint/lang/core/ir/core_builtin_call.h"
#include "src/tint/lang/core/ir/core_unary.h"
#include "src/tint/lang/core/ir/if.h"
#include "src/tint/lang/core/ir/instruction.h"
#include "src/tint/lang/core/ir/let.h"
#include "src/tint/lang/core/ir/load.h"
#include "src/tint/lang/core/ir/loop.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/return.h"
#include "src/tint/lang/core/ir/store.h"
#include "src/tint/lang/core/ir/user_call.h"
#include "src/tint/lang/core/ir/var.h"
#include "src/tint/lang/core/ir/transform/change_immediate_to_uniform.h"
#include "src/tint/lang/core/ir/transform/dead_code_elimination.h"
#include "src/tint/lang/core/ir/transform/single_entry_point.h"
#include "src/tint/lang/core/ir/transform/substitute_overrides.h"
#include "src/tint/lang/core/type/array.h"
#include "src/tint/lang/core/type/bool.h"
#include "src/tint/lang/core/type/f16.h"
#include "src/tint/lang/core/type/f32.h"
#include "src/tint/lang/core/type/i8.h"
#include "src/tint/lang/core/type/i32.h"
#include "src/tint/lang/core/type/pointer.h"
#include "src/tint/lang/core/type/reference.h"
#include "src/tint/lang/core/type/struct.h"
#include "src/tint/lang/core/type/subgroup_matrix.h"
#include "src/tint/lang/core/type/u8.h"
#include "src/tint/lang/core/type/u32.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include <mutex>

namespace nikki_detail {

enum CooperativeMatrixMarker
{
    CooperativeMatrixMarkerLoad,
    CooperativeMatrixMarkerStore,
    CooperativeMatrixMarkerMulAdd,
    CooperativeMatrixMarkerConvert,
    CooperativeMatrixMarkerScalarAdd,
    CooperativeMatrixMarkerScalarSubtract,
    CooperativeMatrixMarkerScalarMultiply
};

struct CooperativeMatrixLegalization
{
    struct ComponentTypeSubstitution
    {
        uint32_t use;
        uint32_t rows;
        uint32_t columns;
        bool is_signed;
    };

    size_t load_count;
    size_t store_count;
    size_t muladd_count;
    size_t convert_count;
    size_t scalar_add_count;
    size_t scalar_subtract_count;
    size_t scalar_multiply_count;
    std::vector<ComponentTypeSubstitution> component_type_substitutions;

    CooperativeMatrixLegalization()
        : load_count(0), store_count(0), muladd_count(0), convert_count(0), scalar_add_count(0), scalar_subtract_count(0), scalar_multiply_count(0)
    {
    }
};

struct SubgroupBuiltinSubstitution
{
    tint::core::BuiltinValue placeholder;
    tint::core::BuiltinValue builtin;
};

inline std::once_flag g_tint_initialize_once;

int normalize_spirv(std::vector<uint32_t>& spirv);
void simplify_bounded_integer_loops(std::vector<uint32_t>& spirv);
int specialize_spirv(std::vector<uint32_t>& spirv, const VkSpecializationInfo* info);
int legalize_non_finite_fp16_constants(std::vector<uint32_t>& spirv);
int legalize_storage_buffer_16bit_integer_spirv(std::vector<uint32_t>& spirv);
std::string spirv_literal_string(const uint32_t* words, size_t word_count);
int legalize_packed_integer_dot_spirv(std::vector<uint32_t>& spirv, size_t& replacement_count);
bool spirv_instruction_has_result_type(spv::Op opcode);
int restore_packed_integer_dot_ir(tint::core::ir::Module& module, size_t expected_count);
int strengthen_atomic_compare_exchange_ir(tint::core::ir::Module& module);
int legalize_cooperative_matrix_spirv(std::vector<uint32_t>& spirv,
                                      CooperativeMatrixLegalization& legalization);
int restore_cooperative_matrix_ir(tint::core::ir::Module& module,
                                  const CooperativeMatrixLegalization& legalization,
                                  uint32_t subgroup_size);
int restore_cooperative_matrix_wgsl_types(
    std::string& wgsl, const CooperativeMatrixLegalization& legalization);
int legalize_uniform_global_invocation_z(std::vector<uint32_t>& spirv);

bool uses_push_constants(const tint::core::ir::Module& module);
void collect_storage_access(
    tint::core::ir::Module& module,
    std::unordered_map<uint64_t, ShaderBufferAccess>& storage_access,
    std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >& storage_writes);
void collect_exclusive_write_binding_pairs(
    const std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >& storage_writes,
    std::unordered_set<uint64_t>& binding_pairs);
int legalize_subgroup_id_builtins(
    tint::core::ir::Module& module,
    std::vector<SubgroupBuiltinSubstitution>& substitutions,
    bool& uses_linear_indexing_placeholder);
int restore_subgroup_id_builtins(
    std::string& wgsl, const std::vector<SubgroupBuiltinSubstitution>& substitutions);
bool spirv_uses_cooperative_matrix(const std::vector<uint32_t>& spirv);
bool spirv_uses_subgroup(const std::vector<uint32_t>& spirv);
bool can_reshape_workgroup(const std::vector<uint32_t>& spirv);
int reflect_shader(tint::Program& program, const char* entry_point,
                   const std::unordered_map<uint64_t, ShaderBufferAccess>& storage_access,
                   const TranslationInput& input, bool use_uniform,
                   TranslationResult& translated);

} // namespace nikki_detail

#endif // NIKKI_TRANSLATOR_STAGES_H
