// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

using namespace tint::core::number_suffixes;
using namespace tint::core::fluent_types;

static const tint::core::type::Type* cooperative_matrix_ir_type(
    tint::core::ir::Module& module, const tint::core::type::Type* type,
    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*>& type_cache,
    bool& failed)
{
    if (!type || failed)
        return type;

    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*>::const_iterator cached = type_cache.find(type);
    if (cached != type_cache.end())
        return cached->second;

    const tint::core::type::Type* replacement = type;
    if (const tint::core::type::Struct* structure = type->As<tint::core::type::Struct>())
    {
        const std::string name = structure->Name().Name();
        const char prefix[] = "nikki_coopmat_";
        if (name.compare(0, sizeof(prefix) - 1, prefix) == 0)
        {
            uint32_t use = 0;
            uint32_t rows = 0;
            uint32_t columns = 0;
            uint32_t type_id = 0;
            uint32_t integer_width = 0;
            uint32_t integer_signed = 0;
            int consumed = 0;
            if (sscanf(name.c_str() + sizeof(prefix) - 1, "%u_%u_%u_%u_%u_%u%n",
                       &use, &rows, &columns, &type_id, &integer_width, &integer_signed, &consumed)
                    != 6
                || sizeof(prefix) - 1 + (size_t)consumed != name.size()
                || use > 2 || rows == 0 || columns == 0 || integer_signed > 1
                || structure->Members().Length() != 1)
            {
                failed = true;
                return type;
            }

            const tint::core::type::Type* component_type = structure->Members()[0]->Type();
            if (integer_width == 8)
            {
                // Tint's SPIR-V reader and WGSL IR writer cannot round-trip the
                // experimental i8/u8 matrix component types yet. Keep their
                // scalar shader representation here and restore the component
                // spelling after Tint has generated otherwise-valid WGSL.
                if (!component_type->IsAnyOf<tint::core::type::I32, tint::core::type::U32>())
                {
                    failed = true;
                    return type;
                }
            }
            else if (integer_width != 0)
            {
                if (integer_width != 32)
                {
                    failed = true;
                    return type;
                }
            }

            if (component_type->Is<tint::core::type::F16>())
                module.properties.Add(tint::core::ir::Property::kAllow16BitFloats);
            else if (!component_type->Is<tint::core::type::F32>()
                     && !component_type->Is<tint::core::type::I32>()
                     && !component_type->Is<tint::core::type::U32>())
            {
                log_error("cooperative matrix component type %s is not supported by WebGPU",
                          component_type->FriendlyName().c_str());
                failed = true;
                return type;
            }

            const tint::core::SubgroupMatrixKind kind = use == 0   ? tint::core::SubgroupMatrixKind::kLeft
                                                        : use == 1 ? tint::core::SubgroupMatrixKind::kRight
                                                                   : tint::core::SubgroupMatrixKind::kResult;
            replacement = module.Types().subgroup_matrix(kind, component_type, columns, rows);
        }
    }
    else if (const tint::core::type::Array* array = type->As<tint::core::type::Array>())
    {
        const tint::core::type::Type* element = cooperative_matrix_ir_type(module, array->ElemType(), type_cache, failed);
        if (element != array->ElemType())
        {
            const std::optional<uint32_t> count = array->ConstantCount();
            if (!count.has_value())
            {
                log_error("runtime arrays of cooperative matrices are not supported");
                failed = true;
                return type;
            }
            replacement = module.Types().array(element, count.value());
        }
    }
    else if (const tint::core::type::Pointer* pointer = type->As<tint::core::type::Pointer>())
    {
        const tint::core::type::Type* store_type = cooperative_matrix_ir_type(module, pointer->StoreType(), type_cache, failed);
        if (store_type != pointer->StoreType())
            replacement = module.Types().ptr(pointer->AddressSpace(), store_type, pointer->Access());
    }
    else if (const tint::core::type::Reference* reference = type->As<tint::core::type::Reference>())
    {
        const tint::core::type::Type* store_type = cooperative_matrix_ir_type(module, reference->StoreType(), type_cache, failed);
        if (store_type != reference->StoreType())
            replacement = module.Types().ref(reference->AddressSpace(), store_type, reference->Access());
    }

    type_cache[type] = replacement;
    return replacement;
}

static int cooperative_matrix_memory_view(tint::core::ir::Module& module,
                                          tint::core::ir::Builder& builder,
                                          tint::core::ir::Instruction* insertion_point,
                                          tint::core::ir::Value* pointer,
                                          tint::core::ir::Value*& base,
                                          tint::core::ir::Value*& offset)
{
    tint::core::ir::InstructionResult* pointer_result = pointer->As<tint::core::ir::InstructionResult>();
    tint::core::ir::Access* access = pointer_result ? pointer_result->Instruction()->As<tint::core::ir::Access>() : 0;
    if (!access || access->Indices().empty())
    {
        log_error("cooperative matrix load/store pointer is not an indexed array element");
        return -1;
    }

    const tint::core::type::Pointer* object_pointer = access->Object()->Type()->As<tint::core::type::Pointer>();
    if (!object_pointer)
        return -1;

    const size_t prefix_count = access->Indices().size() - 1;
    const tint::core::type::Type* prefix_type = object_pointer->StoreType();
    tint::Vector<tint::core::ir::Value*, 4> prefix_indices;
    for (size_t i = 0; i < prefix_count; i++)
    {
        tint::core::ir::Value* index = access->Indices()[i];
        prefix_indices.Push(index);
        if (const tint::core::type::Array* array = prefix_type->As<tint::core::type::Array>())
        {
            prefix_type = array->ElemType();
        }
        else if (const tint::core::type::Struct* structure = prefix_type->As<tint::core::type::Struct>())
        {
            const tint::core::ir::Constant* constant = index->As<tint::core::ir::Constant>();
            const uint32_t member = constant ? constant->Value()->ValueAs<uint32_t>() : UINT32_MAX;
            if (member >= structure->Members().Length())
                return -1;
            prefix_type = structure->Members()[member]->Type();
        }
        else
        {
            return -1;
        }
    }

    if (!prefix_type->Is<tint::core::type::Array>())
    {
        log_error("cooperative matrix load/store backing object is not an array");
        return -1;
    }

    base = access->Object();
    if (prefix_count != 0)
    {
        const tint::core::type::Pointer* base_type = module.Types().ptr(
            object_pointer->AddressSpace(), prefix_type, object_pointer->Access());
        builder.InsertBefore(insertion_point, [&]() {
            base = builder.Access(base_type, access->Object(), prefix_indices)->Result();
        });
    }
    offset = access->Indices().back();
    return 0;
}

static bool cooperative_matrix_layout(tint::core::ir::Value* value, bool& col_major)
{
    const tint::core::ir::Constant* constant = value->As<tint::core::ir::Constant>();
    if (!constant)
        return false;

    const uint32_t layout = constant->Value()->ValueAs<uint32_t>();
    if (layout > 1)
        return false;
    col_major = layout == 1;
    return true;
}

static bool is_vec2u(const tint::core::type::Type* type)
{
    const tint::core::type::Vector* vector = type ? type->As<tint::core::type::Vector>() : 0;
    return vector && vector->Width() == 2 && vector->Type()->Is<tint::core::type::U32>();
}

static bool is_vec4i(const tint::core::type::Type* type)
{
    const tint::core::type::Vector* vector = type ? type->As<tint::core::type::Vector>() : 0;
    return vector && vector->Width() == 4 && vector->Type()->Is<tint::core::type::I32>();
}

static bool is_vec4u(const tint::core::type::Type* type)
{
    const tint::core::type::Vector* vector = type ? type->As<tint::core::type::Vector>() : 0;
    return vector && vector->Width() == 4 && vector->Type()->Is<tint::core::type::U32>();
}

static bool is_vec4f(const tint::core::type::Type* type)
{
    const tint::core::type::Vector* vector = type ? type->As<tint::core::type::Vector>() : 0;
    return vector && vector->Width() == 4 && vector->Type()->Is<tint::core::type::F32>();
}

static const tint::core::type::Type* memory_store_type(const tint::core::type::Type* type)
{
    if (const tint::core::type::Pointer* pointer = type ? type->As<tint::core::type::Pointer>() : 0)
        return pointer->StoreType();
    if (const tint::core::type::Reference* reference = type ? type->As<tint::core::type::Reference>() : 0)
        return reference->StoreType();
    return type;
}

static const tint::core::type::Type* replace_memory_store_type(
    tint::core::ir::Module& module, const tint::core::type::Type* memory_type,
    const tint::core::type::Type* store_type)
{
    if (const tint::core::type::Pointer* pointer = memory_type->As<tint::core::type::Pointer>())
        return module.Types().ptr(pointer->AddressSpace(), store_type, pointer->Access());
    if (const tint::core::type::Reference* reference = memory_type->As<tint::core::type::Reference>())
        return module.Types().ref(reference->AddressSpace(), store_type, reference->Access());
    return store_type;
}

static tint::core::ir::Var* cooperative_matrix_memory_root(tint::core::ir::Value* value)
{
    tint::core::ir::InstructionResult* result = value ? value->As<tint::core::ir::InstructionResult>() : 0;
    if (!result)
        return 0;

    tint::core::ir::Instruction* instruction = result->Instruction();
    if (tint::core::ir::Var* var = instruction->As<tint::core::ir::Var>())
        return var;
    if (tint::core::ir::Access* access = instruction->As<tint::core::ir::Access>())
        return cooperative_matrix_memory_root(access->Object());
    if (tint::core::ir::CoreUnary* unary = instruction->As<tint::core::ir::CoreUnary>())
        return cooperative_matrix_memory_root(unary->Val());
    if (tint::core::ir::Let* let = instruction->As<tint::core::ir::Let>())
        return cooperative_matrix_memory_root(let->Value());
    return 0;
}

enum PackedCooperativeMatrixMemoryFormat
{
    PackedCooperativeMatrixMemoryNone,
    PackedCooperativeMatrixMemoryF16Vec2U,
    PackedCooperativeMatrixMemoryF32Vec4F,
    PackedCooperativeMatrixMemoryI32Vec4I,
    PackedCooperativeMatrixMemoryU32Vec4U
};

static bool packed_cooperative_matrix_vector(
    const tint::core::type::Type* type,
    PackedCooperativeMatrixMemoryFormat format)
{
    if (format == PackedCooperativeMatrixMemoryF16Vec2U)
        return is_vec2u(type);
    if (format == PackedCooperativeMatrixMemoryF32Vec4F)
        return is_vec4f(type);
    if (format == PackedCooperativeMatrixMemoryI32Vec4I)
        return is_vec4i(type);
    if (format == PackedCooperativeMatrixMemoryU32Vec4U)
        return is_vec4u(type);

    return false;
}

static const tint::core::type::Type* packed_cooperative_matrix_scalar(
    tint::core::ir::Module& module,
    PackedCooperativeMatrixMemoryFormat format)
{
    if (format == PackedCooperativeMatrixMemoryF16Vec2U)
        return module.Types().f16();
    if (format == PackedCooperativeMatrixMemoryF32Vec4F)
        return module.Types().f32();
    if (format == PackedCooperativeMatrixMemoryI32Vec4I)
        return module.Types().i32();

    return module.Types().u32();
}

static const tint::core::type::Type* scalarize_packed_cooperative_matrix_type(
    tint::core::ir::Module& module, const tint::core::type::Type* type,
    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*>& type_cache,
    PackedCooperativeMatrixMemoryFormat format, bool& changed, bool& failed)
{
    if (!type || failed)
        return type;

    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*>::const_iterator cached = type_cache.find(type);
    if (cached != type_cache.end())
        return cached->second;

    const tint::core::type::Type* replacement = type;
    if (const tint::core::type::Array* array = type->As<tint::core::type::Array>())
    {
        if (packed_cooperative_matrix_vector(array->ElemType(), format))
        {
            const std::optional<uint32_t> count = array->ConstantCount();
            if (count.has_value())
            {
                if (count.value() > UINT32_MAX / 4)
                {
                    failed = true;
                    return type;
                }
                replacement = module.Types().array(
                    packed_cooperative_matrix_scalar(module, format), count.value() * 4);
            }
            else
            {
                replacement = module.Types().runtime_array(
                    packed_cooperative_matrix_scalar(module, format));
            }
            if (format == PackedCooperativeMatrixMemoryF16Vec2U)
                module.properties.Add(tint::core::ir::Property::kAllow16BitFloats);
            changed = true;
        }
        else
        {
            const tint::core::type::Type* element = scalarize_packed_cooperative_matrix_type(
                module, array->ElemType(), type_cache, format, changed, failed);
            if (element != array->ElemType())
            {
                const std::optional<uint32_t> count = array->ConstantCount();
                if (!count.has_value())
                {
                    failed = true;
                    return type;
                }
                replacement = module.Types().array(element, count.value());
            }
        }
    }
    else if (const tint::core::type::Struct* structure = type->As<tint::core::type::Struct>())
    {
        tint::Vector<const tint::core::type::StructMember*, 4> members;
        bool structure_changed = false;
        for (const tint::core::type::StructMember* member : structure->Members())
        {
            bool member_changed = false;
            const tint::core::type::Type* member_type = scalarize_packed_cooperative_matrix_type(
                module, member->Type(), type_cache, format, member_changed, failed);
            structure_changed |= member_changed;
            tint::core::type::StructMember* new_member = module.Types().Get<tint::core::type::StructMember>(
                member->Name(), member_type, member->Index(), member->Offset(), member_type->Align(),
                member_type->Size(), member->Attributes());
            if (member->RowMajor())
                new_member->SetRowMajor();
            if (member->HasMatrixStride())
                new_member->SetMatrixStride(member->MatrixStride());
            members.Push(new_member);
        }
        if (structure_changed)
        {
            const char* suffix = format == PackedCooperativeMatrixMemoryF16Vec2U   ? "_nikki_f16"
                                 : format == PackedCooperativeMatrixMemoryF32Vec4F ? "_nikki_f32"
                                 : format == PackedCooperativeMatrixMemoryI32Vec4I ? "_nikki_i32"
                                                                                   : "_nikki_u32";
            const std::string name = structure->Name().Name() + suffix;
            tint::core::type::Struct* new_structure = module.Types().Struct(
                module.symbols.New(name), std::move(members));
            for (tint::core::type::StructFlag flag : structure->StructFlags())
                new_structure->SetStructFlag(flag);
            for (tint::core::AddressSpace address_space : structure->AddressSpaceUsage())
                new_structure->AddUsage(address_space);
            for (tint::core::type::PipelineStageUsage usage : structure->PipelineStageUses())
                new_structure->AddUsage(usage);
            replacement = new_structure;
            changed = true;
        }
    }
    else if (const tint::core::type::Pointer* pointer = type->As<tint::core::type::Pointer>())
    {
        const tint::core::type::Type* store_type = scalarize_packed_cooperative_matrix_type(
            module, pointer->StoreType(), type_cache, format, changed, failed);
        if (store_type != pointer->StoreType())
            replacement = module.Types().ptr(pointer->AddressSpace(), store_type, pointer->Access());
    }
    else if (const tint::core::type::Reference* reference = type->As<tint::core::type::Reference>())
    {
        const tint::core::type::Type* store_type = scalarize_packed_cooperative_matrix_type(
            module, reference->StoreType(), type_cache, format, changed, failed);
        if (store_type != reference->StoreType())
            replacement = module.Types().ref(reference->AddressSpace(), store_type, reference->Access());
    }

    type_cache[type] = replacement;
    return replacement;
}

struct CooperativeMatrixMemoryUse
{
    tint::core::ir::CoreBuiltinCall* call;
    tint::core::ir::Var* root;
    uint32_t scale;
    PackedCooperativeMatrixMemoryFormat format;
    bool zero_stride;
};

static bool cooperative_matrix_is_8bit_placeholder(
    const tint::core::type::SubgroupMatrix* matrix,
    const CooperativeMatrixLegalization& info)
{
    if (!matrix || !matrix->Type()->IsAnyOf<tint::core::type::I32, tint::core::type::U32>())
        return false;

    const uint32_t use = matrix->Kind() == tint::core::SubgroupMatrixKind::kLeft
                             ? 0
                         : matrix->Kind() == tint::core::SubgroupMatrixKind::kRight ? 1
                                                                                    : 2;
    const bool is_signed = matrix->Type()->Is<tint::core::type::I32>();
    for (size_t i = 0; i < info.component_type_substitutions.size(); i++)
    {
        const CooperativeMatrixLegalization::ComponentTypeSubstitution& substitution = info.component_type_substitutions[i];
        if (substitution.use == use && substitution.rows == matrix->Rows()
            && substitution.columns == matrix->Columns()
            && substitution.is_signed == is_signed)
            return true;
    }
    return false;
}

struct ScalarizedAccess
{
    tint::core::ir::Value* replacement;
    tint::core::ir::Value* array_base;
    tint::core::ir::Value* scalar_index;
    uint32_t scalar_count;

    ScalarizedAccess()
        : replacement(0), array_base(0), scalar_index(0), scalar_count(0)
    {
    }
};

static tint::core::ir::Value* scale_array_index(tint::core::ir::Builder& builder,
                                                tint::core::ir::Value* index,
                                                uint32_t scale)
{
    if (index->Type()->Is<tint::core::type::I32>())
        return builder.Multiply(index, builder.Constant(tint::core::i32(scale)))->Result();
    return builder.Multiply(index, builder.Constant(tint::core::u32(scale)))->Result();
}

static tint::core::ir::Value* add_array_index(tint::core::ir::Builder& builder,
                                              tint::core::ir::Value* index,
                                              uint32_t addend)
{
    if (addend == 0)
        return index;
    if (index->Type()->Is<tint::core::type::I32>())
        return builder.Add(index, builder.Constant(tint::core::i32(addend)))->Result();
    return builder.Add(index, builder.Constant(tint::core::u32(addend)))->Result();
}

static int propagate_scalarized_memory_type(
    tint::core::ir::Module& module, tint::core::ir::Value* value,
    const tint::core::type::Type* replacement_store_type)
{
    std::vector<tint::core::ir::Value*> worklist(1, value);
    std::unordered_set<tint::core::ir::Value*> visited;
    while (!worklist.empty())
    {
        tint::core::ir::Value* current = worklist.back();
        worklist.pop_back();
        if (!visited.insert(current).second)
            continue;

        const tint::Vector<tint::core::ir::Usage, 4> usages = current->UsagesSorted();
        for (size_t i = 0; i < usages.Length(); i++)
        {
            tint::core::ir::Instruction* instruction = usages[i].instruction;
            if (tint::core::ir::CoreUnary* unary = instruction->As<tint::core::ir::CoreUnary>())
            {
                if (unary->Op() != tint::core::UnaryOp::kAddressOf
                    && unary->Op() != tint::core::UnaryOp::kIndirection)
                    return -1;
                unary->Result()->SetType(replace_memory_store_type(
                    module, unary->Result()->Type(), replacement_store_type));
                worklist.push_back(unary->Result());
            }
            else if (tint::core::ir::Let* let = instruction->As<tint::core::ir::Let>())
            {
                let->Result()->SetType(replace_memory_store_type(
                    module, let->Result()->Type(), replacement_store_type));
                worklist.push_back(let->Result());
            }
        }
    }
    return 0;
}

static int scalarize_packed_cooperative_matrix_backing(
    tint::core::ir::Module& module,
    const std::vector<CooperativeMatrixMemoryUse>& memory_uses,
    PackedCooperativeMatrixMemoryFormat format)
{
    std::unordered_set<tint::core::ir::Var*> roots;
    for (size_t i = 0; i < memory_uses.size(); i++)
    {
        if (memory_uses[i].format == format && memory_uses[i].root)
            roots.insert(memory_uses[i].root);
    }
    if (roots.empty())
        return 0;

    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*> type_cache;
    std::unordered_map<tint::core::ir::Var*, const tint::core::type::Type*> root_types;
    bool failed = false;
    for (std::unordered_set<tint::core::ir::Var*>::const_iterator it = roots.begin(); it != roots.end(); ++it)
    {
        bool changed = false;
        const tint::core::type::Type* replacement = scalarize_packed_cooperative_matrix_type(
            module, (*it)->Result()->Type(), type_cache, format, changed, failed);
        if (failed || replacement == (*it)->Result()->Type())
        {
            log_error("cooperative matrix packed backing type cannot be scalarized");
            return -1;
        }
        root_types[*it] = replacement;
    }

    std::vector<tint::core::ir::Access*> accesses;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::Access* access = instruction->As<tint::core::ir::Access>();
        if (access && roots.count(cooperative_matrix_memory_root(access->Object())) != 0)
            accesses.push_back(access);
    }

    tint::core::ir::Builder builder(module);
    std::unordered_map<tint::core::ir::Access*, ScalarizedAccess> rewritten;
    std::function<int(tint::core::ir::Access*)> rewrite_access = [&](tint::core::ir::Access* access) -> int {
        if (rewritten.count(access) != 0)
            return 0;

        tint::core::ir::Value* old_object = access->Object();
        tint::core::ir::Value* new_object = old_object;
        if (tint::core::ir::InstructionResult* object_result = old_object->As<tint::core::ir::InstructionResult>())
        {
            if (tint::core::ir::Access* object_access = object_result->Instruction()->As<tint::core::ir::Access>())
            {
                if (rewrite_access(object_access) != 0)
                    return -1;
                new_object = rewritten[object_access].replacement;
            }
        }

        const tint::core::type::Type* current_type = memory_store_type(old_object->Type());
        tint::Vector<tint::core::ir::Value*, 8> prefix_indices;
        ScalarizedAccess item;
        for (size_t i = 0; i < access->Indices().size(); i++)
        {
            tint::core::ir::Value* index = access->Indices()[i];
            if (const tint::core::type::Array* array = current_type->As<tint::core::type::Array>())
            {
                if (packed_cooperative_matrix_vector(array->ElemType(), format))
                {
                    const size_t remaining = access->Indices().size() - i - 1;
                    if (remaining > 1)
                        return -1;

                    const tint::core::type::Type* scalar_type = packed_cooperative_matrix_scalar(module, format);
                    const tint::core::type::Array* scalar_array = array->ConstantCount().has_value()
                                                                      ? module.Types().array(scalar_type, array->ConstantCount().value() * 4)
                                                                      : module.Types().runtime_array(scalar_type);
                    if (prefix_indices.IsEmpty())
                    {
                        item.array_base = new_object;
                    }
                    else
                    {
                        const tint::core::type::Type* base_type = replace_memory_store_type(
                            module, old_object->Type(), scalar_array);
                        builder.InsertBefore(access, [&]() {
                            item.array_base = builder.Access(base_type, new_object, prefix_indices)->Result();
                        });
                    }

                    builder.InsertBefore(access, [&]() {
                        item.scalar_index = scale_array_index(builder, index, 4);
                        item.scalar_count = 4;
                        if (remaining == 1)
                        {
                            const tint::core::ir::Constant* component = access->Indices()[i + 1]->As<tint::core::ir::Constant>();
                            if (!component)
                                return;
                            const uint32_t component_index = component->Value()->ValueAs<uint32_t>();
                            const uint32_t vector_width = format == PackedCooperativeMatrixMemoryF16Vec2U ? 2 : 4;
                            if (component_index >= vector_width)
                                return;
                            const uint32_t scalars_per_component = format == PackedCooperativeMatrixMemoryF16Vec2U ? 2 : 1;
                            item.scalar_index = add_array_index(
                                builder, item.scalar_index,
                                component_index * scalars_per_component);
                            item.scalar_count = scalars_per_component;
                        }
                        const tint::core::type::Type* scalar_memory_type = replace_memory_store_type(
                            module, old_object->Type(), scalar_type);
                        item.replacement = builder.Access(
                                                      scalar_memory_type, item.array_base, item.scalar_index)
                                               ->Result();
                    });
                    if (!item.replacement || item.scalar_count == 0)
                        return -1;
                    rewritten[access] = item;
                    return 0;
                }
                current_type = array->ElemType();
            }
            else if (const tint::core::type::Struct* structure = current_type->As<tint::core::type::Struct>())
            {
                const tint::core::ir::Constant* member_index = index->As<tint::core::ir::Constant>();
                const uint32_t member = member_index ? member_index->Value()->ValueAs<uint32_t>() : UINT32_MAX;
                if (member >= structure->Members().Length())
                    return -1;
                current_type = structure->Members()[member]->Type();
            }
            else
            {
                return -1;
            }
            prefix_indices.Push(index);
        }

        bool changed = false;
        const tint::core::type::Type* result_type = scalarize_packed_cooperative_matrix_type(
            module, access->Result()->Type(), type_cache, format, changed, failed);
        if (failed)
            return -1;
        builder.InsertBefore(access, [&]() {
            item.replacement = builder.Access(
                                          result_type, new_object, tint::ToVector<8>(access->Indices()))
                                   ->Result();
        });
        rewritten[access] = item;
        return 0;
    };

    for (size_t i = 0; i < accesses.size(); i++)
    {
        if (rewrite_access(accesses[i]) != 0)
        {
            log_error("cooperative matrix packed access cannot be scalarized");
            return -1;
        }
    }

    for (std::unordered_map<tint::core::ir::Var*, const tint::core::type::Type*>::const_iterator it = root_types.begin();
         it != root_types.end(); ++it)
        it->first->Result()->SetType(it->second);

    for (size_t i = 0; i < accesses.size(); i++)
    {
        tint::core::ir::Access* access = accesses[i];
        ScalarizedAccess& item = rewritten[access];
        if (item.scalar_count == 0)
            continue;

        const tint::Vector<tint::core::ir::Usage, 4> usages = access->Result()->UsagesSorted();
        for (size_t j = 0; j < usages.Length(); j++)
        {
            tint::core::ir::Instruction* instruction = usages[j].instruction;
            if (tint::core::ir::Load* load = instruction->As<tint::core::ir::Load>())
            {
                tint::core::ir::InstructionResult* result = load->DetachResult();
                builder.InsertBefore(load, [&]() {
                    tint::Vector<tint::core::ir::Value*, 4> components;
                    const tint::core::type::Type* scalar_type = packed_cooperative_matrix_scalar(module, format);
                    for (uint32_t k = 0; k < item.scalar_count; k++)
                    {
                        tint::core::ir::Value* scalar_index = add_array_index(builder, item.scalar_index, k);
                        const tint::core::type::Type* scalar_memory_type = replace_memory_store_type(
                            module, item.replacement->Type(), scalar_type);
                        tint::core::ir::Access* scalar = builder.Access(
                            scalar_memory_type, item.array_base, scalar_index);
                        components.Push(builder.Load(scalar)->Result());
                    }
                    if (format == PackedCooperativeMatrixMemoryF16Vec2U)
                    {
                        const tint::core::type::Type* vector_type = item.scalar_count == 4
                                                                        ? (const tint::core::type::Type*)module.Types().vec4h()
                                                                        : (const tint::core::type::Type*)module.Types().vec2h();
                        tint::core::ir::Construct* packed = builder.Construct(vector_type, components);
                        builder.BitcastWithResult(result, packed);
                    }
                    else if (item.scalar_count == 4)
                    {
                        builder.ConstructWithResult(result, components);
                    }
                    else
                    {
                        const tint::core::type::Type* scalar_memory_type = replace_memory_store_type(
                            module, item.replacement->Type(), scalar_type);
                        tint::core::ir::Access* scalar = builder.Access(
                            scalar_memory_type, item.array_base, item.scalar_index);
                        builder.LoadWithResult(result, scalar);
                    }
                });
                load->Destroy();
            }
            else if (tint::core::ir::Store* store = instruction->As<tint::core::ir::Store>())
            {
                builder.InsertBefore(store, [&]() {
                    const tint::core::type::Type* scalar_type = packed_cooperative_matrix_scalar(module, format);
                    tint::core::ir::Value* unpacked = store->From();
                    if (format == PackedCooperativeMatrixMemoryF16Vec2U)
                    {
                        const tint::core::type::Type* vector_type = item.scalar_count == 4
                                                                        ? (const tint::core::type::Type*)module.Types().vec4h()
                                                                        : (const tint::core::type::Type*)module.Types().vec2h();
                        unpacked = builder.Bitcast(vector_type, unpacked)->Result();
                    }
                    for (uint32_t k = 0; k < item.scalar_count; k++)
                    {
                        tint::core::ir::Value* scalar_index = add_array_index(builder, item.scalar_index, k);
                        const tint::core::type::Type* scalar_memory_type = replace_memory_store_type(
                            module, item.replacement->Type(), scalar_type);
                        tint::core::ir::Access* scalar = builder.Access(
                            scalar_memory_type, item.array_base, scalar_index);
                        tint::core::ir::Value* component = unpacked;
                        if (item.scalar_count != 1)
                            component = builder.Access(
                                                   scalar_type, unpacked,
                                                   builder.Constant(tint::core::u32(k)))
                                            ->Result();
                        builder.Store(scalar, component);
                    }
                });
                store->Destroy();
            }
        }

        if (propagate_scalarized_memory_type(
                module, access->Result(), packed_cooperative_matrix_scalar(module, format))
            != 0)
            return -1;
        access->Result()->ReplaceAllUsesWith(item.replacement);
    }

    for (size_t i = 0; i < accesses.size(); i++)
    {
        ScalarizedAccess& item = rewritten[accesses[i]];
        if (item.scalar_count != 0)
            continue;
        const tint::core::type::Type* store_type = memory_store_type(item.replacement->Type());
        if (propagate_scalarized_memory_type(module, accesses[i]->Result(), store_type) != 0)
            return -1;
        accesses[i]->Result()->ReplaceAllUsesWith(item.replacement);
    }

    for (size_t i = accesses.size(); i > 0; i--)
    {
        tint::core::ir::Access* access = accesses[i - 1];
        if (access->Result()->NumUsages() != 0)
        {
            log_error("cooperative matrix packed access still has live uses");
            return -1;
        }
        access->Destroy();
    }

    return 0;
}

static void scale_packed_cooperative_matrix_memory_uses(
    tint::core::ir::Module& module,
    const std::vector<CooperativeMatrixMemoryUse>& memory_uses)
{
    tint::core::ir::Builder builder(module);
    for (size_t i = 0; i < memory_uses.size(); i++)
    {
        const CooperativeMatrixMemoryUse& use = memory_uses[i];
        if (!use.call || use.scale == 1)
            continue;

        const size_t stride_index = use.call->Func() == tint::core::BuiltinFn::kSubgroupMatrixLoad ? 3 : 4;
        builder.InsertBefore(use.call, [&]() {
            use.call->SetArg(1, scale_array_index(builder, use.call->Args()[1], use.scale));
            use.call->SetArg(stride_index, scale_array_index(builder, use.call->Args()[stride_index], use.scale));
        });
    }
}

static tint::core::ir::Value* cooperative_matrix_compute_builtin(
    tint::core::ir::Module& module, tint::core::ir::Builder& builder,
    tint::core::ir::Function* function, tint::core::BuiltinValue builtin,
    const char* name)
{
    for (tint::core::ir::FunctionParam* parameter : function->Params())
    {
        if (parameter->Builtin() == builtin)
            return parameter;

        const tint::core::type::Struct* structure = parameter->Type()->As<tint::core::type::Struct>();
        if (!structure)
            continue;

        for (const tint::core::type::StructMember* member : structure->Members())
        {
            if (member->Attributes().builtin != builtin)
                continue;

            tint::core::ir::Access* access = builder.Access(
                module.Types().u32(), parameter, tint::core::u32(member->Index()));
            access->InsertBefore(function->Block()->Front());
            return access->Result();
        }
    }

    tint::core::ir::FunctionParam* parameter = builder.FunctionParam(name, module.Types().u32());
    parameter->SetBuiltin(builtin);
    function->AppendParam(parameter);
    return parameter;
}

static bool cooperative_matrix_constant_u32(tint::core::ir::Value* value, uint32_t& result)
{
    const tint::core::ir::Constant* constant = value ? value->As<tint::core::ir::Constant>() : 0;
    if (!constant || !constant->Type()->IsAnyOf<tint::core::type::I32, tint::core::type::U32>())
        return false;

    result = constant->Value()->ValueAs<uint32_t>();
    return true;
}

// Vulkan permits a zero cooperative-matrix stride. ncnn uses this to load a
// bias vector and broadcast it over all rows or columns. WebGPU clamps a matrix
// stride to the minor dimension, so materialize the broadcast into workgroup
// memory before issuing the WebGPU load.
static int emulate_zero_stride_f16_cooperative_matrix_loads(
    tint::core::ir::Module& module,
    const std::vector<CooperativeMatrixMemoryUse>& memory_uses,
    uint32_t subgroup_size)
{
    std::vector<tint::core::ir::CoreBuiltinCall*> loads;
    uint32_t maximum_matrix_elements = 0;
    for (size_t i = 0; i < memory_uses.size(); i++)
    {
        tint::core::ir::CoreBuiltinCall* call = memory_uses[i].call;
        if (memory_uses[i].format != PackedCooperativeMatrixMemoryF16Vec2U || !call
            || call->Func() != tint::core::BuiltinFn::kSubgroupMatrixLoad)
            continue;

        if (!memory_uses[i].zero_stride)
            continue;

        const tint::core::type::SubgroupMatrix* matrix = call->Result()->Type()->As<tint::core::type::SubgroupMatrix>();
        if (!matrix || !matrix->Type()->Is<tint::core::type::F16>())
            return -1;
        if (matrix->Rows() > UINT32_MAX / matrix->Columns())
            return -1;

        maximum_matrix_elements = std::max(maximum_matrix_elements,
                                           matrix->Rows() * matrix->Columns());
        loads.push_back(call);
    }
    if (loads.empty())
        return 0;
    if (subgroup_size == 0)
    {
        log_error("zero-stride cooperative matrix load requires a known subgroup size");
        return -1;
    }

    tint::core::ir::Function* compute = 0;
    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        if (!module.functions[i]->IsCompute())
            continue;
        if (compute)
        {
            log_error("zero-stride cooperative matrix load requires one compute entry point");
            return -1;
        }
        compute = module.functions[i];
    }
    if (!compute || !compute->WorkgroupSizeAsConst().has_value())
    {
        log_error("zero-stride cooperative matrix load requires a constant workgroup size");
        return -1;
    }

    const std::array<uint32_t, 3> workgroup_size = compute->WorkgroupSizeAsConst().value();
    const uint64_t invocation_count = (uint64_t)workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    const uint64_t subgroup_count = (invocation_count + subgroup_size - 1) / subgroup_size;
    const uint64_t scratch_count = subgroup_count * maximum_matrix_elements;
    if (subgroup_count == 0 || scratch_count == 0 || scratch_count > UINT32_MAX)
        return -1;

    tint::core::ir::Builder builder(module);
    tint::core::ir::Value* subgroup_invocation_id = cooperative_matrix_compute_builtin(
        module, builder, compute, tint::core::BuiltinValue::kSubgroupInvocationId,
        "nikki_subgroup_invocation_id");
    tint::core::ir::Value* subgroup_id = cooperative_matrix_compute_builtin(
        module, builder, compute, tint::core::BuiltinValue::kSubgroupId,
        "nikki_subgroup_id");

    const tint::core::type::Array* scratch_type = module.Types().array(
        module.Types().f16(), (uint32_t)scratch_count);
    tint::core::ir::Var* scratch = builder.Var(
        "nikki_cooperative_matrix_broadcast",
        module.Types().ptr<tint::core::AddressSpace::kWorkgroup>(scratch_type));
    module.root_block->Append(scratch);

    for (size_t i = 0; i < loads.size(); i++)
    {
        tint::core::ir::CoreBuiltinCall* call = loads[i];
        const tint::core::type::SubgroupMatrix* matrix = call->Result()->Type()->As<tint::core::type::SubgroupMatrix>();
        const bool col_major = call->Args()[2]->As<tint::core::ir::Constant>()->Value()->ValueAs<bool>();
        const uint32_t minor_size = col_major ? matrix->Rows() : matrix->Columns();
        const uint32_t matrix_elements = matrix->Rows() * matrix->Columns();
        tint::core::ir::Value* source = call->Args()[0];
        tint::core::ir::Value* source_offset = call->Args()[1];

        builder.InsertBefore(call, [&]() {
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kWorkgroupBarrier);
            tint::core::ir::Value* subgroup_base = builder.Multiply(
                                                              subgroup_id, builder.Constant(tint::core::u32(maximum_matrix_elements)))
                                                       ->Result();
            const std::function<void(tint::core::ir::Value*)> copy_element =
                [&](tint::core::ir::Value* index) {
                    tint::core::ir::Value* minor_index = builder.Modulo(
                                                                    index, builder.Constant(tint::core::u32(minor_size)))
                                                             ->Result();
                    tint::core::ir::Value* read_index = builder.Add(source_offset, minor_index)->Result();
                    const tint::core::type::Type* source_element_type = replace_memory_store_type(
                        module, source->Type(), module.Types().f16());
                    tint::core::ir::Access* read_pointer = builder.Access(
                        source_element_type, source, read_index);
                    tint::core::ir::Value* write_index = builder.Add(subgroup_base, index)->Result();
                    tint::core::ir::Access* write_pointer = builder.Access(
                        module.Types().ptr<tint::core::AddressSpace::kWorkgroup>(module.Types().f16()),
                        scratch->Result(), write_index);
                    builder.Store(write_pointer, builder.Load(read_pointer));
                };
            for (uint32_t first = 0; first < matrix_elements; first += subgroup_size)
            {
                tint::core::ir::Value* index = first == 0
                                                   ? subgroup_invocation_id
                                                   : builder.Add(subgroup_invocation_id,
                                                                 builder.Constant(tint::core::u32(first)))
                                                         ->Result();
                if (matrix_elements - first >= subgroup_size)
                {
                    copy_element(index);
                }
                else
                {
                    tint::core::ir::If* branch = builder.If(builder.LessThan(
                        index, builder.Constant(tint::core::u32(matrix_elements))));
                    builder.Append(branch->True(), [&]() {
                        copy_element(index);
                        builder.ExitIf(branch);
                    });
                }
            }
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kWorkgroupBarrier);
            call->SetArg(0, scratch->Result());
            call->SetArg(1, subgroup_base);
            call->SetArg(3, builder.Constant(tint::core::u32(minor_size)));
        });
    }

    return 0;
}

static int emulate_f16_f32_cooperative_matrix_conversions(
    tint::core::ir::Module& module,
    const std::vector<tint::core::ir::UserCall*>& conversions,
    uint32_t subgroup_size)
{
    if (conversions.empty())
        return 0;
    if (subgroup_size == 0)
    {
        log_error("cooperative matrix conversion requires a known subgroup size");
        return -1;
    }

    tint::core::ir::Function* compute = 0;
    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        if (!module.functions[i]->IsCompute())
            continue;
        if (compute)
        {
            log_error("cooperative matrix conversion requires one compute entry point");
            return -1;
        }
        compute = module.functions[i];
    }
    if (!compute || !compute->WorkgroupSizeAsConst().has_value())
    {
        log_error("cooperative matrix conversion requires a constant workgroup size");
        return -1;
    }

    uint32_t maximum_matrix_elements = 0;
    for (size_t i = 0; i < conversions.size(); i++)
    {
        tint::core::ir::UserCall* conversion = conversions[i];
        const tint::core::type::SubgroupMatrix* source = conversion->Args()[0]->Type()->As<tint::core::type::SubgroupMatrix>();
        const tint::core::type::SubgroupMatrix* destination = conversion->Result()->Type()->As<tint::core::type::SubgroupMatrix>();
        const bool f16_to_f32 = source && destination
                                && source->Type()->Is<tint::core::type::F16>()
                                && destination->Type()->Is<tint::core::type::F32>();
        const bool f32_to_f16 = source && destination
                                && source->Type()->Is<tint::core::type::F32>()
                                && destination->Type()->Is<tint::core::type::F16>();
        if ((!f16_to_f32 && !f32_to_f16)
            || source->Kind() != destination->Kind()
            || source->Rows() != destination->Rows()
            || source->Columns() != destination->Columns()
            || source->Rows() > UINT32_MAX / source->Columns())
        {
            log_error("only f16 and f32 cooperative matrix conversion is supported");
            return -1;
        }
        maximum_matrix_elements = std::max(maximum_matrix_elements,
                                           source->Rows() * source->Columns());
    }

    const std::array<uint32_t, 3> workgroup_size = compute->WorkgroupSizeAsConst().value();
    const uint64_t invocation_count = (uint64_t)workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    const uint64_t subgroup_count = (invocation_count + subgroup_size - 1) / subgroup_size;
    const uint64_t scratch_count = subgroup_count * maximum_matrix_elements;
    if (subgroup_count == 0 || scratch_count == 0 || scratch_count > UINT32_MAX)
        return -1;

    tint::core::ir::Builder builder(module);
    tint::core::ir::Value* subgroup_invocation_id = cooperative_matrix_compute_builtin(
        module, builder, compute, tint::core::BuiltinValue::kSubgroupInvocationId,
        "nikki_subgroup_invocation_id");
    tint::core::ir::Value* subgroup_id = cooperative_matrix_compute_builtin(
        module, builder, compute, tint::core::BuiltinValue::kSubgroupId,
        "nikki_subgroup_id");

    const tint::core::type::Array* source_scratch_type = module.Types().array(
        module.Types().f16(), (uint32_t)scratch_count);
    tint::core::ir::Var* source_scratch = builder.Var(
        "nikki_cooperative_matrix_convert_f16",
        module.Types().ptr<tint::core::AddressSpace::kWorkgroup>(source_scratch_type));
    module.root_block->Append(source_scratch);

    const tint::core::type::Array* destination_scratch_type = module.Types().array(
        module.Types().f32(), (uint32_t)scratch_count);
    tint::core::ir::Var* destination_scratch = builder.Var(
        "nikki_cooperative_matrix_convert_f32",
        module.Types().ptr<tint::core::AddressSpace::kWorkgroup>(destination_scratch_type));
    module.root_block->Append(destination_scratch);

    for (size_t i = 0; i < conversions.size(); i++)
    {
        tint::core::ir::UserCall* conversion = conversions[i];
        const tint::core::type::SubgroupMatrix* matrix = conversion->Result()->Type()->As<tint::core::type::SubgroupMatrix>();
        const tint::core::type::SubgroupMatrix* source_matrix_type = conversion->Args()[0]->Type()->As<tint::core::type::SubgroupMatrix>();
        const bool f16_to_f32 = source_matrix_type->Type()->Is<tint::core::type::F16>();
        tint::core::ir::Var* input_scratch = f16_to_f32 ? source_scratch : destination_scratch;
        tint::core::ir::Var* output_scratch = f16_to_f32 ? destination_scratch : source_scratch;
        const tint::core::type::Type* input_type = f16_to_f32
                                                       ? (const tint::core::type::Type*)module.Types().f16()
                                                       : (const tint::core::type::Type*)module.Types().f32();
        const tint::core::type::Type* output_type = f16_to_f32
                                                        ? (const tint::core::type::Type*)module.Types().f32()
                                                        : (const tint::core::type::Type*)module.Types().f16();
        const uint32_t matrix_elements = matrix->Rows() * matrix->Columns();
        tint::core::ir::InstructionResult* result = conversion->DetachResult();
        tint::core::ir::Value* source_matrix = conversion->Args()[0];

        builder.InsertBefore(conversion, [&]() {
            tint::core::ir::Value* subgroup_base = builder.Multiply(
                                                              subgroup_id, builder.Constant(tint::core::u32(maximum_matrix_elements)))
                                                       ->Result();
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kSubgroupMatrixStore,
                         input_scratch->Result(), subgroup_base, source_matrix,
                         builder.Constant(false), builder.Constant(tint::core::u32(matrix->Columns())));
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kWorkgroupBarrier);

            const std::function<void(tint::core::ir::Value*)> convert_element =
                [&](tint::core::ir::Value* index) {
                    tint::core::ir::Value* scratch_index = builder.Add(subgroup_base, index)->Result();
                    tint::core::ir::Access* source_pointer = builder.Access(
                        module.Types().ptr(tint::core::AddressSpace::kWorkgroup, input_type,
                                           tint::core::Access::kReadWrite),
                        input_scratch->Result(), scratch_index);
                    tint::core::ir::Access* destination_pointer = builder.Access(
                        module.Types().ptr(tint::core::AddressSpace::kWorkgroup, output_type,
                                           tint::core::Access::kReadWrite),
                        output_scratch->Result(), scratch_index);
                    builder.Store(destination_pointer,
                                  builder.Convert(output_type, builder.Load(source_pointer))->Result());
                };
            for (uint32_t first = 0; first < matrix_elements; first += subgroup_size)
            {
                tint::core::ir::Value* index = first == 0
                                                   ? subgroup_invocation_id
                                                   : builder.Add(subgroup_invocation_id,
                                                                 builder.Constant(tint::core::u32(first)))
                                                         ->Result();
                if (matrix_elements - first >= subgroup_size)
                {
                    convert_element(index);
                }
                else
                {
                    tint::core::ir::If* branch = builder.If(builder.LessThan(
                        index, builder.Constant(tint::core::u32(matrix_elements))));
                    builder.Append(branch->True(), [&]() {
                        convert_element(index);
                        builder.ExitIf(branch);
                    });
                }
            }
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kWorkgroupBarrier);
            builder.CallExplicitWithResult<tint::core::ir::CoreBuiltinCall>(
                result, tint::core::BuiltinFn::kSubgroupMatrixLoad,
                tint::Vector<tint::core::ir::TemplateParameter, 1>{matrix},
                output_scratch->Result(), subgroup_base, builder.Constant(false),
                builder.Constant(tint::core::u32(matrix->Columns())));
            builder.Call(module.Types().void_(), tint::core::BuiltinFn::kWorkgroupBarrier);
        });
        conversion->Destroy();
    }

    return 0;
}

int restore_cooperative_matrix_ir(tint::core::ir::Module& module,
                                  const CooperativeMatrixLegalization& info,
                                  uint32_t subgroup_size)
{
    const size_t expected_count = info.load_count + info.store_count + info.muladd_count
                                  + info.convert_count
                                  + info.scalar_add_count + info.scalar_subtract_count
                                  + info.scalar_multiply_count;
    if (expected_count == 0)
        return 0;

    std::unordered_map<const tint::core::type::Type*, const tint::core::type::Type*> type_cache;
    bool failed = false;

    std::vector<tint::core::ir::Constant*> matrix_constants;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        for (tint::core::ir::Value* operand : instruction->Operands())
        {
            tint::core::ir::Constant* constant = operand ? operand->As<tint::core::ir::Constant>() : 0;
            if (!constant)
                continue;
            const tint::core::type::Type* replacement = cooperative_matrix_ir_type(
                module, constant->Type(), type_cache, failed);
            if (replacement != constant->Type()
                && std::find(matrix_constants.begin(), matrix_constants.end(), constant) == matrix_constants.end())
                matrix_constants.push_back(constant);
        }
    }

    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        tint::core::ir::Function* function = module.functions[i];
        function->SetReturnType(cooperative_matrix_ir_type(module, function->ReturnType(), type_cache, failed));
        for (tint::core::ir::FunctionParam* parameter : function->Params())
            parameter->SetType(cooperative_matrix_ir_type(module, parameter->Type(), type_cache, failed));
    }
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        for (tint::core::ir::InstructionResult* result : instruction->Results())
            result->SetType(cooperative_matrix_ir_type(module, result->Type(), type_cache, failed));
    }
    if (failed)
        return -1;

    tint::core::ir::Builder builder(module);
    for (size_t i = 0; i < matrix_constants.size(); i++)
    {
        tint::core::ir::Constant* constant = matrix_constants[i];
        const tint::core::type::SubgroupMatrix* matrix_type = cooperative_matrix_ir_type(
                                                                  module, constant->Type(), type_cache, failed)
                                                                  ->As<tint::core::type::SubgroupMatrix>();
        if (!matrix_type)
        {
            log_error("composite constants containing cooperative matrices are not supported");
            return -1;
        }

        const tint::core::constant::Value* component = constant->Value()->Index(0);
        constant->ReplaceAllUsesWith([&](tint::core::ir::Usage use) -> tint::core::ir::Value* {
            tint::core::ir::Value* replacement = 0;
            builder.InsertBefore(use.instruction, [&]() {
                tint::core::ir::Value* scalar = component ? builder.Constant(component)
                                                          : builder.Zero(matrix_type->Type());
                replacement = builder.Construct(matrix_type, scalar)->Result();
            });
            return replacement;
        });
    }

    struct MarkerCall
    {
        tint::core::ir::UserCall* call;
        CooperativeMatrixMarker marker;
    };
    std::vector<MarkerCall> marker_calls;
    std::vector<tint::core::ir::Function*> helper_functions;
    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        tint::core::ir::Function* function = module.functions[i];
        const std::string name = module.NameOf(function).Name();
        CooperativeMatrixMarker marker;
        const char load_prefix[] = "nikki_coop_load_";
        const char store_prefix[] = "nikki_coop_store_";
        const char muladd_prefix[] = "nikki_coop_muladd_";
        const char convert_prefix[] = "nikki_coop_convert_";
        const char scalar_add_prefix[] = "nikki_coop_scalar_add_";
        const char scalar_subtract_prefix[] = "nikki_coop_scalar_subtract_";
        const char scalar_multiply_prefix[] = "nikki_coop_scalar_multiply_";
        if (name.compare(0, sizeof(load_prefix) - 1, load_prefix) == 0)
            marker = CooperativeMatrixMarkerLoad;
        else if (name.compare(0, sizeof(store_prefix) - 1, store_prefix) == 0)
            marker = CooperativeMatrixMarkerStore;
        else if (name.compare(0, sizeof(muladd_prefix) - 1, muladd_prefix) == 0)
            marker = CooperativeMatrixMarkerMulAdd;
        else if (name.compare(0, sizeof(convert_prefix) - 1, convert_prefix) == 0)
            marker = CooperativeMatrixMarkerConvert;
        else if (name.compare(0, sizeof(scalar_add_prefix) - 1, scalar_add_prefix) == 0)
            marker = CooperativeMatrixMarkerScalarAdd;
        else if (name.compare(0, sizeof(scalar_subtract_prefix) - 1, scalar_subtract_prefix) == 0)
            marker = CooperativeMatrixMarkerScalarSubtract;
        else if (name.compare(0, sizeof(scalar_multiply_prefix) - 1, scalar_multiply_prefix) == 0)
            marker = CooperativeMatrixMarkerScalarMultiply;
        else
            continue;
        helper_functions.push_back(function);

        for (tint::core::ir::Instruction* instruction : module.Instructions())
        {
            tint::core::ir::UserCall* call = instruction->As<tint::core::ir::UserCall>();
            if (call && call->Target() == function)
                marker_calls.push_back(MarkerCall{call, marker});
        }
    }

    if (marker_calls.size() != expected_count)
    {
        log_error("restored %llu of %llu cooperative matrix markers",
                  (unsigned long long)marker_calls.size(), (unsigned long long)expected_count);
        return -1;
    }

    size_t restored_load_count = 0;
    size_t restored_store_count = 0;
    size_t restored_muladd_count = 0;
    size_t restored_convert_count = 0;
    size_t restored_scalar_add_count = 0;
    size_t restored_scalar_subtract_count = 0;
    size_t restored_scalar_multiply_count = 0;
    std::vector<CooperativeMatrixMemoryUse> memory_uses;
    std::vector<tint::core::ir::UserCall*> conversions;
    for (size_t i = 0; i < marker_calls.size(); i++)
    {
        tint::core::ir::UserCall* call = marker_calls[i].call;
        const CooperativeMatrixMarker marker = marker_calls[i].marker;
        if (marker == CooperativeMatrixMarkerLoad)
        {
            if (call->Args().size() != 3 || !call->Result()->Type()->Is<tint::core::type::SubgroupMatrix>())
                return -1;
            tint::core::ir::InstructionResult* loaded_component = call->Args()[0]->As<tint::core::ir::InstructionResult>();
            tint::core::ir::Load* load = loaded_component ? loaded_component->Instruction()->As<tint::core::ir::Load>() : 0;
            if (!load)
                return -1;
            tint::core::ir::Value* base = 0;
            tint::core::ir::Value* offset = 0;
            bool col_major;
            if (cooperative_matrix_memory_view(module, builder, call, load->From(), base, offset) != 0
                || !cooperative_matrix_layout(call->Args()[1], col_major))
                return -1;

            tint::core::ir::InstructionResult* result = call->DetachResult();
            tint::core::ir::CoreBuiltinCall* builtin_call = 0;
            builder.InsertBefore(call, [&]() {
                builtin_call = builder.CallExplicitWithResult<tint::core::ir::CoreBuiltinCall>(
                    result, tint::core::BuiltinFn::kSubgroupMatrixLoad,
                    tint::Vector<tint::core::ir::TemplateParameter, 1>{result->Type()},
                    base, offset, builder.Constant(col_major), call->Args()[2]);
            });
            const tint::core::type::SubgroupMatrix* matrix_type = result->Type()->As<tint::core::type::SubgroupMatrix>();
            if (matrix_type && matrix_type->Type()->Is<tint::core::type::F16>()
                && is_vec2u(memory_store_type(load->From()->Type())))
            {
                uint32_t stride = 0;
                const bool zero_stride = cooperative_matrix_constant_u32(call->Args()[2], stride)
                                         && stride == 0;
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryF16Vec2U, zero_stride});
            }
            else if (cooperative_matrix_is_8bit_placeholder(matrix_type, info)
                     && memory_store_type(load->From()->Type())->IsAnyOf<tint::core::type::I32, tint::core::type::U32>())
            {
                uint32_t stride = 0;
                const bool zero_stride = cooperative_matrix_constant_u32(call->Args()[2], stride)
                                         && stride == 0;
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryNone, zero_stride});
            }
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::I32>()
                     && is_vec4i(memory_store_type(load->From()->Type())))
            {
                uint32_t stride = 0;
                const bool zero_stride = cooperative_matrix_constant_u32(call->Args()[2], stride)
                                         && stride == 0;
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryI32Vec4I, zero_stride});
            }
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::U32>()
                     && is_vec4u(memory_store_type(load->From()->Type())))
            {
                uint32_t stride = 0;
                const bool zero_stride = cooperative_matrix_constant_u32(call->Args()[2], stride)
                                         && stride == 0;
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryU32Vec4U, zero_stride});
            }
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::F32>()
                     && is_vec4f(memory_store_type(load->From()->Type())))
            {
                uint32_t stride = 0;
                const bool zero_stride = cooperative_matrix_constant_u32(call->Args()[2], stride)
                                         && stride == 0;
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryF32Vec4F, zero_stride});
            }
            call->Destroy();
            if (load->Result()->NumUsages() == 0)
                load->Destroy();
            restored_load_count++;
        }
        else if (marker == CooperativeMatrixMarkerStore)
        {
            if (call->Args().size() != 3 || !call->Args()[0]->Type()->Is<tint::core::type::SubgroupMatrix>())
                return -1;
            const tint::Vector<tint::core::ir::Usage, 4> usages = call->Result()->UsagesSorted();
            tint::core::ir::Store* store = usages.Length() == 1
                                               ? usages[0].instruction->As<tint::core::ir::Store>()
                                               : 0;
            if (!store || store->From() != call->Result())
                return -1;
            tint::core::ir::Value* base = 0;
            tint::core::ir::Value* offset = 0;
            bool col_major;
            if (cooperative_matrix_memory_view(module, builder, call, store->To(), base, offset) != 0
                || !cooperative_matrix_layout(call->Args()[1], col_major))
                return -1;

            tint::core::ir::CoreBuiltinCall* builtin_call = 0;
            builder.InsertBefore(call, [&]() {
                builtin_call = builder.Call(module.Types().void_(), tint::core::BuiltinFn::kSubgroupMatrixStore,
                                            base, offset, call->Args()[0], builder.Constant(col_major), call->Args()[2]);
            });
            const tint::core::type::SubgroupMatrix* matrix_type = call->Args()[0]->Type()->As<tint::core::type::SubgroupMatrix>();
            if (matrix_type && matrix_type->Type()->Is<tint::core::type::F16>()
                && is_vec2u(memory_store_type(store->To()->Type())))
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryF16Vec2U, false});
            else if (cooperative_matrix_is_8bit_placeholder(matrix_type, info)
                     && memory_store_type(store->To()->Type())->IsAnyOf<tint::core::type::I32, tint::core::type::U32>())
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryNone, false});
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::I32>()
                     && is_vec4i(memory_store_type(store->To()->Type())))
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryI32Vec4I, false});
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::U32>()
                     && is_vec4u(memory_store_type(store->To()->Type())))
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryU32Vec4U, false});
            else if (matrix_type && matrix_type->Type()->Is<tint::core::type::F32>()
                     && is_vec4f(memory_store_type(store->To()->Type())))
                memory_uses.push_back(CooperativeMatrixMemoryUse{
                    builtin_call, cooperative_matrix_memory_root(base), 4,
                    PackedCooperativeMatrixMemoryF32Vec4F, false});
            store->Destroy();
            call->Destroy();
            restored_store_count++;
        }
        else if (marker == CooperativeMatrixMarkerMulAdd)
        {
            if (call->Args().size() != 3 || !call->Result()->Type()->Is<tint::core::type::SubgroupMatrix>())
                return -1;
            tint::core::ir::InstructionResult* result = call->DetachResult();
            builder.InsertBefore(call, [&]() {
                builder.CallWithResult(result, tint::core::BuiltinFn::kSubgroupMatrixMultiplyAccumulate,
                                       call->Args()[0], call->Args()[1], call->Args()[2]);
            });
            call->Destroy();
            restored_muladd_count++;
        }
        else if (marker == CooperativeMatrixMarkerConvert)
        {
            if (call->Args().size() != 1
                || !call->Args()[0]->Type()->Is<tint::core::type::SubgroupMatrix>()
                || !call->Result()->Type()->Is<tint::core::type::SubgroupMatrix>())
                return -1;
            conversions.push_back(call);
            restored_convert_count++;
        }
        else
        {
            if (call->Args().size() != 2 || !call->Result()->Type()->Is<tint::core::type::SubgroupMatrix>())
                return -1;

            tint::core::BuiltinFn builtin;
            if (marker == CooperativeMatrixMarkerScalarAdd)
                builtin = tint::core::BuiltinFn::kSubgroupMatrixScalarAdd;
            else if (marker == CooperativeMatrixMarkerScalarSubtract)
                builtin = tint::core::BuiltinFn::kSubgroupMatrixScalarSubtract;
            else
                builtin = tint::core::BuiltinFn::kSubgroupMatrixScalarMultiply;

            tint::core::ir::InstructionResult* result = call->DetachResult();
            builder.InsertBefore(call, [&]() {
                builder.CallWithResult(result, builtin, call->Args()[0], call->Args()[1]);
            });
            call->Destroy();

            if (marker == CooperativeMatrixMarkerScalarAdd)
                restored_scalar_add_count++;
            else if (marker == CooperativeMatrixMarkerScalarSubtract)
                restored_scalar_subtract_count++;
            else
                restored_scalar_multiply_count++;
        }
    }

    if (restored_load_count != info.load_count || restored_store_count != info.store_count
        || restored_muladd_count != info.muladd_count
        || restored_convert_count != info.convert_count
        || restored_scalar_add_count != info.scalar_add_count
        || restored_scalar_subtract_count != info.scalar_subtract_count
        || restored_scalar_multiply_count != info.scalar_multiply_count)
        return -1;

    if (emulate_f16_f32_cooperative_matrix_conversions(
            module, conversions, subgroup_size)
        != 0)
        return -1;

    for (size_t i = 0; i < helper_functions.size(); i++)
        module.Destroy(helper_functions[i]);

    if (scalarize_packed_cooperative_matrix_backing(
            module, memory_uses, PackedCooperativeMatrixMemoryF16Vec2U)
        != 0)
        return -1;
    if (scalarize_packed_cooperative_matrix_backing(
            module, memory_uses, PackedCooperativeMatrixMemoryF32Vec4F)
        != 0)
        return -1;
    if (scalarize_packed_cooperative_matrix_backing(
            module, memory_uses, PackedCooperativeMatrixMemoryI32Vec4I)
        != 0)
        return -1;
    if (scalarize_packed_cooperative_matrix_backing(
            module, memory_uses, PackedCooperativeMatrixMemoryU32Vec4U)
        != 0)
        return -1;
    scale_packed_cooperative_matrix_memory_uses(module, memory_uses);
    if (emulate_zero_stride_f16_cooperative_matrix_loads(module, memory_uses,
                                                         subgroup_size)
        != 0)
        return -1;

    return 0;
}

} // namespace nikki_detail
