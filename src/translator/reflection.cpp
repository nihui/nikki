// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

static uint64_t binding_key(uint32_t group, uint32_t binding)
{
    return ((uint64_t)group << 32) | binding;
}

static bool storage_variable_writes(tint::core::ir::Value* value,
                                    std::vector<tint::core::ir::Instruction*>& writes)
{
    std::vector<tint::core::ir::Value*> pending;
    std::unordered_set<tint::core::ir::Value*> visited;
    pending.push_back(value);

    while (!pending.empty())
    {
        tint::core::ir::Value* current = pending.back();
        pending.pop_back();
        if (!current || !visited.insert(current).second)
            continue;

        for (const tint::core::ir::Usage& usage : current->UsagesUnsorted())
        {
            tint::core::ir::Instruction* instruction = usage.instruction;
            if (instruction->GetSideEffects().Contains(tint::core::ir::Instruction::Access::kStore))
            {
                if (std::find(writes.begin(), writes.end(), instruction) == writes.end())
                    writes.push_back(instruction);
                continue;
            }

            for (tint::core::ir::InstructionResult* result : instruction->Results())
            {
                if (result->Type()->Is<tint::core::type::Pointer>())
                    pending.push_back(result);
            }
        }
    }

    return !writes.empty();
}

void collect_storage_access(tint::core::ir::Module& module,
                            std::unordered_map<uint64_t, ShaderBufferAccess>& access,
                            std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >& writes)
{
    std::vector<tint::core::ir::Var*> inactive;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::Var* var = instruction->As<tint::core::ir::Var>();
        if (!var)
            continue;

        std::optional<tint::BindingPoint> binding_point = var->BindingPoint();
        const tint::core::type::Pointer* pointer = var->Result()->Type()->As<tint::core::type::Pointer>();
        if (!binding_point.has_value() || !pointer
            || pointer->AddressSpace() != tint::core::AddressSpace::kStorage)
            continue;
        if (!var->Result()->IsUsed())
        {
            inactive.push_back(var);
            continue;
        }

        const uint64_t key = binding_key(binding_point->group, binding_point->binding);
        access[key] = storage_variable_writes(var->Result(), writes[key])
                          ? SHADER_BUFFER_ACCESS_READ_WRITE
                          : SHADER_BUFFER_ACCESS_READ;
    }

    for (size_t i = 0; i < inactive.size(); i++)
        inactive[i]->Destroy();
}

static const tint::core::ir::Instruction* instruction_in_ancestor_block(
    const tint::core::ir::Instruction* instruction, const tint::core::ir::Block* ancestor)
{
    if (!instruction || !ancestor)
        return 0;

    if (instruction->Block() == ancestor)
        return instruction;

    for (const tint::core::ir::Block* block = instruction->Block(); block && block->Parent();)
    {
        const tint::core::ir::ControlInstruction* parent = block->Parent();
        if (parent->Block() == ancestor)
            return parent;
        block = parent->Block();
    }

    return 0;
}

static bool instruction_is_after(const tint::core::ir::Instruction* position,
                                 const tint::core::ir::Instruction* instruction)
{
    if (!position || !instruction || position->Block() != instruction->Block())
        return false;

    for (const tint::core::ir::Instruction* current = position->next; current; current = current->next)
    {
        if (current == instruction)
            return true;
    }

    return false;
}

static bool instruction_returns_before(const tint::core::ir::Instruction* a,
                                       const tint::core::ir::Instruction* b)
{
    for (const tint::core::ir::Block* block = a ? a->Block() : 0; block && block->Parent();)
    {
        const tint::core::ir::ControlInstruction* parent = block->Parent();
        const tint::core::ir::If* if_instruction = parent->As<tint::core::ir::If>();
        if (if_instruction
            && block->Terminator() && block->Terminator()->Is<tint::core::ir::Return>())
        {
            const tint::core::ir::Instruction* b_in_parent = instruction_in_ancestor_block(b, if_instruction->Block());
            if (instruction_is_after(if_instruction, b_in_parent))
                return true;
        }
        block = parent->Block();
    }

    return false;
}

static bool instructions_are_mutually_exclusive(const tint::core::ir::Instruction* a,
                                                const tint::core::ir::Instruction* b)
{
    std::unordered_map<const tint::core::ir::If*, bool> branches;
    for (const tint::core::ir::Block* block = a ? a->Block() : 0; block && block->Parent();)
    {
        const tint::core::ir::ControlInstruction* parent = block->Parent();
        if (const tint::core::ir::If* if_instruction = parent->As<tint::core::ir::If>())
        {
            if (block == if_instruction->True())
                branches[if_instruction] = true;
            else if (block == if_instruction->False())
                branches[if_instruction] = false;
        }
        block = parent->Block();
    }

    for (const tint::core::ir::Block* block = b ? b->Block() : 0; block && block->Parent();)
    {
        const tint::core::ir::ControlInstruction* parent = block->Parent();
        if (const tint::core::ir::If* if_instruction = parent->As<tint::core::ir::If>())
        {
            std::unordered_map<const tint::core::ir::If*, bool>::const_iterator it = branches.find(if_instruction);
            if (it != branches.end())
            {
                const bool true_branch = block == if_instruction->True();
                const bool false_branch = block == if_instruction->False();
                if ((true_branch && !it->second) || (false_branch && it->second))
                    return true;
            }
        }
        block = parent->Block();
    }

    return instruction_returns_before(a, b) || instruction_returns_before(b, a);
}

static bool storage_writes_are_mutually_exclusive(
    const std::vector<tint::core::ir::Instruction*>& a,
    const std::vector<tint::core::ir::Instruction*>& b)
{
    if (a.empty() || b.empty())
        return false;

    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < b.size(); j++)
        {
            if (!instructions_are_mutually_exclusive(a[i], b[j]))
                return false;
        }
    }

    return true;
}

void collect_exclusive_write_binding_pairs(
    const std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >& storage_writes,
    std::unordered_set<uint64_t>& binding_pairs)
{
    for (std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >::const_iterator i = storage_writes.begin(); i != storage_writes.end(); ++i)
    {
        if ((uint32_t)(i->first >> 32) != 0)
            continue;

        std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> >::const_iterator j = i;
        for (++j; j != storage_writes.end(); ++j)
        {
            if ((uint32_t)(j->first >> 32) != 0)
                continue;
            if (storage_writes_are_mutually_exclusive(i->second, j->second))
            {
                uint32_t a = (uint32_t)i->first;
                uint32_t b = (uint32_t)j->first;
                if (a > b)
                    std::swap(a, b);
                binding_pairs.insert(((uint64_t)a << 32) | b);
            }
        }
    }
}

int reflect_shader(tint::Program& program, const char* entry_point,
                   const std::unordered_map<uint64_t, ShaderBufferAccess>& storage_access,
                   const TranslationInput& input, bool use_uniform,
                   TranslationResult& translated)
{
    tint::inspector::Inspector inspector(program);
    tint::inspector::EntryPoint reflected_entry_point = inspector.GetEntryPoint(entry_point);
    if (inspector.has_error() || reflected_entry_point.stage != tint::inspector::PipelineStage::kCompute
        || !reflected_entry_point.workgroup_size.has_value())
    {
        log_error("Tint compute entry point reflection failed: %s", inspector.error().c_str());
        return -1;
    }

    translated.immediate_size = reflected_entry_point.immediate_data_size;
    if (translated.immediate_size > input.push_constant_size)
    {
        log_error("shader immediate size=%u exceeds pipeline layout push constant size=%u",
                  translated.immediate_size, input.push_constant_size);
        return -1;
    }
    if (!use_uniform)
        translated.push_constant_data_size = translated.immediate_size;
    translated.workgroup_size_x = reflected_entry_point.workgroup_size->x;
    translated.workgroup_size_y = reflected_entry_point.workgroup_size->y;
    translated.workgroup_size_z = reflected_entry_point.workgroup_size->z;

    std::vector<tint::inspector::ResourceBinding> resources = inspector.GetResourceBindings(entry_point);
    if (inspector.has_error())
    {
        log_error("Tint resource reflection failed: %s", inspector.error().c_str());
        return -1;
    }

    for (size_t i = 0; i < resources.size(); i++)
    {
        const tint::inspector::ResourceBinding& resource = resources[i];
        ResourceBindingABI binding;
        binding.group = resource.bind_group;
        binding.binding = resource.binding;
        binding.min_binding_size = resource.size;
        binding.internal_uniform = resource.bind_group == 1 && resource.binding == 0 && use_uniform;
        std::unordered_map<uint64_t, ShaderBufferAccess>::const_iterator access_it = storage_access.find(binding_key(resource.bind_group, resource.binding));

        if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kReadOnlyStorageBuffer)
        {
            if (access_it == storage_access.end())
                continue;
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.access = SHADER_BUFFER_ACCESS_READ;
            binding.layout_access = SHADER_BUFFER_ACCESS_READ;
        }
        else if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kStorageBuffer)
        {
            if (access_it == storage_access.end())
                continue;
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.access = access_it->second;
            binding.layout_access = SHADER_BUFFER_ACCESS_READ_WRITE;
        }
        else if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kUniformBuffer)
        {
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.access = SHADER_BUFFER_ACCESS_READ;
            binding.layout_access = SHADER_BUFFER_ACCESS_READ;
            if (resource.size == 0
                || resource.size > input.shader_profile.max_uniform_buffer_binding_size
                || resource.size > UINT32_MAX)
            {
                log_error("invalid uniform buffer size=%llu group=%u binding=%u",
                          (unsigned long long)resource.size, binding.group, binding.binding);
                return -1;
            }
            if (binding.internal_uniform && translated.push_constant_uniform)
            {
                log_error("multiple internal push uniform bindings");
                return -1;
            }
            if (binding.internal_uniform)
            {
                translated.push_constant_uniform = true;
                translated.push_constant_data_size = (uint32_t)resource.size;
                translated.push_constant_uniform_size = (uint32_t)resource.size;
            }
        }
        else
        {
            log_error("unsupported shader resource group=%u binding=%u type=%d",
                      resource.bind_group, resource.binding, (int)resource.resource_type);
            return -1;
        }

        if (!binding.internal_uniform && binding.group != 0)
        {
            log_error("only Vulkan descriptor set 0 is supported, shader uses group=%u binding=%u", binding.group, binding.binding);
            return -1;
        }

        translated.bindings.push_back(binding);
    }

    return 0;
}

} // namespace nikki_detail
