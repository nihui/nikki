// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace nikki_detail {
namespace {

struct IntegerRangeInstruction
{
    size_t offset;
    uint32_t count;
    spv::Op opcode;
    uint32_t type;
    uint32_t result;
    uint32_t function;
    uint32_t block;
};

struct IntegerRangeBlock
{
    uint32_t function;
    std::vector<size_t> instructions;
    std::set<uint32_t> predecessors;
    std::set<uint32_t> successors;
    size_t dominator_index;
    std::vector<uint64_t> dominators;
};

// A deliberately narrow analysis of canonical, ascending integer loops. It
// does not change control flow or floating-point evaluation. In particular,
// it never relies on a compiler assuming that signed overflow is undefined.
class BoundedIntegerLoops
{
public:
    explicit BoundedIntegerLoops(std::vector<uint32_t>& words_)
        : words(words_)
    {
    }

    void run();

private:
    const uint32_t* operands(const IntegerRangeInstruction& instruction) const
    {
        return words.data() + instruction.offset + 1;
    }

    const IntegerRangeInstruction* definition(uint32_t id) const
    {
        const auto found = definitions.find(id);
        return found == definitions.end() ? 0 : &instructions[found->second];
    }

    bool integer_type(uint32_t type, bool& is_signed) const
    {
        const IntegerRangeInstruction* instruction = definition(type);
        if (!instruction || instruction->opcode != spv::Op::OpTypeInt || instruction->count != 4 || operands(*instruction)[1] != 32)
            return false;
        is_signed = operands(*instruction)[2] != 0;
        return true;
    }

    bool dominates(const IntegerRangeInstruction& first, const IntegerRangeInstruction& second) const
    {
        if (first.block == 0)
            return first.function == 0;
        if (first.function != second.function)
            return false;
        if (first.block == second.block)
            return first.offset < second.offset;
        const auto before = blocks.find(first.block);
        const auto after = blocks.find(second.block);
        if (before == blocks.end() || after == blocks.end() || before->second.dominators.empty() || after->second.dominators.empty())
            return false;
        const size_t index = before->second.dominator_index;
        return (after->second.dominators[index / 64] & (uint64_t(1) << (index % 64))) != 0;
    }

    bool parse();
    void find_dominators();
    bool variable_stores(uint32_t variable, std::vector<size_t>& stores) const;
    bool constant(uint32_t id, const IntegerRangeInstruction& use, uint32_t& value, unsigned depth = 0) const;
    void simplify_loop(uint32_t header);

    std::vector<uint32_t>& words;
    std::vector<IntegerRangeInstruction> instructions;
    std::map<uint32_t, size_t> definitions;
    std::map<uint32_t, IntegerRangeBlock> blocks;
    std::map<uint32_t, uint32_t> entries;
    std::map<size_t, uint32_t> replacements;
    std::map<uint32_t, uint32_t> zero_constants;
    std::vector<uint32_t> added_constants;
};

bool BoundedIntegerLoops::parse()
{
    if (words.size() < 5)
        return false;
    uint32_t function = 0;
    uint32_t block = 0;
    for (size_t offset = 5; offset < words.size();)
    {
        const uint32_t count = words[offset] >> 16;
        if (count == 0 || count > words.size() - offset)
            return false;
        IntegerRangeInstruction instruction = {offset, count, static_cast<spv::Op>(words[offset] & 0xffffu), 0, 0, function, block};
        bool has_result = false;
        bool has_type = false;
        spv::HasResultAndType(instruction.opcode, &has_result, &has_type);
        if (has_result)
        {
            if (count <= (has_type ? 2u : 1u))
                return false;
            instruction.result = words[offset + (has_type ? 2 : 1)];
            definitions[instruction.result] = instructions.size();
        }
        if (has_type)
            instruction.type = words[offset + 1];
        if (instruction.opcode == spv::Op::OpFunction)
            instruction.function = function = instruction.result;
        if (instruction.opcode == spv::Op::OpLabel)
        {
            instruction.block = block = instruction.result;
            blocks[block].function = function;
            if (!entries.count(function))
                entries[function] = block;
        }
        if (block && instruction.opcode != spv::Op::OpFunctionEnd)
            blocks[block].instructions.push_back(instructions.size());
        instructions.push_back(instruction);
        if (instruction.opcode == spv::Op::OpFunctionEnd)
            block = function = 0;
        offset += count;
    }
    for (auto& item : blocks)
    {
        IntegerRangeBlock& block_info = item.second;
        if (block_info.instructions.empty())
            return false;
        const IntegerRangeInstruction& end = instructions[block_info.instructions.back()];
        const uint32_t* ops = operands(end);
        if (end.opcode == spv::Op::OpBranch)
            block_info.successors.insert(ops[0]);
        else if (end.opcode == spv::Op::OpBranchConditional)
        {
            block_info.successors.insert(ops[1]);
            block_info.successors.insert(ops[2]);
        }
        else if (end.opcode == spv::Op::OpSwitch)
        {
            const IntegerRangeInstruction* selector = definition(ops[0]);
            const IntegerRangeInstruction* type = selector ? definition(selector->type) : 0;
            if (!type || type->opcode != spv::Op::OpTypeInt || (operands(*type)[1] != 32 && operands(*type)[1] != 64))
                return false;
            const size_t literal_words = operands(*type)[1] / 32;
            block_info.successors.insert(ops[1]);
            for (size_t i = 2; i + literal_words < end.count - 1; i += literal_words + 1)
                block_info.successors.insert(ops[i + literal_words]);
        }
        else if (end.opcode != spv::Op::OpReturn && end.opcode != spv::Op::OpReturnValue && end.opcode != spv::Op::OpKill && end.opcode != spv::Op::OpUnreachable && end.opcode != spv::Op::OpTerminateInvocation)
            return false;
        for (uint32_t successor : block_info.successors)
        {
            const auto target = blocks.find(successor);
            if (target == blocks.end() || target->second.function != block_info.function)
                return false;
            target->second.predecessors.insert(item.first);
        }
    }
    return true;
}

void BoundedIntegerLoops::find_dominators()
{
    for (const auto& entry : entries)
    {
        std::set<uint32_t> reachable;
        std::vector<uint32_t> pending(1, entry.second);
        while (!pending.empty())
        {
            const uint32_t id = pending.back();
            pending.pop_back();
            if (!reachable.insert(id).second)
                continue;
            pending.insert(pending.end(), blocks[id].successors.begin(), blocks[id].successors.end());
        }
        const size_t word_count = (reachable.size() + 63) / 64;
        size_t index = 0;
        for (uint32_t id : reachable)
        {
            IntegerRangeBlock& block = blocks[id];
            block.dominator_index = index++;
            block.dominators.assign(word_count, id == entry.second ? 0 : ~uint64_t(0));
            block.dominators[block.dominator_index / 64] |= uint64_t(1) << (block.dominator_index % 64);
        }
        // Dense bit sets avoid allocating a tree node for every possible
        // dominator at every fixed-point iteration.
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (uint32_t id : reachable)
            {
                if (id == entry.second)
                    continue;
                std::vector<uint64_t> next(word_count, ~uint64_t(0));
                for (uint32_t predecessor : blocks[id].predecessors)
                {
                    if (!reachable.count(predecessor))
                        continue;
                    const std::vector<uint64_t>& prior = blocks[predecessor].dominators;
                    for (size_t i = 0; i < word_count; i++)
                        next[i] &= prior[i];
                }
                const size_t own_index = blocks[id].dominator_index;
                next[own_index / 64] |= uint64_t(1) << (own_index % 64);
                if (next != blocks[id].dominators)
                {
                    blocks[id].dominators.swap(next);
                    changed = true;
                }
            }
        }
    }
}

bool BoundedIntegerLoops::variable_stores(uint32_t variable, std::vector<size_t>& stores) const
{
    const IntegerRangeInstruction* declaration = definition(variable);
    const IntegerRangeInstruction* pointer_type = declaration ? definition(declaration->type) : 0;
    if (!declaration || declaration->opcode != spv::Op::OpVariable || declaration->count != 4 || operands(*declaration)[2] != (uint32_t)spv::StorageClass::Function
        || !pointer_type || pointer_type->opcode != spv::Op::OpTypePointer)
        return false;
    bool is_signed = false;
    if (!integer_type(operands(*pointer_type)[2], is_signed))
        return false;
    stores.clear();
    for (size_t i = 0; i < instructions.size(); i++)
    {
        const IntegerRangeInstruction& instruction = instructions[i];
        const uint32_t* ops = operands(instruction);
        if (instruction.opcode == spv::Op::OpDecorate && ops[0] == variable && ops[1] == (uint32_t)spv::Decoration::Volatile)
            return false;
        if (instruction.function != declaration->function)
            continue;
        const size_t start = instruction.result ? (instruction.type ? 2 : 1) : 0;
        for (size_t j = start; j + 1 < instruction.count; j++)
        {
            if (ops[j] != variable)
                continue;
            if (instruction.opcode == spv::Op::OpLoad && j == 2 && instruction.count == 4)
                continue;
            if (instruction.opcode == spv::Op::OpStore && j == 0 && instruction.count == 3)
                stores.push_back(i);
            else
                // Reject escaped/aliased pointers, atomics and memory-access
                // qualifiers. Treating a coincident literal as an ID only
                // makes this proof more conservative.
                return false;
        }
    }
    return true;
}

bool BoundedIntegerLoops::constant(uint32_t id, const IntegerRangeInstruction& use, uint32_t& value, unsigned depth) const
{
    if (depth > 32)
        return false;
    const IntegerRangeInstruction* instruction = definition(id);
    bool is_signed = false;
    if (!instruction || !integer_type(instruction->type, is_signed) || !dominates(*instruction, use))
        return false;
    const uint32_t* ops = operands(*instruction);
    if (instruction->opcode == spv::Op::OpConstant && instruction->count == 4)
    {
        value = ops[2];
        return true;
    }
    if (instruction->opcode == spv::Op::OpConstantNull)
    {
        value = 0;
        return true;
    }
    if (instruction->opcode == spv::Op::OpCopyObject || instruction->opcode == spv::Op::OpBitcast)
        return constant(ops[2], *instruction, value, depth + 1);
    if (instruction->opcode == spv::Op::OpLoad && instruction->count == 4)
    {
        std::vector<size_t> stores;
        if (!variable_stores(ops[2], stores) || stores.size() != 1)
            return false;
        const IntegerRangeInstruction& store = instructions[stores[0]];
        return dominates(store, *instruction) && constant(operands(store)[1], store, value, depth + 1);
    }
    if (instruction->opcode != spv::Op::OpIAdd && instruction->opcode != spv::Op::OpISub && instruction->opcode != spv::Op::OpIMul)
        return false;
    uint32_t left = 0;
    uint32_t right = 0;
    if (!constant(ops[2], *instruction, left, depth + 1) || !constant(ops[3], *instruction, right, depth + 1))
        return false;
    value = instruction->opcode == spv::Op::OpIAdd ? left + right : instruction->opcode == spv::Op::OpISub ? left - right
                                                                                                           : left * right;
    return true;
}

void BoundedIntegerLoops::simplify_loop(uint32_t header)
{
    const IntegerRangeBlock& header_block = blocks[header];
    if (header_block.dominators.empty())
        return;
    const IntegerRangeInstruction* loop_merge = 0;
    for (size_t index : header_block.instructions)
        if (instructions[index].opcode == spv::Op::OpLoopMerge)
            loop_merge = &instructions[index];
    if (!loop_merge)
        return;
    const uint32_t merge = operands(*loop_merge)[0];
    const uint32_t continuing = operands(*loop_merge)[1];
    if (continuing == header || !blocks.count(continuing) || header_block.predecessors.size() != 2 || !header_block.predecessors.count(continuing))
        return;
    uint32_t preheader = 0;
    for (uint32_t predecessor : header_block.predecessors)
        if (predecessor != continuing)
            preheader = predecessor;
    const IntegerRangeInstruction& before = instructions[blocks[preheader].instructions.back()];
    const IntegerRangeInstruction& back = instructions[blocks[continuing].instructions.back()];
    if (before.opcode != spv::Op::OpBranch || operands(before)[0] != header || back.opcode != spv::Op::OpBranch || operands(back)[0] != header)
        return;

    uint32_t condition_block = header;
    const IntegerRangeInstruction* branch = &instructions[header_block.instructions.back()];
    if (branch->opcode == spv::Op::OpBranch)
    {
        condition_block = operands(*branch)[0];
        if (!blocks.count(condition_block) || blocks[condition_block].predecessors != std::set<uint32_t>{header})
            return;
        branch = &instructions[blocks[condition_block].instructions.back()];
    }
    if (branch->opcode != spv::Op::OpBranchConditional || operands(*branch)[2] != merge)
        return;
    const uint32_t body_entry = operands(*branch)[1];
    if (body_entry == merge || body_entry == continuing || body_entry == header || body_entry == condition_block)
        return;
    const IntegerRangeInstruction* comparison = definition(operands(*branch)[0]);
    if (!comparison || comparison->block != condition_block || (comparison->opcode != spv::Op::OpSLessThan && comparison->opcode != spv::Op::OpULessThan))
        return;
    const uint32_t* compare_ops = operands(*comparison);
    const IntegerRangeInstruction* condition_load = definition(compare_ops[2]);
    if (!condition_load || condition_load->opcode != spv::Op::OpLoad || condition_load->count != 4 || condition_load->block != condition_block)
        return;
    const uint32_t variable = operands(*condition_load)[2];
    const uint32_t type = condition_load->type;
    bool is_signed = false;
    if (!integer_type(type, is_signed) || (comparison->opcode == spv::Op::OpSLessThan) != is_signed)
        return;
    std::vector<size_t> stores;
    if (!variable_stores(variable, stores) || stores.size() != 2)
        return;
    const IntegerRangeInstruction* initialize = 0;
    const IntegerRangeInstruction* advance = 0;
    for (size_t store : stores)
    {
        if (instructions[store].block == preheader)
            initialize = &instructions[store];
        if (instructions[store].block == continuing)
            advance = &instructions[store];
    }
    if (!initialize || !advance || !dominates(*initialize, *condition_load))
        return;
    uint32_t initial = 0;
    uint32_t bound = 0;
    if (!constant(operands(*initialize)[1], *initialize, initial) || !constant(compare_ops[3], *comparison, bound)
        || initial >= bound || (is_signed && bound > 0x7fffffffu))
        return;

    // With a strict upper bound representable in the counter's type, the
    // final increment reaches the bound without wrapping. Negative initial
    // signed values fail the unsigned initial < bound test above.
    const IntegerRangeInstruction* increment = definition(operands(*advance)[1]);
    if (!increment || increment->opcode != spv::Op::OpIAdd || increment->type != type || increment->block != continuing || !dominates(*increment, *advance))
        return;
    const uint32_t* increment_ops = operands(*increment);
    bool unit_step = false;
    for (size_t side = 2; side <= 3; side++)
    {
        const IntegerRangeInstruction* prior = definition(increment_ops[side]);
        uint32_t step = 0;
        if (prior && prior->opcode == spv::Op::OpLoad && prior->count == 4 && prior->block == continuing && operands(*prior)[2] == variable
            && dominates(*prior, *increment) && constant(increment_ops[5 - side], *increment, step) && step == 1)
            unit_step = true;
    }
    if (!unit_step)
        return;

    std::set<uint32_t> body;
    std::vector<uint32_t> pending(1, body_entry);
    while (!pending.empty())
    {
        const uint32_t id = pending.back();
        pending.pop_back();
        if (id == header || id == condition_block || id == continuing || id == merge || !body.insert(id).second)
            continue;
        if (!blocks.count(id) || blocks[id].function != header_block.function)
            return;
        pending.insert(pending.end(), blocks[id].successors.begin(), blocks[id].successors.end());
    }
    for (uint32_t id : body)
        for (uint32_t predecessor : blocks[id].predecessors)
            if (!body.count(predecessor) && !(id == body_entry && predecessor == condition_block))
                return;

    for (uint32_t id : body)
    {
        for (size_t index : blocks[id].instructions)
        {
            const IntegerRangeInstruction& instruction = instructions[index];
            const bool division = instruction.opcode == (is_signed ? spv::Op::OpSDiv : spv::Op::OpUDiv);
            const bool remainder = is_signed ? instruction.opcode == spv::Op::OpSMod || instruction.opcode == spv::Op::OpSRem : instruction.opcode == spv::Op::OpUMod;
            if ((!division && !remainder) || instruction.type != type || instruction.count != 5)
                continue;
            const uint32_t* ops = operands(instruction);
            const IntegerRangeInstruction* load = definition(ops[2]);
            if (!load || load->opcode != spv::Op::OpLoad || load->count != 4 || operands(*load)[2] != variable
                || (!body.count(load->block) && load->block != condition_block) || !dominates(*load, instruction))
                continue;
            uint32_t divisor = 0;
            if (!constant(ops[3], instruction, divisor) || divisor < bound || (is_signed && divisor > 0x7fffffffu))
                continue;
            if (division)
            {
                if (!zero_constants.count(type))
                {
                    if (words[3] == std::numeric_limits<uint32_t>::max())
                        continue;
                    const uint32_t zero = words[3]++;
                    zero_constants[type] = zero;
                    const uint32_t declaration[] = {(4u << 16) | (uint32_t)spv::Op::OpConstant, type, zero, 0};
                    added_constants.insert(added_constants.end(), declaration, declaration + 4);
                }
                replacements[instruction.offset] = zero_constants[type];
            }
            else
                replacements[instruction.offset] = ops[2];
        }
    }
}

void BoundedIntegerLoops::run()
{
    if (words.size() < 5)
        return;
    // Most shaders cannot benefit. Avoid building definition/CFG maps for
    // them, which matters when translating many pipelines on mobile CPUs.
    bool has_division = false;
    bool has_loop = false;
    for (size_t offset = 5; offset < words.size();)
    {
        const uint32_t count = words[offset] >> 16;
        if (count == 0 || count > words.size() - offset)
            return;
        const spv::Op opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        // Group-applied decorations require expansion before proving that a
        // function variable is non-volatile. Leave such modules untouched.
        if (opcode == spv::Op::OpGroupDecorate || opcode == spv::Op::OpGroupMemberDecorate)
            return;
        has_loop |= opcode == spv::Op::OpLoopMerge;
        has_division |= opcode == spv::Op::OpSDiv || opcode == spv::Op::OpUDiv || opcode == spv::Op::OpSMod || opcode == spv::Op::OpSRem || opcode == spv::Op::OpUMod;
        offset += count;
    }
    if (!has_loop || !has_division || !parse())
        return;
    for (const IntegerRangeInstruction& instruction : instructions)
    {
        if (instruction.opcode == spv::Op::OpConstantNull || (instruction.opcode == spv::Op::OpConstant && instruction.count == 4 && operands(instruction)[2] == 0))
        {
            bool is_signed = false;
            if (integer_type(instruction.type, is_signed))
                zero_constants[instruction.type] = instruction.result;
        }
    }
    find_dominators();
    for (const auto& block : blocks)
        simplify_loop(block.first);
    if (replacements.empty())
        return;

    std::vector<uint32_t> rewritten(words.begin(), words.begin() + 5);
    bool constants_inserted = false;
    for (const IntegerRangeInstruction& instruction : instructions)
    {
        if (!constants_inserted && instruction.opcode == spv::Op::OpFunction)
        {
            rewritten.insert(rewritten.end(), added_constants.begin(), added_constants.end());
            constants_inserted = true;
        }
        const auto replacement = replacements.find(instruction.offset);
        if (replacement == replacements.end())
            rewritten.insert(rewritten.end(), words.begin() + instruction.offset, words.begin() + instruction.offset + instruction.count);
        else
        {
            const uint32_t copy[] = {(4u << 16) | (uint32_t)spv::Op::OpCopyObject, instruction.type, instruction.result, replacement->second};
            rewritten.insert(rewritten.end(), copy, copy + 4);
        }
    }
    words.swap(rewritten);
}

} // namespace

void simplify_bounded_integer_loops(std::vector<uint32_t>& words)
{
    BoundedIntegerLoops(words).run();
}

} // namespace nikki_detail
