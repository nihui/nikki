// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_DISPATCH_H
#define NIKKI_BACKEND_DISPATCH_H

#include "runtime.h"

namespace nikki_detail {

struct SameBufferCopyPlan
{
    uint64_t binding_offset;
    uint64_t binding_size;
    uint32_t src_word;
    uint32_t dst_word;
    uint32_t word_count;
};

struct SameBufferCopyParams
{
    uint32_t src_word;
    uint32_t dst_word;
    uint32_t word_count;
    uint32_t base_word;
};

uint64_t same_buffer_copy_max_words(const Device* device);
bool make_same_buffer_copy_plan(Device* device, const Command& command,
                                const CopyRegion& region, SameBufferCopyPlan& plan);
int encode_same_buffer_copy(Device* device, ReplayState& replay,
                            const Command& command, const SameBufferCopyPlan& plan,
                            const std::shared_ptr<Submission>& submission);
int encode_dispatch(Device* device, ReplayState& replay, const Command& command,
                    const std::shared_ptr<Submission>& submission);
uint64_t binding_pair_key(uint32_t a, uint32_t b);

} // namespace nikki_detail

#endif // NIKKI_BACKEND_DISPATCH_H
