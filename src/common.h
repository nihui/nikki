// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_COMMON_H
#define NIKKI_COMMON_H

#include <cstdarg>
#include <cstdio>

namespace nikki_detail {

inline void log_error(const char* format, ...)
{
    fprintf(stderr, "[nikki] ");

    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);

    fprintf(stderr, "\n");
}

inline void log_info(const char* format, ...)
{
    fprintf(stderr, "[nikki] ");

    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);

    fprintf(stderr, "\n");
}

} // namespace nikki_detail

#endif // NIKKI_COMMON_H
