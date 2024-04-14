/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_histogram.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cstdint>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

// TODO Existing test cases:
// [HistogramEven]
// - TestOverflow: if (max_level - min_level) * num_bins overflows, cudaErrorInvalidValue is returned
// - TestIntegerBinCalcs: Integer bin calc rounding edge cases
// [HistogramRange]
// - TestLevelsAliasing: Levels are included in the samples.
// [Both]
// - NumChannels / NumActiveChannels [1-4]
// - Types/levels per table below
// - Problem sizes (num_row_pixels, num_rows):
//   (0, 0), (1920, 0),
//   (15, 1), (1920, 1080),
//   ([1, 1000, 1000000], [1, 1000, 1000000])
// - Row strides: (No padding, 13 samples padding)
// - Entropy levels: -1 (all samples 0), 0, 5
// - NumLevels: channel0: max_levels, channelN: num_levels[N - 1 / 2] + 1
// - LevelTypes: Even and Range
//   - See TestRange/TestEven for level calcs
//   - Use iterators for range tests

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::Bulk, device_bulk);

// sample_t  counter_t  level_t    offset_t    max_level max_num_levels
//
// half,           int, half,           int   (256,  257)
// signed char,    int, int,            int   (256,  257)
// usigned short,  int, int,            int   (8192, 8193)
// unsigned short, int, unsigned short, int   (USHORT_MAX, USHORT_MAX + 1)
// unsigned int,   int, unsigned int,   int   (UINT_MAX, 8193)
// float,          int, float,          int   (1.0, 257)
// float,          int, int,            int   (12, 7)
// unsigned char,  int, int,            int64 (256, 257)

using sample_types  = c2h::type_list<>;
using counter_types = c2h::type_list<>;
using level_types   = c2h::type_list<>;
using offset_types  = c2h::type_list<>;

CUB_TEST("name", "[histogram][device]", sample_types) {}
