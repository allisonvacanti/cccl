//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief An example of Fibonacci sequence illustrating how we can use
 *        dynamically created logical data and the run_once utility
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int fibo_ref(int n) {
    if (n < 2) {
        return n;
    } else {
        return fibo_ref(n - 1) + fibo_ref(n - 2);
    }
}

__global__ void add(slice<int> out, const slice<int> in1, const slice<int> in2) {
    out(0) = in1(0) + in2(0);
}

__global__ void set(slice<int> out, int val) {
    out(0) = val;
}

logical_data<slice<int>> compute_fibo(context& ctx, int n) {
    // The result for a given value n is memoized in a logical_data that will be reused every time we compute the same
    // value
    return run_once(n)->*[&](int n) {
        auto result = ctx.logical_data(shape_of<slice<int>>(1)).set_symbol(std::to_string(n));
        if (n < 2) {
            ctx.task(result.write()).set_symbol("fibo" + std::to_string(n))->*[=](cudaStream_t s, auto sresult) {
                set<<<1, 1, 0, s>>>(sresult, n);
            };
        } else {
            auto fib2 = compute_fibo(ctx, n - 2);
            auto fib1 = compute_fibo(ctx, n - 1);
            ctx.task(fib1.read(), fib2.read(), result.write()).set_symbol("fibo" + std::to_string(n))
                            ->*
                    [=](cudaStream_t s, auto s1, auto s2, auto sresult) { add<<<1, 1, 0, s>>>(sresult, s1, s2); };
        }
        return result;
    };
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4;

    context ctx;
    auto result = compute_fibo(ctx, n);
    ctx.host_launch(result.read())->*[&](auto res) { EXPECT(res(0) == fibo_ref(n)); };
    ctx.finalize();
}
