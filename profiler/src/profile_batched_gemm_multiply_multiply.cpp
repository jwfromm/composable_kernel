// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdint>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_batched_gemm_multiply_multiply_impl.hpp"
#include "profiler_operation_registry.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

enum struct GemmMatrixLayout
{
    MK_NK_MN, // 0
};

enum struct GemmDataType
{
    F8_F8_BF16, // 0
};

#define OP_NAME "batched_gemm_multiply_multiply"
#define OP_DESC "Batched GEMM with multiply-multiply epilogue"

int profile_batched_gemm_multiply_multiply(int argc, char* argv[])
{
    if(argc != 18)
    {
        // clang-format off
        printf("arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n");
        printf("arg2: data type (0: fp8 input bf16 output)\n");
        printf("arg3: matrix layout (0: A[g, m, k] * B[g, n, k] = C[g, m, n];\n");
        printf("arg4: verification (0: no; 1: yes)\n");
        printf("arg5: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg6: print tensor value (0: no; 1: yes)\n");
        printf("arg7: time kernel (0=n0, 1=yes)\n");
        printf("arg8 to 17: M, N, K, StrideA, StrideB, StrideC, BatchStrideA, BatchStrideB, BatchStrideC, BatchCount\n");
        // clang-format on
        exit(1);
    }

    const auto data_type       = static_cast<GemmDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<GemmMatrixLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);

    const int M = std::stoi(argv[8]);
    const int N = std::stoi(argv[9]);
    const int K = std::stoi(argv[10]);

    const int StrideA = std::stoi(argv[11]);
    const int StrideB = std::stoi(argv[12]);
    const int StrideE = std::stoi(argv[13]);

    const int BatchStrideA = std::stoi(argv[14]);
    const int BatchStrideB = std::stoi(argv[15]);
    const int BatchStrideE = std::stoi(argv[16]);

    const int BatchCount = std::stoi(argv[17]);

    using F8 = ck::f8_t;
    using BF16 = ck::bhalf_t;
    using F32 = float;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    auto profile = [&](auto a_type,
                       auto b_type,
                       auto acc_type,
                       auto d0_type,
                       auto d1_type,
                       auto e_type,
                       auto a_layout,
                       auto b_layout,
                       auto e_layout) {
        using ADataType   = decltype(a_type);
        using BDataType   = decltype(b_type);
        using AccDataType = decltype(acc_type);
        using D0DataType  = decltype(d0_type);
        using D1DataType  = decltype(d1_type);
        using EDataType   = decltype(e_type);

        using ALayout  = decltype(a_layout);
        using BLayout  = decltype(b_layout);
        using D0Layout = Row;
        using D1Layout = Col;
        using ELayout  = decltype(e_layout);

        const int DefaultStrideA = ck::is_same_v<ALayout, Row> ? K : M;
        const int DefaultStrideB = ck::is_same_v<BLayout, Row> ? N : K;
        const int DefaultStrideE = ck::is_same_v<ELayout, Row> ? N : M;

        const int StrideA_  = (StrideA < 0) ? DefaultStrideA : StrideA;
        const int StrideB_  = (StrideB < 0) ? DefaultStrideB : StrideB;
        const int StrideD0_ = 0;
        const int StrideD1_ = 0;
        const int StrideE_  = (StrideE < 0) ? DefaultStrideE : StrideE;

        const int DefaultBatchStrideA = (ck::is_same_v<ALayout, Row> ? M : K) * StrideA_;
        const int DefaultBatchStrideB = (ck::is_same_v<BLayout, Row> ? K : N) * StrideB_;
        const int DefaultBatchStrideE = (ck::is_same_v<ELayout, Row> ? M : N) * StrideE_;

        const int BatchStrideA_ = (BatchStrideA < 0) ? DefaultBatchStrideA : BatchStrideA;
        const int BatchStrideB_ = (BatchStrideB < 0) ? DefaultBatchStrideB : BatchStrideB;
        const int BatchStrideD0_ = M;
        const int BatchStrideD1_ = N;
        const int BatchStrideE_ = (BatchStrideE < 0) ? DefaultBatchStrideE : BatchStrideE;

        bool pass =
            ck::profiler::profile_batched_gemm_multiply_multiply_impl<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      D0DataType,
                                                                      D1DataType,
                                                                      EDataType,
                                                                      ALayout,
                                                                      BLayout,
                                                                      D0Layout,
                                                                      D1Layout,
                                                                      ELayout>(do_verification,
                                                                               init_method,
                                                                               do_log,
                                                                               time_kernel,
                                                                               M,
                                                                               N,
                                                                               K,
                                                                               BatchStrideA_,
                                                                               BatchStrideB_,
                                                                               BatchStrideD0_,
                                                                               BatchStrideD1_,
                                                                               BatchStrideE_,
                                                                               StrideA_,
                                                                               StrideB_,
                                                                               StrideD0_,
                                                                               StrideD1_,
                                                                               StrideE_,
                                                                               BatchCount);

        return pass ? 0 : 1;
    };

    if(data_type == GemmDataType::F8_F8_BF16 && layout == GemmMatrixLayout::MK_NK_MN)
    {
        return profile(F8{}, F8{}, F32{}, F32{}, F32{}, BF16{}, Row{}, Col{}, Row{});
    }
    else
    {
        std::cout << "this data_type & layout is not implemented" << std::endl;

        return 1;
    }
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_batched_gemm_multiply_multiply);
