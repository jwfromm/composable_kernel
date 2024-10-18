// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multiply_multiply.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
bool profile_batched_gemm_multiply_multiply_impl(int do_verification,
                                                 int init_method,
                                                 bool do_log,
                                                 bool time_kernel,
                                                 int M,
                                                 int N,
                                                 int K,
                                                 int BatchStrideA,
                                                 int BatchStrideB,
                                                 int BatchStrideD0,
                                                 int BatchStrideD1,
                                                 int BatchStrideE,
                                                 int StrideA,
                                                 int StrideB,
                                                 int StrideD0,
                                                 int StrideD1,
                                                 int StrideE,
                                                 int BatchCount)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        using namespace ck::literals;

        if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB, BatchStrideB, BLayout{}));
    Tensor<D0DataType> d0_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, 1, StrideD0, BatchStrideD0, D0Layout{}));
    Tensor<D1DataType> d1_g_m_n(
        f_host_tensor_descriptor(BatchCount, 1, N, StrideD1, BatchStrideD1, D1Layout{}));
    Tensor<EDataType> e_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideE, BatchStrideE, ELayout{}));
    Tensor<EDataType> e_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideE, BatchStrideE, ELayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "d0_g_m_n: " << d0_g_m_n.mDesc << std::endl;
    std::cout << "d1_g_m_n: " << d1_g_m_n.mDesc << std::endl;
    std::cout << "e_g_m_n: " << e_g_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-5, 5});
        d1_g_m_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-5, 5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_g_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }

    using PassThrough      = ck::tensor_operation::element_wise::PassThrough;
    using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = MultiplyMultiply;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_verification)
    {
        Tensor<AccDataType> c_m_n({BatchCount, M, N});

        // Compare to batch gemm without scaling and perform manual scaling.
        using ReferenceBatchedGemmInstance =
            ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                             BDataType,
                                                             EDataType,
                                                             AccDataType,
                                                             AElementOp,
                                                             BElementOp,
                                                             PassThrough>;

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, e_g_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        // Do manual scaling to simulate epilogue.
        for(int b = 0; b < BatchCount; ++b)
        {
            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    c_element_op(
                        e_g_m_n_host_result(b, m, n), c_m_n(b, m, n), d0_g_m_n(b, m, 0), d1_g_m_n(b, 0, n));
                }
            }
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_g_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());
    d0_device_buf.ToDevice(d0_g_m_n.mData.data());
    d1_device_buf.ToDevice(d1_g_m_n.mData.data());
    e_device_buf.ToDevice(e_g_m_n_device_result.mData.data());

    using DeviceOp =
        ck::tensor_operation::device::DeviceBatchedGemmMultiD<ALayout,
                                                              BLayout,
                                                              ck::Tuple<D0Layout, D1Layout>,
                                                              ELayout,
                                                              ADataType,
                                                              BDataType,
                                                              ck::Tuple<D0DataType, D1DataType>,
                                                              EDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        std::unique_ptr<tensor_operation::device::BaseArgument> argument_ptr;
        argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                        static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                        std::array<const void*, 2>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer()},
                                        static_cast<EDataType*>(e_device_buf.GetDeviceBuffer()),
                                        M,
                                        N,
                                        K,
                                        BatchCount,
                                        StrideA,
                                        StrideB,
                                        std::array<ck::index_t, 2>{StrideD0, StrideD1},
                                        StrideE,
                                        BatchStrideA,
                                        BatchStrideB,
                                        std::array<ck::index_t, 2>{BatchStrideD0, BatchStrideD1},
                                        BatchStrideE,
                                        a_element_op,
                                        b_element_op,
                                        c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init E to zero before profiling next kernel
            e_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * BatchCount * M * N * K;

            std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                     sizeof(EDataType) * M * N) *
                                    BatchCount;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                e_device_buf.FromDevice(e_g_m_n_device_result.mData.data());

                pass = pass & ck::utils::check_err(e_g_m_n_device_result, e_g_m_n_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", e_g_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_device: ", e_g_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
