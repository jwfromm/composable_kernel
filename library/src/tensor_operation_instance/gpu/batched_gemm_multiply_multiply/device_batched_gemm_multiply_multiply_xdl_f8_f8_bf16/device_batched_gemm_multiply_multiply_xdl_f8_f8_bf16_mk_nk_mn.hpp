// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8   = f8_t;
using BF16 = bhalf_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using PassThrough      = element_wise::PassThrough;
using MultiplyMultiply = element_wise::MultiplyMultiply;

static constexpr auto GemmDefault    = GemmSpecialization::Default;
static constexpr auto GemmKPadding   = GemmSpecialization::KPadding;
static constexpr auto GemmMNPadding  = GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Default = LoopScheduler::Default;
static constexpr auto Interwave = LoopScheduler::Interwave;

template <GemmSpecialization GemmSpec>
using device_batched_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_instances = std::tuple<
    // clang-format off
    //######                       | ALayout| BLayout|         DsLayout| ELayout|     AData|     BData|     AccData|   CShuffle|           DsData|     EData|           A|           B|              CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|            LoopScheduler|
    //######                       |        |        |                 |        |      Type|      Type|        Type|   DataType|             Type|      Type| Elementwise| Elementwise|      Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|                         |
    //######                       |        |        |                 |        |          |          |            |           |                 |          |   Operation|   Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|                         |
    //######                       |        |        |                 |        |          |          |            |           |                 |          |            |            |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                         |
        DeviceBatchedGemmMultiD_Xdl<     Row,     Col,  Tuple<Row, Col>,     Row,        F8,        F8,         F32,        F32,  Tuple<F32, F32>,      BF16, PassThrough, PassThrough, MultiplyMultiply,       GemmSpec,        1,   256,   256,   128,    64,  16,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16,  LoopScheduler::Default>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
