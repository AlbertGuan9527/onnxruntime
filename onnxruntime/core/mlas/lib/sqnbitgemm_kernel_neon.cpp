/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON.

--*/

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

//
// Quantized B data packing function implementation.
//

namespace
{

size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 4;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    constexpr size_t BlkBitWidth = 4;

    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t Iterations = N * BlockCountK;  // one iteration per block

    const size_t SubBlkLen = (ComputeType == CompInt8)
                                 ? ((BlkLen == 16) ? 16 : 32)
                                 : 16;

    const size_t SubBlkDataSize = SubBlkLen / 2;
    const size_t SubBlkBytePairCount = SubBlkLen / 4;

    //
    // For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
    //
    // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
    //   =>
    // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
    //

    //
    // For SubBlkLen == 32, pack 32 4-bit values (16 bytes) at a time like this:
    //
    // src: | v0  v1  | v2  v3  | ... | v28 v29 | v30 v31 |
    //   =>
    // dst: | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    //

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;

            const size_t data_offset = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;
            const std::byte* QuantBData = QuantBDataBegin + data_offset;
            std::byte* PackedQuantBData = PackedQuantBDataBegin + data_offset;

            for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
                for (size_t byte_pair_idx = 0; byte_pair_idx < SubBlkBytePairCount; ++byte_pair_idx) {
                    const std::byte src0 = QuantBData[byte_pair_idx];
                    const std::byte src1 = QuantBData[byte_pair_idx + SubBlkDataSize / 2];

                    std::byte& dst0 = PackedQuantBData[2 * byte_pair_idx];
                    std::byte& dst1 = PackedQuantBData[2 * byte_pair_idx + 1];

                    dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
                }

                QuantBData += SubBlkDataSize;
                PackedQuantBData += SubBlkDataSize;
            }
        }
    );
}

//
// Workspace size calculation function implementation.
//

size_t
SQ4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch(ComputeType) {
        case CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
SQ4BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    switch (ComputeType) {
        case CompInt8: {
            return Q8BlkAlignment();
        }
        default: {
            return 1;
        }
    }
}

}  // namespace

//
// General helpers.
//

namespace
{

template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}

MLAS_FORCEINLINE void
Transpose4x4(float32x4_t& a0, float32x4_t& a1, float32x4_t& a2, float32x4_t& a3)
{
    // aN: aN_0 aN_1 aN_2 aN_3

    float32x4_t b0 = vzip1q_f32(a0, a1);  // a0_0 a1_0 a0_1 a1_1
    float32x4_t b1 = vzip2q_f32(a0, a1);  // a0_2 a1_2 a0_3 a1_3
    float32x4_t b2 = vzip1q_f32(a2, a3);  // a2_0 a3_0 a2_1 a3_1
    float32x4_t b3 = vzip2q_f32(a2, a3);  // a2_2 a3_2 a2_3 a3_3

    // a0_0 a1_0 a2_0 a3_0
    a0 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(b0), vreinterpretq_f64_f32(b2)));
    // a0_1 a1_1 a2_1 a3_1
    a1 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(b0), vreinterpretq_f64_f32(b2)));
    // a0_2 a1_2 a3_2 a3_2
    a2 = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(b1), vreinterpretq_f64_f32(b3)));
    // a0_3 a1_3 a2_3 a3_3
    a3 = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(b1), vreinterpretq_f64_f32(b3)));
}

MLAS_FORCEINLINE float32x4_t
FoldAccumulators(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3)
{
    Transpose4x4(a0, a1, a2, a3);
    return vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
}

template <size_t Capacity>
MLAS_FORCEINLINE void
LoadFloatData(const float* src, size_t count, float32x4_t (&dst)[Capacity / 4])
{
    static_assert(Capacity % 4 == 0, "Capacity must be divisible by 4.");

    assert(count <= Capacity);

    size_t vi = 0;  // vector index

    // handle 4 values at a time
    while (count > 3) {
        dst[vi] = vld1q_f32(src);

        vi += 1;
        src += 4;
        count -= 4;
    }

    // handle remaining values
    if (count > 0) {
        dst[vi] = vsetq_lane_f32(src[0], dst[vi], 0);

        if (count > 1) {
            dst[vi] = vsetq_lane_f32(src[1], dst[vi], 1);

            if (count > 2) {
                dst[vi] = vsetq_lane_f32(src[2], dst[vi], 2);
            }
        }
    }
}

}  // namespace

//
// CompFp32 kernel implementation.
//

namespace
{

namespace fp32_conversion
{

// Manual conversion to float takes place in two steps:
// 1. Map 4-bit values from [0, 15] to float values from [16.0f, 31.0f].
//    This target float range is convenient because the 4-bit source values can be placed directly into the
//    target float bits.
// 2. Subtract the conversion offset of 16 from the float result.

// The high 16 bits of an IEEE 754 32-bit float used as a template for creating float values.
constexpr uint16_t float_high_half_template = 0b0'10000011'0000000;
//                                           sign|exponent|partial mantissa
//                                              +|131: 2^4|~~~~ <- 4 bits go here

const uint16x8_t float_high_half_template_v = vdupq_n_u16(float_high_half_template);

constexpr float offset = 16.0f;

}  // namespace fp32_conversion

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompFp32(
    size_t BlkLen,
    const float* ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkLen = 16;

    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    assert(BlkLen >= SubBlkLen && BlkLen % SubBlkLen == 0);

    const uint8x8_t LowMask = vdup_n_u8(0x0F);

    float32x4_t acc[NCols]{};

    const std::byte* QuantBData = QuantBDataColPtr;
    const float* QuantBScale = QuantBScaleColPtr;
    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
                                                     // only used if HasZeroPoint is true

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        float scale[NCols];
        UnrolledLoop<NCols>(
            [&](size_t i) { scale[i] = QuantBScale[i * StrideQuantBScale]; }
        );

        [[maybe_unused]] float offset[NCols];  // Includes zero point and float conversion offset.
                                               // only used if HasZeroPoint is true
        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
            });
        }

        for (size_t k_idx_in_blk = 0; k_idx_in_blk < k_blk_len; k_idx_in_blk += SubBlkLen) {
            // load A row vector elements

            // load `SubBlkLen` elements from A, padded with 0's if there aren't enough
            const size_t k_subblk_len = std::min(k_blk_len - k_idx_in_blk, SubBlkLen);
            float32x4_t av[4]{};
            LoadFloatData<SubBlkLen>(ARowPtr + k + k_idx_in_blk, k_subblk_len, av);

            // load B column vectors
            uint8x8_t bv_packed[NCols];
            const size_t b_data_block_offset = k_idx_in_blk * BlkBitWidth / 8;
            UnrolledLoop<NCols>([&](size_t i) {
                bv_packed[i] = vld1_u8(
                    reinterpret_cast<const uint8_t*>(QuantBData) + i * StrideQuantBData + b_data_block_offset
                );
            });

            uint8x8_t bv_u8[NCols][2];
            UnrolledLoop<NCols>([&](size_t i) {
                bv_u8[i][0] = vand_u8(bv_packed[i], LowMask);
                bv_u8[i][1] = vshr_n_u8(bv_packed[i], 4);
            });

            // shift left 3 and widen to 16 bits
            uint16x8_t bv_u16[NCols][2];
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr int shift = 3;
                bv_u16[i][0] = vshll_n_u8(bv_u8[i][0], shift);
                bv_u16[i][1] = vshll_n_u8(bv_u8[i][1], shift);
            });

            // combine 4 bits with float high half template
            UnrolledLoop<NCols>([&](size_t i) {
                bv_u16[i][0] = vorrq_u16(bv_u16[i][0], fp32_conversion::float_high_half_template_v);
                bv_u16[i][1] = vorrq_u16(bv_u16[i][1], fp32_conversion::float_high_half_template_v);
            });

            // `SubBlkLen` floats of B
            float32x4_t bv[NCols][4];

            // shift left 16, widen to 32 bits, and reinterpret as float
            UnrolledLoop<NCols>([&](size_t i) {
                constexpr int shift = 16;
                bv[i][0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][0]), shift));
                bv[i][1] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][0], shift));

                bv[i][2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][1]), shift));
                bv[i][3] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][1], shift));
            });

            // subtract float conversion offset and zero point
            if constexpr (HasZeroPoint) {
                UnrolledLoop<NCols>([&](size_t i) {
                    const float32x4_t offset_v = vdupq_n_f32(offset[i]);
                    UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
                });
            } else {
                const float32x4_t offset_v = vdupq_n_f32(fp32_conversion::offset + 8.0f);
                UnrolledLoop<NCols>([&](size_t i) {
                    UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
                });
            }

            // multiply by scale
            UnrolledLoop<NCols>([&](size_t i) {
                const float32x4_t scale_v = vdupq_n_f32(scale[i]);
                UnrolledLoop<4>([&](size_t j) { bv[i][j] = vmulq_f32(bv[i][j], scale_v); });
            });

            // c[m,n] += a[m,k] * b[k,n]
            UnrolledLoop<4>([&](size_t j) {
                UnrolledLoop<NCols>([&](size_t i) { acc[i] = vfmaq_f32(acc[i], av[j], bv[i][j]); });
            });
        }

        // increment pointers to next block
        QuantBData += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        QuantBScale += 1;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }

    if constexpr (NCols == 4) {
        float32x4_t sum = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);

        if (BiasPtr != nullptr) {
            sum = vaddq_f32(sum, vld1q_f32(BiasPtr));
        }

        vst1q_f32(SumPtr, sum);
    } else {
        for (size_t i = 0; i < NCols; ++i) {
            SumPtr[i] = vaddvq_f32(acc[i]);
            if (BiasPtr != nullptr) {
                SumPtr[i] += BiasPtr[i];
            }
        }
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t NCols = 4;

    const float* ARowPtr = A;
    float* CRowPtr = C;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<NCols, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompFp32<1, HasZeroPoint>(
            BlkLen,
            ARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        constexpr bool HasZeroPoint = true;
        SQ4BitGemmM1Kernel_CompFp32_Impl<HasZeroPoint>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockCountK,
            Bias
        );
    } else {
        constexpr bool HasZeroPoint = false;
        SQ4BitGemmM1Kernel_CompFp32_Impl<HasZeroPoint>(
            BlkLen,
            A,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountN,
            CountK,
            BlockCountK,
            Bias
        );
    }
}

// Block dequantize a 16 x NCols section of B from column major source to row major destination.
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4BitBlkDequantB_16xNCols(
    const std::byte* QuantBDataPtr,
    size_t StrideQuantBData,
    const float* QuantBColScalePtr,                    // pointer to NCols scales of adjacent columns
    [[maybe_unused]] const float* QuantBColOffsetPtr,  // pointer to NCols offsets of adjacent columns
                                                       // only used if HasZeroPoint is true
    float* DstColPtr
)
{
    const uint8x8_t LowMask = vdup_n_u8(0x0F);

    // load B column vectors
    uint8x8_t bv_packed[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
        bv_packed[i] = vld1_u8(
            reinterpret_cast<const uint8_t*>(QuantBDataPtr) + i * StrideQuantBData
        );
    });

    uint8x8_t bv_u8[NCols][2];
    UnrolledLoop<NCols>([&](size_t i) {
        bv_u8[i][0] = vand_u8(bv_packed[i], LowMask);
        bv_u8[i][1] = vshr_n_u8(bv_packed[i], 4);
    });

    // shift left 3 and widen to 16 bits
    uint16x8_t bv_u16[NCols][2];
    UnrolledLoop<NCols>([&](size_t i) {
        constexpr int shift = 3;
        bv_u16[i][0] = vshll_n_u8(bv_u8[i][0], shift);
        bv_u16[i][1] = vshll_n_u8(bv_u8[i][1], shift);
    });

    // combine 4 bits with float high half template
    UnrolledLoop<NCols>([&](size_t i) {
        bv_u16[i][0] = vorrq_u16(bv_u16[i][0], fp32_conversion::float_high_half_template_v);
        bv_u16[i][1] = vorrq_u16(bv_u16[i][1], fp32_conversion::float_high_half_template_v);
    });

    // `SubBlkLen` floats of B
    float32x4_t bv[NCols][4];

    // shift left 16, widen to 32 bits, and reinterpret as float
    UnrolledLoop<NCols>([&](size_t i) {
        constexpr int shift = 16;
        bv[i][0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][0]), shift));
        bv[i][1] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][0], shift));

        bv[i][2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bv_u16[i][1]), shift));
        bv[i][3] = vreinterpretq_f32_u32(vshll_high_n_u16(bv_u16[i][1], shift));
    });

    // subtract float conversion offset and zero point
    if constexpr (HasZeroPoint) {
        UnrolledLoop<NCols>([&](size_t i) {
            const float32x4_t offset_v = vdupq_n_f32(QuantBColOffsetPtr[i]);
            UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
        });
    } else {
        const float32x4_t offset_v = vdupq_n_f32(fp32_conversion::offset + 8.0f);
        UnrolledLoop<NCols>([&](size_t i) {
            UnrolledLoop<4>([&](size_t j) { bv[i][j] = vsubq_f32(bv[i][j], offset_v); });
        });
    }

    // multiply by scale
    UnrolledLoop<NCols>([&](size_t i) {
        const float32x4_t scale_v = vdupq_n_f32(QuantBColScalePtr[i]);
        UnrolledLoop<4>([&](size_t j) { bv[i][j] = vmulq_f32(bv[i][j], scale_v); });
    });

    // write, transposed, 16 x NCols values
    if constexpr (NCols == 4) {
        UnrolledLoop<4>([&](size_t j) {
            Transpose4x4(bv[0][j], bv[1][j], bv[2][j], bv[3][j]);

            vst1q_f32(&DstColPtr[(j * 4 + 0) * 16], bv[0][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 1) * 16], bv[1][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 2) * 16], bv[2][j]);
            vst1q_f32(&DstColPtr[(j * 4 + 3) * 16], bv[3][j]);
        });
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            UnrolledLoop<4>([&](size_t j) {
                DstColPtr[(j * 4 + 0) * 16 + i] = vgetq_lane_f32(bv[i][j], 0);
                DstColPtr[(j * 4 + 1) * 16 + i] = vgetq_lane_f32(bv[i][j], 1);
                DstColPtr[(j * 4 + 2) * 16 + i] = vgetq_lane_f32(bv[i][j], 2);
                DstColPtr[(j * 4 + 3) * 16 + i] = vgetq_lane_f32(bv[i][j], 3);
            });
        });
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    constexpr size_t BlkBitWidth = 4;

    float* Dst = FpData;

    const std::byte* QuantBDataCol = QuantBData;
    const float* QuantBScaleCol = QuantBScale;
    [[maybe_unused]] const std::byte* QuantBZeroPointCol = QuantBZeroPoint;  // only used if HasZeroPoint is true

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    [[maybe_unused]] const size_t StrideQuantBZeroPoint =  // only used if HasZeroPoint is true
        MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    //
    // Proceed down 16 column-wide regions of B. Dequantize and write output 16 x 16 elements at a time.
    //

    // scales of blocks from 16 adjacent columns
    float scale[16];
    // float conversion offsets (including zero point) of blocks from 16 adjacent columns
    [[maybe_unused]] float offset[16];  // only used if HasZeroPoint is true

    size_t n_cols_remaining = CountN;
    while (n_cols_remaining > 15) {
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            for (size_t nn = 0; nn < 16; ++nn) {
                scale[nn] = QuantBScaleCol[nn * BlockCountK + k_blk_idx];

                if constexpr (HasZeroPoint) {
                    const std::byte zp_packed =
                        QuantBZeroPointCol[nn * StrideQuantBZeroPoint + k_blk_idx / 2];
                    const std::byte zp = ((k_blk_idx & 1) == 1)
                                             ? (zp_packed >> 4)
                                             : (zp_packed & std::byte{0x0F});
                    offset[nn] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
                }
            }

            const size_t kklen = std::min(CountK - k, BlkLen);

            for (size_t kk = 0; kk < kklen; kk += 16) {
                constexpr size_t NCols = 4;

                const float* ScalePtr = &scale[0];
                const float* OffsetPtr = HasZeroPoint ? &offset[0] : nullptr;

                float* DstColPtr = Dst;

                for (size_t nn = 0; nn < 16; nn += NCols) {
                    const std::byte* QuantBDataPtr = QuantBDataCol + nn * StrideQuantBData + (k + kk) * BlkBitWidth / 8;

                    Q4BitBlkDequantB_16xNCols<NCols, HasZeroPoint>(
                        QuantBDataPtr,
                        StrideQuantBData,
                        ScalePtr,
                        OffsetPtr,
                        DstColPtr
                    );

                    ScalePtr += NCols;
                    if constexpr (HasZeroPoint) {
                        OffsetPtr += NCols;
                    }
                    DstColPtr += NCols;
                }

                Dst += 16 * std::min(kklen - kk, size_t{16});
            }
        }

        n_cols_remaining -= 16;

        QuantBDataCol += 16 * StrideQuantBData;
        QuantBScaleCol += 16 * BlockCountK;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointCol += 16 * StrideQuantBZeroPoint;
        }
    }

    if (n_cols_remaining > 0) {
        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            for (size_t nn = 0; nn < n_cols_remaining; ++nn) {
                scale[nn] = QuantBScaleCol[nn * BlockCountK + k_blk_idx];

                if constexpr (HasZeroPoint) {
                    const std::byte zp_packed =
                        QuantBZeroPointCol[nn * StrideQuantBZeroPoint + k_blk_idx / 2];
                    const std::byte zp = ((k_blk_idx & 1) == 1)
                                             ? (zp_packed >> 4)
                                             : (zp_packed & std::byte{0x0F});
                    offset[nn] = fp32_conversion::offset + std::to_integer<uint8_t>(zp);
                }
            }

            const size_t kklen = std::min(CountK - k, BlkLen);

            for (size_t kk = 0; kk < kklen; kk += 16) {
                // zero out the 16x16 block in Dst first to ensure zero padding
                const float32x4_t zero_v = vdupq_n_f32(0.0f);
                UnrolledLoop<16 * 4>([&](size_t i) {
                    vst1q_f32(Dst + 4 * i, zero_v);
                });

                const float* ScalePtr = &scale[0];
                const float* OffsetPtr = HasZeroPoint ? &offset[0] : nullptr;

                float* DstColPtr = Dst;

                for (size_t nn = 0; nn < n_cols_remaining; ++nn) {
                    const std::byte* QuantBDataPtr = QuantBDataCol + nn * StrideQuantBData + (k + kk) * BlkBitWidth / 8;

                    Q4BitBlkDequantB_16xNCols<1, HasZeroPoint>(
                        QuantBDataPtr,
                        StrideQuantBData,
                        ScalePtr,
                        OffsetPtr,
                        DstColPtr
                    );

                    ScalePtr += 1;
                    if constexpr (HasZeroPoint) {
                        OffsetPtr += 1;
                    }
                    DstColPtr += 1;
                }

                Dst += 16 * std::min(kklen - kk, size_t{16});
            }
        }
    }
}

void
Q4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen,
            FpData,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            CountN,
            CountK,
            BlockCountK
        );
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen,
            FpData,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            CountN,
            CountK,
            BlockCountK
        );
    }
}

//
// CompInt8 kernel implementation.
//

template <size_t SubBlkLen>
MLAS_FORCEINLINE void
QuantizeBlock(
    size_t BlkLen,
    const float* A,
    size_t ElementCount,
    std::byte* QuantA
)
{
    static_assert(SubBlkLen >= 16 && SubBlkLen % 16 == 0);

    assert(BlkLen % SubBlkLen == 0);

    //
    // Scan block values first to determine scale.
    //

    float amax = 0.0f;  // max of absolute values of A block

    size_t k;
    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[SubBlkLen / 4]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        float32x4_t abs_a[SubBlkLen / 4];
        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            abs_a[i] = vabsq_f32(a[i]);
        });

        // find amax of SubBlkLen elements
        for (size_t interval = SubBlkLen / 4 / 2; interval > 0; interval /= 2) {
            for (size_t i = 0; i < interval; ++i) {
                abs_a[i] = vmaxq_f32(abs_a[i], abs_a[i + interval]);
            }
        }

        // update existing amax
        amax = std::max(amax, vmaxvq_f32(abs_a[0]));
    }

    constexpr float range_max = (1 << 7) - 1;
    const float scale = amax / range_max;
    const float scale_reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;

    Q8BlkScale(QuantA) = scale;

    //
    // Compute quantized block values.
    //

    int8_t* QuantAData = Q8BlkData(QuantA);

    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[SubBlkLen / 4]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            a[i] = vmulq_n_f32(a[i], scale_reciprocal);
        });

        int32x4_t a_s32[SubBlkLen / 4];
        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            a_s32[i] = vcvtaq_s32_f32(a[i]);
        });

        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            QuantAData[k + i * 4 + 0] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 0));
            QuantAData[k + i * 4 + 1] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 1));
            QuantAData[k + i * 4 + 2] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 2));
            QuantAData[k + i * 4 + 3] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 3));
        });
    }

    //
    // Zero out any remaining sub-block elements.
    //

    for (; k < BlkLen; k += SubBlkLen) {
        const int8x16_t Zeros = vdupq_n_s8(0);
        UnrolledLoop<SubBlkLen / 16>([&](size_t i) {
            vst1q_s8(QuantAData + k + i * 16, Zeros);
        });
    }
}

void
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    const float* ADataBlkPtr = A;
    std::byte* QuantABlkPtr = QuantA;

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        QuantizeBlock<16>(BlkLen, ADataBlkPtr, k_blk_len, QuantABlkPtr);

        ADataBlkPtr += BlkLen;
        QuantABlkPtr += Q8BlkSize(BlkLen);
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute2x2_BlkLen16(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK,
    size_t StrideQuantA,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    size_t ldc
)
{
    constexpr size_t BlkLen = 16;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc00{}, acc01{}, acc10{}, acc11{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlkRow0 = QuantAPtr;
        const std::byte* QuantABlkRow1 = QuantAPtr + StrideQuantA;

        const float QuantBScaleCol0 = *QuantBScalePtr;
        const float QuantBScaleCol1 = *(QuantBScalePtr + StrideQuantBScale);

        // compute combined scales
        const float scale00 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol0;
        const float scale01 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol1;
        const float scale10 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol0;
        const float scale11 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol1;

        // load B zero point
        int8_t bzp_col0;
        int8_t bzp_col1;
        if constexpr (HasZeroPoint) {
            const std::byte QuantBZeroPointByteCol0 = *QuantBZeroPointPtr;
            const std::byte QuantBZeroPointByteCol1 = *(QuantBZeroPointPtr + StrideQuantBZeroPoint);
            if ((k_blk_idx & 1) == 0) {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 & std::byte{0x0F});
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 & std::byte{0x0F});
            } else {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 >> 4);
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 >> 4);
            }
        } else {
            bzp_col0 = 8;
            bzp_col1 = 8;
        }

        const int8_t* QuantADataPtrRow0 = Q8BlkData(QuantABlkRow0);
        const int8_t* QuantADataPtrRow1 = Q8BlkData(QuantABlkRow1);

        // TODO handling only 16 elements per accumulator at a time here, probably can do better
        {
            // load A
            const int8x16_t av_row0 = vld1q_s8(QuantADataPtrRow0 + 0);
            const int8x16_t av_row1 = vld1q_s8(QuantADataPtrRow1 + 0);

            // load B
            const uint8x8_t bv_packed_col0 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x8_t bv_packed_col1 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + StrideQuantBData);

            const uint8x8_t LowMaskU8x8 = vdup_n_u8(0x0F);

            int8x16_t bv_col0 = vreinterpretq_s8_u8(
                vcombine_u8(
                    vand_u8(bv_packed_col0, LowMaskU8x8),
                    vshr_n_u8(bv_packed_col0, 4)
                )
            );
            int8x16_t bv_col1 = vreinterpretq_s8_u8(
                vcombine_u8(
                    vand_u8(bv_packed_col1, LowMaskU8x8),
                    vshr_n_u8(bv_packed_col1, 4)
                )
            );

            // subtract B zero point
            bv_col0 = vsubq_s8(bv_col0, vdupq_n_s8(bzp_col0));
            bv_col1 = vsubq_s8(bv_col1, vdupq_n_s8(bzp_col1));

            // quantized dot product
            int32x4_t dot00{}, dot01{}, dot10{}, dot11{};
            dot00 = vdotq_s32(dot00, av_row0, bv_col0);
            dot01 = vdotq_s32(dot01, av_row0, bv_col1);
            dot10 = vdotq_s32(dot10, av_row1, bv_col0);
            dot11 = vdotq_s32(dot11, av_row1, bv_col1);

            // convert to float
            const float32x4_t dot_f32_00 = vcvtq_f32_s32(dot00);
            const float32x4_t dot_f32_01 = vcvtq_f32_s32(dot01);
            const float32x4_t dot_f32_10 = vcvtq_f32_s32(dot10);
            const float32x4_t dot_f32_11 = vcvtq_f32_s32(dot11);

            // multiply by scale and update accumulator
            acc00 = vfmaq_f32(acc00, dot_f32_00, vdupq_n_f32(scale00));
            acc01 = vfmaq_f32(acc01, dot_f32_01, vdupq_n_f32(scale01));
            acc10 = vfmaq_f32(acc10, dot_f32_10, vdupq_n_f32(scale10));
            acc11 = vfmaq_f32(acc11, dot_f32_11, vdupq_n_f32(scale11));
        }

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBDataPtr += 8;
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    SumPtr[0] = vaddvq_f32(acc00);
    SumPtr[1] = vaddvq_f32(acc01);
    SumPtr[ldc + 0] = vaddvq_f32(acc10);
    SumPtr[ldc + 1] = vaddvq_f32(acc11);

    if (BiasPtr != nullptr) {
        SumPtr[0] += BiasPtr[0];
        SumPtr[1] += BiasPtr[1];
        SumPtr[ldc + 0] += BiasPtr[0];
        SumPtr[ldc + 1] += BiasPtr[1];
    }
}

template<bool HasZeroPoint>
MLAS_FORCEINLINE
void SQ4BitGemm_CompInt8_Compute2x2_BlkLenGreaterThan16(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK,
    size_t StrideQuantA,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    size_t ldc
)
{
    // process blocks in 32-element sub-blocks
    const size_t SubBlksPerBlk = BlkLen / 32;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc00{}, acc01{}, acc10{}, acc11{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlkRow0 = QuantAPtr;
        const std::byte* QuantABlkRow1 = QuantAPtr + StrideQuantA;

        const float QuantBScaleCol0 = *QuantBScalePtr;
        const float QuantBScaleCol1 = *(QuantBScalePtr + StrideQuantBScale);

        // compute combined scales
        const float scale00 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol0;
        const float scale01 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol1;
        const float scale10 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol0;
        const float scale11 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol1;

        // load B zero point
        int8_t bzp_col0;
        int8_t bzp_col1;
        if constexpr (HasZeroPoint) {
            const std::byte QuantBZeroPointByteCol0 = *QuantBZeroPointPtr;
            const std::byte QuantBZeroPointByteCol1 = *(QuantBZeroPointPtr + StrideQuantBZeroPoint);
            if ((k_blk_idx & 1) == 0) {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 & std::byte{0x0F});
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 & std::byte{0x0F});
            } else {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 >> 4);
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 >> 4);
            }
        } else {
            bzp_col0 = 8;
            bzp_col1 = 8;
        }

        const int8_t* QuantADataPtrRow0 = Q8BlkData(QuantABlkRow0);
        const int8_t* QuantADataPtrRow1 = Q8BlkData(QuantABlkRow1);

        for (size_t sub_blk_idx = 0; sub_blk_idx < SubBlksPerBlk; ++sub_blk_idx) {
            // load A
            const int8x16_t av_row0_0 = vld1q_s8(QuantADataPtrRow0 + 0);
            const int8x16_t av_row0_1 = vld1q_s8(QuantADataPtrRow0 + 16);
            const int8x16_t av_row1_0 = vld1q_s8(QuantADataPtrRow1 + 0);
            const int8x16_t av_row1_1 = vld1q_s8(QuantADataPtrRow1 + 16);

            // load B
            const uint8x16_t bv_packed_col0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x16_t bv_packed_col1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + StrideQuantBData);

            const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

            int8x16_t bv_col0_0 = vreinterpretq_s8_u8(vandq_u8(bv_packed_col0, LowMaskU8x16));
            int8x16_t bv_col0_1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed_col0, 4));
            int8x16_t bv_col1_0 = vreinterpretq_s8_u8(vandq_u8(bv_packed_col1, LowMaskU8x16));
            int8x16_t bv_col1_1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed_col1, 4));

            // subtract B zero point
            bv_col0_0 = vsubq_s8(bv_col0_0, vdupq_n_s8(bzp_col0));
            bv_col0_1 = vsubq_s8(bv_col0_1, vdupq_n_s8(bzp_col0));
            bv_col1_0 = vsubq_s8(bv_col1_0, vdupq_n_s8(bzp_col1));
            bv_col1_1 = vsubq_s8(bv_col1_1, vdupq_n_s8(bzp_col1));

            // quantized dot product
            int32x4_t dot00{}, dot01{}, dot10{}, dot11{};
            dot00 = vdotq_s32(vdotq_s32(dot00, av_row0_0, bv_col0_0), av_row0_1, bv_col0_1);
            dot01 = vdotq_s32(vdotq_s32(dot01, av_row0_0, bv_col1_0), av_row0_1, bv_col1_1);
            dot10 = vdotq_s32(vdotq_s32(dot10, av_row1_0, bv_col0_0), av_row1_1, bv_col0_1);
            dot11 = vdotq_s32(vdotq_s32(dot11, av_row1_0, bv_col1_0), av_row1_1, bv_col1_1);

            // convert to float
            const float32x4_t dot_f32_00 = vcvtq_f32_s32(dot00);
            const float32x4_t dot_f32_01 = vcvtq_f32_s32(dot01);
            const float32x4_t dot_f32_10 = vcvtq_f32_s32(dot10);
            const float32x4_t dot_f32_11 = vcvtq_f32_s32(dot11);

            // multiply by scale and update accumulator
            acc00 = vfmaq_f32(acc00, dot_f32_00, vdupq_n_f32(scale00));
            acc01 = vfmaq_f32(acc01, dot_f32_01, vdupq_n_f32(scale01));
            acc10 = vfmaq_f32(acc10, dot_f32_10, vdupq_n_f32(scale10));
            acc11 = vfmaq_f32(acc11, dot_f32_11, vdupq_n_f32(scale11));

            // increment block data pointers to next sub-block
            QuantADataPtrRow0 += 32;
            QuantADataPtrRow1 += 32;
            QuantBDataPtr += 16;
        }

        // increment other block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    SumPtr[0] = vaddvq_f32(acc00);
    SumPtr[1] = vaddvq_f32(acc01);
    SumPtr[ldc + 0] = vaddvq_f32(acc10);
    SumPtr[ldc + 1] = vaddvq_f32(acc11);

    if (BiasPtr != nullptr) {
        SumPtr[0] += BiasPtr[0];
        SumPtr[1] += BiasPtr[1];
        SumPtr[ldc + 0] += BiasPtr[0];
        SumPtr[ldc + 1] += BiasPtr[1];
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute1x1_BlkLen16(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    constexpr size_t BlkLen = 16;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
        const std::byte* QuantABlk0 = QuantAPtr;
        const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
        const float32x4_t scale1 = vdupq_n_f32(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );
        const int8x16_t bzp1 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4) : 8
        );

        // load A
        const int8x16_t av0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av1 = vld1q_s8(Q8BlkData(QuantABlk1));

        // load B
        const uint8x16_t bv_packed01 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

        const uint8x16_t bv_lo01 = vandq_u8(bv_packed01, LowMaskU8x16);
        const uint8x16_t bv_hi01 = vshrq_n_u8(bv_packed01, 4);

        int8x16_t bv0 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(bv_lo01), vget_low_u8(bv_hi01)));
        int8x16_t bv1 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(bv_lo01), vget_high_u8(bv_hi01)));

        // subtract B zero point
        bv0 = vsubq_s8(bv0, bzp0);
        bv1 = vsubq_s8(bv1, bzp1);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(vdupq_n_s32(0), av0, bv0);
        const int32x4_t dot1 = vdotq_s32(vdupq_n_s32(0), av1, bv1);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
        const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
        acc1 = vfmaq_f32(acc1, dot_f32_1, scale1);

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen) * 2;
        QuantBDataPtr += 8 * 2;
        QuantBScalePtr += 2;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += 1;
        }
    }

    if (k_blks_remaining > 0) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * (*QuantBScalePtr));

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );

        // load A
        const int8x16_t av0 = vld1q_s8(Q8BlkData(QuantABlk0));

        // load B
        const uint8x8_t bv_packed0 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        const uint8x8_t LowMaskU8x8 = vdup_n_u8(0x0F);

        const uint8x8_t bv_lo0 = vand_u8(bv_packed0, LowMaskU8x8);
        const uint8x8_t bv_hi0 = vshr_n_u8(bv_packed0, 4);

        int8x16_t bv0 = vreinterpretq_s8_u8(vcombine_u8(bv_lo0, bv_hi0));

        // subtract B zero point
        bv0 = vsubq_s8(bv0, bzp0);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(vdupq_n_s32(0), av0, bv0);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE
void
SQ4BitGemm_CompInt8_Compute1x1_BlkLen32(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    constexpr size_t BlkLen = 32;

    const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
        const std::byte* QuantABlk0 = QuantAPtr;
        const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
        const float32x4_t scale1 = vdupq_n_f32(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );
        const int8x16_t bzp1 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4) : 8
        );

        // load A
        const int8x16_t av_lo0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av_hi0 = vld1q_s8(Q8BlkData(QuantABlk0) + 16);
        const int8x16_t av_lo1 = vld1q_s8(Q8BlkData(QuantABlk1));
        const int8x16_t av_hi1 = vld1q_s8(Q8BlkData(QuantABlk1) + 16);

        // load B
        const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
        const uint8x16_t bv_packed1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + 16);

        int8x16_t bv_lo0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
        int8x16_t bv_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));
        int8x16_t bv_lo1 = vreinterpretq_s8_u8(vandq_u8(bv_packed1, LowMaskU8x16));
        int8x16_t bv_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed1, 4));

        // subtract B zero point
        bv_lo0 = vsubq_s8(bv_lo0, bzp0);
        bv_hi0 = vsubq_s8(bv_hi0, bzp0);
        bv_lo1 = vsubq_s8(bv_lo1, bzp1);
        bv_hi1 = vsubq_s8(bv_hi1, bzp1);

        // quantized dot product
        int32x4_t dot0{}, dot1{};
        dot0 = vdotq_s32(vdotq_s32(dot0, av_lo0, bv_lo0), av_hi0, bv_hi0);
        dot1 = vdotq_s32(vdotq_s32(dot1, av_lo1, bv_lo1), av_hi1, bv_hi1);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
        const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
        acc1 = vfmaq_f32(acc1, dot_f32_1, scale1);

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen) * 2;
        QuantBDataPtr += 16 * 2;
        QuantBScalePtr += 2;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += 1;
        }
    }

    if (k_blks_remaining > 0) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * (*QuantBScalePtr));

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );

        // load A
        const int8x16_t av_lo0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av_hi0 = vld1q_s8(Q8BlkData(QuantABlk0) + 16);

        // load B
        const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        int8x16_t bv_lo0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
        int8x16_t bv_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));

        // subtract B zero point
        bv_lo0 = vsubq_s8(bv_lo0, bzp0);
        bv_hi0 = vsubq_s8(bv_hi0, bzp0);

        // quantized dot product
        int32x4_t dot0{};
        dot0 = vdotq_s32(vdotq_s32(dot0, av_lo0, bv_lo0), av_hi0, bv_hi0);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

    // process blocks in 32-element sub-blocks
    const size_t SubBlksPerBlk = BlkLen / 32;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);

        // load B zero point
        const int8x16_t bzp = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );

        const int8_t* QuantADataPtr = Q8BlkData(QuantAPtr);

        for (size_t sub_blk_idx = 0; sub_blk_idx < SubBlksPerBlk; sub_blk_idx += 2) {
            // load A
            const int8x16_t av0 = vld1q_s8(QuantADataPtr + 0);
            const int8x16_t av1 = vld1q_s8(QuantADataPtr + 16);
            const int8x16_t av2 = vld1q_s8(QuantADataPtr + 32);
            const int8x16_t av3 = vld1q_s8(QuantADataPtr + 48);

            // load B
            const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x16_t bv_packed1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + 16);

            int8x16_t bv0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
            int8x16_t bv1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));
            int8x16_t bv2 = vreinterpretq_s8_u8(vandq_u8(bv_packed1, LowMaskU8x16));
            int8x16_t bv3 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed1, 4));

            // subtract B zero point
            bv0 = vsubq_s8(bv0, bzp);
            bv1 = vsubq_s8(bv1, bzp);
            bv2 = vsubq_s8(bv2, bzp);
            bv3 = vsubq_s8(bv3, bzp);

            // quantized dot product
            int32x4_t dot0{}, dot1{};
            dot0 = vdotq_s32(vdotq_s32(dot0, av0, bv0), av1, bv1);
            dot1 = vdotq_s32(vdotq_s32(dot1, av2, bv2), av3, bv3);

            // convert to float
            const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
            const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

            // multiply by scale and update accumulator
            acc0 = vfmaq_f32(acc0, dot_f32_0, scale);
            acc1 = vfmaq_f32(acc1, dot_f32_1, scale);

            // increment block data pointers to next sub-block
            QuantADataPtr += 16 * 4;
            QuantBDataPtr += 16 * 2;
        }

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_BlkLen16(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BlkLen = 16;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 1) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 2x2 tiles of output
            SQ4BitGemm_CompInt8_Compute2x2_BlkLen16<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // move to next 2 columns

            QuantBDataColPtr += 2 * StrideQuantBData;
            QuantBScaleColPtr += 2 * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += 2 * StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 2 : 0;
            SumPtr += 2;

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 2x1 tile of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen16<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen16<HasZeroPoint>(
                QuantARowPtr + StrideQuantA,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr + ldc,
                BlockCountK
            );
        }

        // move to next 2 rows

        QuantARowPtr += 2 * StrideQuantA;
        SumRowPtr += 2 * ldc;

        m_remaining -= 2;
    }

    if (m_remaining > 0) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen16<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // move to next column

            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;

            n_remaining -= 1;
        }
    }
}

template <bool HasZeroPoint>
void SQ4BitGemmKernel_CompInt8_BlkLen32(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BlkLen = 32;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 1) {

        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 2x2 tiles of output
            SQ4BitGemm_CompInt8_Compute2x2_BlkLenGreaterThan16<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // move to next 2 columns

            QuantBDataColPtr += 2 * StrideQuantBData;
            QuantBScaleColPtr += 2 * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += 2 * StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 2 : 0;
            SumPtr += 2;

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 2x1 tile of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen32<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen32<HasZeroPoint>(
                QuantARowPtr + StrideQuantA,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr + ldc,
                BlockCountK
            );

        }

        // move to next 2 rows

        QuantARowPtr += 2 * StrideQuantA;
        SumRowPtr += 2 * ldc;

        m_remaining -= 2;

    }

    if (m_remaining > 0) {

        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLen32<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // move to next column

            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;

            n_remaining -= 1;
        }

    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_BlkLenGreaterThan32(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 1) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 2x2 tiles of output
            SQ4BitGemm_CompInt8_Compute2x2_BlkLenGreaterThan16<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // move to next 2 columns

            QuantBDataColPtr += 2 * StrideQuantBData;
            QuantBScaleColPtr += 2 * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += 2 * StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 2 : 0;
            SumPtr += 2;

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 2x1 tile of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32<HasZeroPoint>(
                BlkLen,
                QuantARowPtr + StrideQuantA,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr + ldc,
                BlockCountK
            );
        }

        // move to next 2 rows

        QuantARowPtr += 2 * StrideQuantA;
        SumRowPtr += 2 * ldc;

        m_remaining -= 2;
    }

    if (m_remaining > 0) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output

            SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // move to next column

            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;

            n_remaining -= 1;
        }
    }
}

template<bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (BlkLen == 16) {
        SQ4BitGemmKernel_CompInt8_BlkLen16<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    }
    else if (BlkLen == 32) {
        SQ4BitGemmKernel_CompInt8_BlkLen32<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    } else {
        SQ4BitGemmKernel_CompInt8_BlkLenGreaterThan32<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    }
}

size_t
SQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        constexpr bool HasZeroPoint = true;
        SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    } else {
        constexpr bool HasZeroPoint = false;
        SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    }

    return CountM;
}

}  // namespace

//
// Kernel dispatch structure definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchNeon = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmPerGemmWorkspaceSize = SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32;

    d.SQ4BitGemmKernel_CompInt8 = SQ4BitGemmKernel_CompInt8;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8;

    return d;
}();
