// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/operators/conv_transpose.h"
namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    FusedConvTranspose,
    kMSDomain,
    1,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    onnxruntime::js::ConvTranspose<false, true>);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
