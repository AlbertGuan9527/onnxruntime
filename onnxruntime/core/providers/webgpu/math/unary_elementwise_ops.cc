// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status UnaryElementwise::Prepare(OpKernelContext* context, UnaryElementwisePreparation* p) const {
  p->input_tensor = context->Input<Tensor>(0);
  p->output_tensor = context->Output(0, p->input_tensor->Shape());
  return Status::OK();
}

Status UnaryElementwise::ComputeInternal(OpKernelContext* context) const {
  UnaryElementwisePreparation preparation;
  ORT_RETURN_IF_ERROR(Prepare(context, &preparation));
  return Status(common::ONNXRUNTIME, common::FAIL);
}

#define WEBGPU_ELEMENTWISE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE)  \
  ONNX_OPERATOR_KERNEL_EX(                                             \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,         \
      KernelDefBuilder().TypeConstraint("T", TYPE),                    \
      KERNEL_CLASS);

#define WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                             \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                  \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                              \
      KERNEL_CLASS);

WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Abs, 6, 12, Abs, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Abs, 13, Abs, WebGpuSupportedFloatTypes())


}  // namespace webgpu
}  // namespace onnxruntime
