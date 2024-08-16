// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

struct UnaryElementwisePreparation {
  const Tensor* input_tensor = nullptr;
  Tensor* output_tensor = nullptr;
};

class UnaryElementwise : public WebGpuKernel {
 protected:
  UnaryElementwise(const OpKernelInfo& info) : WebGpuKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override;
  Status Prepare(OpKernelContext* context, UnaryElementwisePreparation* p) const;
};

class Abs final : public UnaryElementwise {
 public:
  Abs(const OpKernelInfo& info) : UnaryElementwise(info) {}
  // Status ComputeInternal(OpKernelContext* context) const override;
};


}  // namespace webgpu
}  // namespace onnxruntime
