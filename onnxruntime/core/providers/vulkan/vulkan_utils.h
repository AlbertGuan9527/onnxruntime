// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/providers/vulkan/vulkan_execution_provider.h"

#include "ncnn-src/src/layer.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/kernel_def_builder.h"

#define REGISTER_ONNX_OPERATOR_VULKAN_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kVulkanExecutionProvider, builder, __VA_ARGS__)

#define REGISTER_ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL(name, since_version, end_version, builder, ...)      \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, since_version, end_version, kVulkanExecutionProvider, builder, \
                          __VA_ARGS__)

namespace onnxruntime {
namespace vulkan {
namespace {
template <typename TMat>
void InitMatFromTensor(const Tensor& tensor, TMat& mat) {
  const auto& shape = tensor.Shape();
  const size_t rank = shape.NumDimensions();
  const int64_t num_elements = shape.Size();
  const size_t element_size = tensor.DataType()->Size();

  ORT_ENFORCE(rank > 0, "TODO: do scalars need to be 1D?");

  mat.elemsize = element_size;
  mat.elempack = 1;

  // NCNN doesn't support batches so the dims are C, D, H, W where 'D' is depth.
  // For now map the same way (3D -> C, 1, H, W).
  mat.dims = gsl::narrow_cast<int32_t>(rank);
  mat.w = gsl::narrow_cast<int32_t>(shape[rank - 1]);
  mat.h = 1;
  mat.d = 1;
  mat.c = 1;

  switch (shape.NumDimensions()) {
    case 1:
      break;
    case 2:
      mat.h = gsl::narrow_cast<int32_t>(shape[0]);
      break;
    case 3:
      mat.c = gsl::narrow_cast<int32_t>(shape[0]);
      mat.h = gsl::narrow_cast<int32_t>(shape[1]);
      break;
    case 4:
      mat.c = gsl::narrow_cast<int32_t>(shape[0]);
      mat.d = gsl::narrow_cast<int32_t>(shape[1]);
      mat.h = gsl::narrow_cast<int32_t>(shape[2]);
      break;
    default:
      ORT_THROW("Tensor shape is not supported in Vulkan EP. Must be 4D or less. shape:", shape);
  }

  mat.cstep = ncnn::alignSize(narrow<size_t>(num_elements * mat.elemsize), 16) / mat.elemsize;
  auto bytes_required = mat.cstep * mat.c;

  // NCNN uses a few bytes past the end of the allocation for the VkMat refernece counter.
  // we're not directly using the reference counter (we set it to nullptr) but it may happen if there are internal
  // allocations  made by NCNN (e.g. the Convolution kernel uses a Padding Layer internally). we don't control those.
  //
  // GPU memory allocated with the Vulkan EP allocator adds NCNN_MALLOC_OVERREAD to match NCNN so _should_ be safe.
  //
  // Not sure of a good way to check/ensure that is always the case.
  // Putting this here for now to see if it's hit. If it is we need to double check how much additional buffer our
  // allocations need.
  ORT_ENFORCE(bytes_required <= tensor.CalculateTensorStorageSize(tensor.DataType(), tensor.Shape()),
              "Need extra buffer in allocation for NCNN");
}
}  // namespace

const VulkanExecutionProvider& GetVulkanExecutionProvider(const onnxruntime::OpKernelInfo& info) {
  return *static_cast<const VulkanExecutionProvider*>(info.GetExecutionProvider());
}

// Get the index of the layer in the ncnn model. Throws if not found.
int GetNcnnLayerIndex(const std::string& layer_name) {
  int index = ncnn::layer_to_index(layer_name.c_str());
  if (index == -1) {
    // should never happen outside of during development
    ORT_THROW("Failed to find ", layer_name, " in the NCNN kernels.");
  }

  return index;
}

ncnn::Mat TensorToMat(const Tensor& tensor) {
  ncnn::Mat mat;

  InitMatFromTensor(tensor, mat);
  return mat;
}

ncnn::VkMat TensorToVkMat(const Tensor& tensor, ncnn::VkAllocator& allocator) {
  ncnn::VkMat vkmat;

  InitMatFromTensor(tensor, vkmat);
  vkmat.allocator = &allocator;

  return vkmat;
}

}  // namespace vulkan
}  // namespace onnxruntime
