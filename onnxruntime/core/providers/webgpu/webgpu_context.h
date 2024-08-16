// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <memory>
#include <mutex>

#include <webgpu/webgpu_cpp.h>

#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

// Class WebGpuContext includes all necessary resources for the context.
class WebGpuContext {
 public:
  WebGpuContext() {}

  void Init();

  Status Wait(wgpu::Future f) {
    auto status = instance_.WaitAny(f, UINT64_MAX);

    if (status == wgpu::WaitStatus::Success) {
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to wait for the operation:", uint32_t(status));
  }

  // wgpu::Instance GetInstance() { return instance_; }
  wgpu::Adapter Adapter() { return adapter_; }
  wgpu::Device Device() { return device_; }

  IBufferManager& BufferManager() { return *buffer_mgr_; }

 private:
  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;

  std::unique_ptr<IBufferManager> buffer_mgr_;
};

WebGpuContext& GetContext();

}  // namespace webgpu
}  // namespace onnxruntime
