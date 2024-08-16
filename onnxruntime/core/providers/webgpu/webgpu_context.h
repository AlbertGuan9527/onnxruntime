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
class WebGpuContext final {
 public:
  void Initialize();

  Status Wait(wgpu::Future f) const;

  const wgpu::Adapter& Adapter() const { return adapter_; }
  const wgpu::Device& Device() const { return device_; }

  const IBufferManager& BufferManager() const { return *buffer_mgr_; }

 private:
  WebGpuContext() {}

  std::once_flag init_flag_;

  mutable wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;

  std::unique_ptr<IBufferManager> buffer_mgr_;

  friend class WebGpuContextFactory;
};

class WebGpuContextFactory {
 public:
  static WebGpuContext& GetOrCreateContext(int32_t context_id = 0);

 private:
  WebGpuContextFactory() {}

  static std::unordered_map<int32_t, std::unique_ptr<WebGpuContext>> contexts_;
  static std::mutex mutex_;
};

}  // namespace webgpu
}  // namespace onnxruntime
