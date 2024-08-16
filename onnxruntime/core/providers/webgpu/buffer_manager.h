// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class IBufferCacheManager {
 public:
  virtual ~IBufferCacheManager() = default;

  virtual wgpu::Buffer GetBuffer(wgpu::Device device, size_t size) = 0;
  virtual void ReleaseBuffer(wgpu::Buffer buffer) = 0;
};

class IBufferManager {
 protected:
  IBufferManager(wgpu::Device device) : device_(device) {}

 public:
  virtual ~IBufferManager() = default;
  virtual void Upload(void* src, WGPUBuffer dst, size_t size) = 0;
  virtual void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) = 0;
  virtual wgpu::Buffer Create(size_t size, wgpu::BufferUsage usage) = 0;
  virtual void Release(WGPUBuffer buffer) = 0;
  virtual wgpu::Future Download(WGPUBuffer src, void* dst, size_t size) = 0;
  virtual void RefreshPendingBuffers() = 0;

 protected:
  wgpu::Device device_;
};

class BufferManager : public IBufferManager {
 public:
  BufferManager(wgpu::Device device) : IBufferManager(device) {}

  void Upload(void* src, WGPUBuffer dst, size_t size) override;
  void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) override;
  wgpu::Buffer Create(size_t size, wgpu::BufferUsage usage) override;
  void Release(WGPUBuffer buffer) override;
  wgpu::Future Download(WGPUBuffer src, void* dst, size_t size) override;
  void RefreshPendingBuffers() override;

 private:
  struct PendingBuffer {
    wgpu::Buffer buffer;
    void* data;
    size_t size;
  };

  std::vector<PendingBuffer> pending_buffers_;
};

}  // namespace webgpu
}  // namespace onnxruntime
