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

enum class BufferCacheMode {
  None,
  Simple,
  Bucket
};

class IBufferCacheManager {
 public:
  virtual ~IBufferCacheManager() = default;

  // calculate actual buffer size to allocate based on the requested size.
  virtual size_t CalculateBufferSize(size_t request_size) = 0;

  // return a buffer if available in cache. otherwise empty.
  virtual WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size, wgpu::BufferUsage usage) = 0;

  // register a newly created buffer
  virtual void RegisterBuffer(WGPUBuffer buffer, size_t request_size, size_t buffer_size, wgpu::BufferUsage usage) = 0;

  // release a buffer
  virtual void ReleaseBuffer(WGPUBuffer buffer, size_t buffer_size, wgpu::BufferUsage usage) = 0;

  // when a stream refresh is requested
  virtual void OnRefresh() = 0;
};

class IBufferManager {
 protected:
  IBufferManager(wgpu::Device device, std::unique_ptr<IBufferCacheManager> cache) : device_{device}, cache_{std::move(cache)} {}

 public:
  virtual ~IBufferManager() = default;
  virtual void Upload(void* src, WGPUBuffer dst, size_t size) const = 0;
  virtual void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const = 0;
  virtual WGPUBuffer Create(size_t size, wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst) const = 0;
  virtual void Release(WGPUBuffer buffer) const = 0;
  virtual wgpu::Future Download(WGPUBuffer src, void* dst, size_t size) const = 0;
  virtual void RefreshPendingBuffers() const = 0;

  // TODO: add statistics

 protected:
  wgpu::Device device_;
  std::unique_ptr<IBufferCacheManager> cache_;
};

class BufferManager : public IBufferManager {
 public:
  BufferManager(wgpu::Device device, BufferCacheMode cache_mode);

  void Upload(void* src, WGPUBuffer dst, size_t size) const override;
  void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const override;
  WGPUBuffer Create(size_t size, wgpu::BufferUsage usage) const override;
  void Release(WGPUBuffer buffer) const override;
  wgpu::Future Download(WGPUBuffer src, void* dst, size_t size) const override;
  void RefreshPendingBuffers() const override;

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
