// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

class DisabledCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return (request_size + 15) / 16 * 16;
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/, wgpu::BufferUsage /*usage*/) override {
    // always return empty buffer
    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/, size_t /*buffer_size*/, wgpu::BufferUsage /*usage*/) override {
    // no-op
  }
  void ReleaseBuffer(WGPUBuffer buffer, size_t /*buffer_size*/, wgpu::BufferUsage /*usage*/) override {
    wgpuBufferDestroy(buffer);
  }

  void OnRefresh() override {
    // no-op
  }
};

class SimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return (request_size + 15) / 16 * 16;
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size, wgpu::BufferUsage usage) override {
    if (usage | wgpu::BufferUsage::Storage) {
      auto it = buffers_.find(buffer_size);
      if (it != buffers_.end() && !it->second.empty()) {
        auto buffer = it->second.back();
        it->second.pop_back();
        return buffer;
      }
    }

    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer buffer, size_t /*request_size*/, size_t buffer_size, wgpu::BufferUsage usage) override {
  }
  void ReleaseBuffer(WGPUBuffer buffer, size_t buffer_size, wgpu::BufferUsage usage) override {
    if (usage | wgpu::BufferUsage::Storage) {
      pending_buffers_.emplace_back(buffer, buffer_size);
    } else {
      wgpuBufferDestroy(buffer);
    }
  }
  void OnRefresh() override {
    for (auto& pair : pending_buffers_) {
      buffers_[pair.second].push_back(pair.first);
    }
    pending_buffers_.clear();
  }

  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<std::pair<WGPUBuffer, size_t>> pending_buffers_;
};

class BucketCacheManager : public IBufferCacheManager {
  static const std::unordered_map<size_t, size_t> kBucketSizes;
  static const std::array<size_t> kBucketSizesArray;

  size_t CalculateBufferSize(size_t request_size) override {
    ORT_NOT_IMPLEMENTED("TODO");
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size, wgpu::BufferUsage usage) override {
    ORT_NOT_IMPLEMENTED("TODO");
  }
  void RegisterBuffer(WGPUBuffer buffer, size_t request_size, size_t buffer_size, wgpu::BufferUsage usage) override {
    ORT_NOT_IMPLEMENTED("TODO");
  }
  void ReleaseBuffer(WGPUBuffer buffer, size_t buffer_size, wgpu::BufferUsage usage) override {
    ORT_NOT_IMPLEMENTED("TODO");
  }
  void OnRefresh() override {
    ORT_NOT_IMPLEMENTED("TODO");
  }
};

constexpr std::initializer_list<std::pair<size_t, size_t>> BUCKET_TABLE = {
    {64, 250},
    {128, 200},
    {256, 200},
    {512, 200},
    {2048, 230},
    {4096, 200},
    {8192, 50},
    {16384, 50},
    {32768, 50},
    {65536, 50},
    {131072, 50},
    {262144, 50},
    {524288, 50},
    {1048576, 50},
    {2097152, 30},
    {4194304, 20},
    {8388608, 10},
    {12582912, 10},
    {16777216, 10},
    {26214400, 15},
    {33554432, 22},
    {44236800, 2},
    {58982400, 6},
    // we don't want to cache the bucket sizes below but not caching them
    // results in some major performance hits for models like sd-turbo.
    {67108864, 6},
    {134217728, 6},
    {167772160, 6},
};
const std::unordered_map<size_t, size_t> BucketCacheManager::kBucketSizes{BUCKET_TABLE};


std::unique_ptr<IBufferCacheManager> CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::None:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

BufferManager::BufferManager(wgpu::Device device, BufferCacheMode cache_mode) : IBufferManager{device, CreateBufferCacheManager(cache_mode)} {
}

void BufferManager::Upload(void* src, WGPUBuffer dst, size_t size) const {
}

void BufferManager::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const {
}

WGPUBuffer BufferManager::Create(size_t size, wgpu::BufferUsage usage) const {
  auto buffer_size = cache_->CalculateBufferSize(size);

  auto buffer = cache_->TryAcquireCachedBuffer(buffer_size, usage);
  if (buffer) {
    return buffer;
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc;
  desc.size = buffer_size;
  desc.usage = usage;
  buffer = device_.CreateBuffer(&desc);

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", usage, ".");

  cache_->RegisterBuffer(buffer, size, buffer_size, usage);
  return buffer;
}

void BufferManager::Release(WGPUBuffer buffer) const {
  cache_->ReleaseBuffer(buffer, 0, wgpu::BufferUsage::None);
}

wgpu::Future BufferManager::Download(WGPUBuffer src, void* dst, size_t size) const {
  return wgpu::Future();
}

void BufferManager::RefreshPendingBuffers() const {
  cache_->OnRefresh();
}

}  // namespace webgpu
}  // namespace onnxruntime
