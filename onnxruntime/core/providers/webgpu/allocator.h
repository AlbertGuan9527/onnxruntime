// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class GpuBufferAllocator : public IAllocator {
 public:
  GpuBufferAllocator()
      : IAllocator(
            OrtMemoryInfo(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
                          0, OrtMemTypeDefault)) {
  }

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
};

}  // namespace webgpu
}  // namespace onnxruntime
