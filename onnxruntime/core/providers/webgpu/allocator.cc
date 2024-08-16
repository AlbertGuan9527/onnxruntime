// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  // GetContext().Device().CreateBuffer()

  //void* p = EM_ASM_PTR({ return Module.jsepAlloc($0); }, size);
  ORT_ENFORCE(false, "not implemented");

  stats_.num_allocs++;
  stats_.bytes_in_use += size;
  return nullptr;
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    //size_t size = (size_t)(void*)EM_ASM_PTR({ return Module.jsepFree($0); }, p);
    ORT_ENFORCE(false, "not implemented");

    //stats_.bytes_in_use -= size;
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

}  // namespace webgpu
}  // namespace onnxruntime
