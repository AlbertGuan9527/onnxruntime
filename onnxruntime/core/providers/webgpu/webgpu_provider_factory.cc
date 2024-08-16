// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory(const webgpu::WebGpuContext& context, const ProviderOptions& provider_options, const SessionOptions* session_options)
      : context_{context},
        info_{provider_options},
        session_options_(session_options) {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(context_, info_, session_options_);
  }

 private:
  WebGpuExecutionProviderInfo info_;
  const SessionOptions* session_options_;
  const webgpu::WebGpuContext& context_;
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(
    const ProviderOptions& provider_options, const SessionOptions* session_options) {
  // TODO: pass-in context id from session_options
  auto& context = webgpu::WebGpuContextFactory::GetOrCreateContext(0 /* context id */);
  context.Initialize();

  return std::make_shared<WebGpuProviderFactory>(context, provider_options, session_options);
}

}  // namespace onnxruntime
