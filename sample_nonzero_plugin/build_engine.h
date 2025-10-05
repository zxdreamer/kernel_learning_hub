#pragma once

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "nonzero_kernel.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

namespace {
using namespace nvinfer1;
using half = __half;
} // namespace

template <typename T>
using SampleUniquePtr = std::unique_ptr<T>;

class SampleNonZeroPlugin {
public:
  SampleNonZeroPlugin(samplesCommon::NonZeroParams const& params)
      : mParams(params), mRuntime(nullptr), mEngine(nullptr) {
    mSeed = static_cast<uint32_t>(time(nullptr));
  }
  bool build();
  bool infer();

private:
  samplesCommon::NonZeroParams mParams;

  nvinfer1::Dims mInputDims;
  nvinfer1::Dims mOutputDims;

  std::shared_ptr<nvinfer1::IRuntime> mRuntime;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  uint32_t mSeed{};
  bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig>& config);
  bool processInput(samplesCommon::BufferManager const& buffers);
  void dumpOutput(samplesCommon::BufferManager const& buffers) const;
};
