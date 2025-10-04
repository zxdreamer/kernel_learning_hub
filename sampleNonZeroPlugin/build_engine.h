#pragma once

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "nonZeroKernel.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

// namespace nvinfer1 {
// namespace plugin {
namespace {
using namespace nvinfer1;
using half = __half;
} // namespace

template <typename T>
using SampleUniquePtr = std::unique_ptr<T>;

struct NonZeroParams {
  int32_t batchSize{ 1 };
  int32_t dlaCore{ -1 };
  bool int8{ false };
  bool fp16{ false };
  bool bf16{ false };
  std::vector<std::string> dataDirs;
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
  std::string timingCacheFile;
  bool rowOrder{ true };
  std::string engineFile{};
};

class SampleNonZeroPlugin {
public:
  SampleNonZeroPlugin(NonZeroParams const& params)
      : mParams(params), mRuntime(nullptr), mEngine(nullptr) {
    mSeed = static_cast<uint32_t>(time(nullptr));
  }
  bool build();
  bool infer();

private:
  NonZeroParams mParams;

  nvinfer1::Dims mInputDims;
  nvinfer1::Dims mOutputDims;

  std::shared_ptr<nvinfer1::IRuntime> mRuntime;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  uint32_t mSeed{};
  bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig>& config);
  bool processInput(samplesCommon::BufferManager const& buffers);
  bool verifyOutput(samplesCommon::BufferManager const& buffers);
};
// } // namespace plugin
// } // namespace nvinfer1
