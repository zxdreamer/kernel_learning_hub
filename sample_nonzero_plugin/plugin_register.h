#pragma once
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

using namespace nvinfer1;

namespace nvinfer1 {
namespace plugin {

ILogger* getPluginLogger();

} // namespace plugin
} // namespace nvinfer1

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
