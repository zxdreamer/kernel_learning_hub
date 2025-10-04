#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "nonZeroKernel.h"
#include "sampleNonZeroPlugin.h"

#include <cstring>
#include <memory>

// namespace nvinfer1 {
// namespace plugin {
using namespace nvinfer1;
using half = __half;

void nonZeroIndicesHelper(nvinfer1::DataType type,
                          void const* X,
                          void* output,
                          void* count,
                          void const* K,
                          int32_t R,
                          int32_t C,
                          bool rowOrder,
                          cudaStream_t stream) {
  if (type == nvinfer1::DataType::kFLOAT) {
    nonZeroIndicesImpl<float>(static_cast<float const*>(X),
                              static_cast<int32_t*>(output),
                              static_cast<int32_t*>(count),
                              static_cast<int32_t const*>(K),
                              R,
                              C,
                              rowOrder,
                              stream);
  } else if (type == nvinfer1::DataType::kHALF) {
    nonZeroIndicesImpl<half>(static_cast<half const*>(X),
                             static_cast<int32_t*>(output),
                             static_cast<int32_t*>(count),
                             static_cast<int32_t const*>(K),
                             R,
                             C,
                             rowOrder,
                             stream);
  } else {
    samplesCommon::gLogError << "Unsupported data type" << std::endl;
    return;
  }
}

NonZeroPlugin::NonZeroPlugin(bool rowOrder) : mRowOrder(rowOrder) {
  initFieldsToSerialize();
}

void NonZeroPlugin::initFieldsToSerialize() {
  mDataToSerialize.clear();
  mDataToSerialize.emplace_back(PluginField("rowOrder", &mRowOrder, PluginFieldType::kINT32, 1));
  mFCToSerialize.nbFields = mDataToSerialize.size();
  mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* NonZeroPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime*>(this);
    }
    ASSERT_TRUE(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore*>(this);
  } catch (std::exception const& e) {
    samplesCommon::gLogError << e.what() << std::endl;
  }
  return nullptr;
}

IPluginV3* NonZeroPlugin::clone() noexcept {
  auto clone = std::make_unique<NonZeroPlugin>(*this);
  clone->initFieldsToSerialize();
  return clone.release(); // 释放clone的控制权，返回指针，并且置clone为空
}

char const* NonZeroPlugin::getPluginName() const noexcept {
  return "NonZeroPlugin";
}

char const* NonZeroPlugin::getPluginVersion() const noexcept {
  return "1";
}

char const* NonZeroPlugin::getPluginNamespace() const noexcept {
  return "";
}

int32_t NonZeroPlugin::getNbOutputs() const noexcept {
  return 2;
}

int32_t NonZeroPlugin::configurePlugin(DynamicPluginTensorDesc const* in,
                                       int32_t nbInputs,
                                       DynamicPluginTensorDesc const* out,
                                       int32_t nbOutputs) noexcept {
  return 0;
}

bool NonZeroPlugin::supportsFormatCombination(int32_t pos,
                                              DynamicPluginTensorDesc const* inOut,
                                              int32_t,
                                              int32_t) noexcept {
  bool typeOk = false;
  if (pos == 0)
    typeOk = (inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF);
  else if (pos == 1)
    typeOk = (inOut[1].desc.type == DataType::kINT32);
  else
    typeOk = (inOut[2].desc.type == DataType::kINT32);
  samplesCommon::gLogInfo << "complate supportsFormatCombination...." << std::endl;
  return inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR && typeOk;
}

int32_t NonZeroPlugin::getOutputDataTypes(DataType* outputTypes,
                                          int32_t nbOutputs,
                                          DataType const* inputTypes,
                                          int32_t nbInputs) const noexcept {
  outputTypes[0] = DataType::kINT32;
  outputTypes[1] = DataType::kINT32;
  return 0;
}

int32_t NonZeroPlugin::getOutputShapes(DimsExprs const* inputs,
                                       int32_t nbInputs,
                                       DimsExprs const* shapeInputs,
                                       int32_t nbShapeInputs,
                                       DimsExprs* outputs,
                                       int32_t nbOutputs,
                                       IExprBuilder& exprBuilder) noexcept {
  if (inputs[0].nbDims != 2) {
    return -1;
  }

  outputs[0].nbDims = 2;

  auto upperBound =
      exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[0], *inputs[0].d[1]);
  auto optValue =
      exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *upperBound, *exprBuilder.constant(2));

  // 当某个输出维度（如K）不是输入维度的确定函数，需要等运行时算出来时，你就在
  // getOutputShapes() 里用 declareSizeTensor(...)
  // 声明这个“将由运行期产生的尺寸”
  // 参数从左到右： outputindex,opt,upper
  auto numNonZeroSizeTensor = exprBuilder.declareSizeTensor(1, *optValue, *upperBound);

  if (!mRowOrder) {
    outputs[0].d[0] = exprBuilder.constant(2);
    outputs[0].d[1] = numNonZeroSizeTensor;
  } else {
    outputs[0].d[0] = numNonZeroSizeTensor;
    outputs[0].d[1] = exprBuilder.constant(2);
  }
  outputs[1].nbDims = 0;
  samplesCommon::gLogInfo << "complate getOutputShapes...." << std::endl;
  return 0;
}

// 实际的推理函数，准备输入数据，调用kernel，获得结果
int32_t NonZeroPlugin::enqueue(PluginTensorDesc const* inputDesc,
                               PluginTensorDesc const* outputDesc,
                               void const* const* inputs,
                               void* const* outputs,
                               void* workspace,
                               cudaStream_t stream) noexcept {
  // 1. 数据检查
  int32_t const R = inputDesc[0].dims.d[0];
  int32_t const C = inputDesc[0].dims.d[1];

  auto type = inputDesc[0].type;

  if (!(type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT)) {
    samplesCommon::gLogError << "Unsupported: Sample only supports DataType::kHALF "
                                "and DataType::FLOAT"
                             << std::endl;
    return -1;
  }
  // 2. reset
  cudaMemsetAsync(outputs[1], 0, sizeof(int32_t), stream);

  if (workspace == nullptr) {
    samplesCommon::gLogError << "Unsupported: workspace is null" << std::endl;
    return -1;
  }
  // 3. 调用kernel，填充结果
  if (!mRowOrder) {
    // When constructing a column major output, the kernel needs to be aware
    // of the total number of non-zero elements so as to write the non-zero
    // output at the correct places. Therefore, we will launch the kernel
    // twice: first, only to calculate the total non-zero count, which will be
    // stored in workspace; and then to actually write the non-zero output to
    // the outputs[0] buffer.
    cudaMemsetAsync(workspace, 0, sizeof(int32_t), stream);
    nonZeroIndicesHelper(type, inputs[0], nullptr, workspace, 0, R, C, mRowOrder, stream);
    nonZeroIndicesHelper(
        type, inputs[0], outputs[0], outputs[1], workspace, R, C, mRowOrder, stream);
  } else {
    nonZeroIndicesHelper(type, inputs[0], outputs[0], outputs[1], 0, R, C, mRowOrder, stream);
  }
  samplesCommon::gLogInfo << "complate enqueue...." << std::endl;
  return 0;
}

// 做运行就绪回调，可以拿到实际的shape/type等
int32_t NonZeroPlugin::onShapeChange(PluginTensorDesc const* in,
                                     int32_t nbInputs,
                                     PluginTensorDesc const* out,
                                     int32_t nbOutputs) noexcept {
  return 0;
}

// 拿到NonZeroPlugin插件的全部上下文信息
IPluginV3* NonZeroPlugin::attachToContext(IPluginResourceContext* context) noexcept {
  return clone();
}
// 用于保存插件NonZeroPlugin的自己的数据，即序列化。在NonZeroPluginCreator的函数createPlugin中反序列化出来
PluginFieldCollection const* NonZeroPlugin::getFieldsToSerialize() noexcept {
  return &mFCToSerialize;
}
// 获取插件运行时临时的workspace，以防止频繁申请和释放
// 在enqueue中，会传入这块地址
size_t NonZeroPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
                                       int32_t nbInputs,
                                       DynamicPluginTensorDesc const* outputs,
                                       int32_t nbOutputs) const noexcept {
  return sizeof(int32_t);
}

NonZeroPluginCreator::NonZeroPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(PluginField("rowOrder", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

// getPluginName/getPluginNamespace/getPluginVersion返回插件的身份信息，必须与IPluginV3OnceCore完全一致的元信息
char const* NonZeroPluginCreator::getPluginName() const noexcept {
  return "NonZeroPlugin";
}
char const* NonZeroPluginCreator::getPluginNamespace() const noexcept {
  return "";
}
char const* NonZeroPluginCreator::getPluginVersion() const noexcept {
  return "1";
}

PluginFieldCollection const* NonZeroPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

// 获得插件的PluginFieldCollection，重新创建插件对象
IPluginV3* NonZeroPluginCreator::createPlugin(char const* name,
                                              PluginFieldCollection const* fc,
                                              TensorRTPhase phase) noexcept {
  try {
    bool rowOrder{ true };
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      auto const fieldName(fc->fields[i].name);
      if (std::strcmp(fieldName, "rowOrder") == 0) {
        rowOrder = *static_cast<bool const*>(fc->fields[i].data);
      }
    }
    return new NonZeroPlugin(rowOrder);
  } catch (std::exception const& e) {
    samplesCommon::gLogError << e.what() << std::endl;
  }
  return nullptr;
}
// void NonZeroPluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
//   ASSERT_TRUE(libNamespace != nullptr);
//   mNamespace = libNamespace;
// }
REGISTER_TENSORRT_PLUGIN(NonZeroPluginCreator);
// } // namespace plugin
// } // namespace nvinfer1
