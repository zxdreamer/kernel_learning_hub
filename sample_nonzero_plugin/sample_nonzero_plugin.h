#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "nonzero_kernel.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>

void nonZeroIndicesHelper(nvinfer1::DataType type,
                          void const* X,
                          void* output,
                          void* count,
                          void const* K,
                          int32_t R,
                          int32_t C,
                          bool rowOrder,
                          cudaStream_t stream);

class NonZeroPlugin : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
public:
  NonZeroPlugin(NonZeroPlugin const&) = default;
  explicit NonZeroPlugin(bool rowOrder);

  void initFieldsToSerialize();

  // IPluginV3
  nvinfer1::IPluginCapability*
  getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
  nvinfer1::IPluginV3* clone() noexcept override;

  // IPluginV3OneCore
  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  char const* getPluginNamespace() const noexcept override;

  // IPluginV3OneBuild
  int32_t getNbOutputs() const noexcept override;
  int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                          int32_t nbInputs,
                          nvinfer1::DynamicPluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept override;
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::DynamicPluginTensorDesc const* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;
  int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
                             int32_t nbOutputs,
                             nvinfer1::DataType const* inputTypes,
                             int32_t nbInputs) const noexcept override;
  int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
                          int32_t nbInputs,
                          nvinfer1::DimsExprs const* shapeInputs,
                          int32_t nbShapeInputs,
                          nvinfer1::DimsExprs* outputs,
                          int32_t nbOutputs,
                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  // IPluginV3OneRuntime
  int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                  nvinfer1::PluginTensorDesc const* outputDesc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;
  int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in,
                        int32_t nbInputs,
                        nvinfer1::PluginTensorDesc const* out,
                        int32_t nbOutputs) noexcept override;

  nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
  size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                          int32_t nbInputs,
                          nvinfer1::DynamicPluginTensorDesc const* outputs,
                          int32_t nbOutputs) const noexcept override;

private:
  bool mRowOrder{ true };
  std::vector<nvinfer1::PluginField> mDataToSerialize;
  nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class NonZeroPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
  NonZeroPluginCreator();
  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
  nvinfer1::IPluginV3* createPlugin(char const* name,
                                    nvinfer1::PluginFieldCollection const* fc,
                                    nvinfer1::TensorRTPhase phase) noexcept override;
  char const* getPluginNamespace() const noexcept override;
  // void setPluginNamespace(char const* libNamespace) noexcept override;

private:
  nvinfer1::PluginFieldCollection mFC{};
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};
