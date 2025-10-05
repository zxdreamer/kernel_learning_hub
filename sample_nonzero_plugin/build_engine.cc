#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "build_engine.h"
#include "common.h"
#include "nonzero_kernel.h"
#include "sample_nonzero_plugin.h"
#include <cuda_runtime_api.h>

static std::vector<char> loadFile(std::string const& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return {};
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  f.read(buf.data(), size);
  return buf;
}

bool SampleNonZeroPlugin::build() {
  // 1. 反序列化前注册对应的 Creator
  static NonZeroPluginCreator gNonZeroCreator;
  auto* reg = getPluginRegistry();
  if (!reg->registerCreator(gNonZeroCreator, "")) {
    std::cout << "registerCreator failed" << std::endl;
    return false;
  }
  // 2. 读取engine并反序列化
  auto planBytes = loadFile(mParams.engineFile);
  if (planBytes.empty()) {
    std::cout << "Cannot read engine file: " << mParams.engineFile << std::endl;
    return false;
  }

  mRuntime.reset(nvinfer1::createInferRuntime(samplesCommon::gLogger.getTRTLogger()));
  if (!mRuntime)
    return false;

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      mRuntime->deserializeCudaEngine(planBytes.data(), planBytes.size()),
      samplesCommon::InferDeleter());
  if (!mEngine)
    return false;

  // 3. 记录 I/O 维度与名称
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    auto name = mEngine->getIOTensorName(i);
    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      mInputDims = mEngine->getTensorShape(name);
      if (mParams.inputTensorNames.empty())
        mParams.inputTensorNames = { name };
    } else {
      if (mParams.outputTensorNames.size() < 2) {
        mParams.outputTensorNames.push_back(name);
      }
    }
  }

  return true;
}

bool SampleNonZeroPlugin::infer() {
  // 1. 创建 context
  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
  if (!context) {
    return false;
  }

  // 2. 用引擎的静态维度计算体积
  int32_t const R = mInputDims.d[0];
  int32_t const C = mInputDims.d[1];
  std::vector<int64_t> ioVolumes = { int64_t(R) * C, int64_t(R) * C * 2, 1 };

  samplesCommon::BufferManager buffers(mEngine, ioVolumes);

  // 3. 绑定地址
  for (int i = 0, e = mEngine->getNbIOTensors(); i < e; ++i) {
    auto const name = mEngine->getIOTensorName(i);
    context->setTensorAddress(name, buffers.getDeviceBuffer(name));
  }

  // 4. 准备输入 → 运行 → 拷回输出
  if (!processInput(buffers))
    return false;
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  buffers.copyInputToDeviceAsync(stream);
  bool ok = context->enqueueV3(stream);
  buffers.copyOutputToHostAsync(stream);
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaStreamDestroy(stream));
  if (!ok)
    return false;

  // 5. 输出
  dumpOutput(buffers);
  return true;
}

bool SampleNonZeroPlugin::processInput(samplesCommon::BufferManager const& buffers) {
  int32_t const inputH = mInputDims.d[0];
  int32_t const inputW = mInputDims.d[1];

  std::vector<uint8_t> fileData(inputH * inputW);

  std::default_random_engine generator(mSeed);
  std::uniform_int_distribution<int32_t> distr(0, 9);
  auto const number = distr(generator);
  samplesCommon::readPGMFile(
      samplesCommon::locateFile(std::to_string(number) + ".pgm", mParams.dataDirs),
      fileData.data(),
      inputH,
      inputW);

  float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
  for (int32_t i = 0; i < inputH * inputW; ++i) {
    auto const raw = 1.0 - float(fileData[i] / 255.0);
    hostDataBuffer[i] = raw;
  }

  std::cout << "Input:" << std::endl;
  for (int32_t i = 0; i < inputH; ++i) {
    for (int32_t j = 0; j < inputW; ++j) {
      std::cout << hostDataBuffer[i * inputW + j];
      if (j < inputW - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return true;
}

void SampleNonZeroPlugin::dumpOutput(samplesCommon::BufferManager const& buffers) const {
  auto* out0 =
      static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[0])); // 索引数组
  auto* pCnt =
      static_cast<int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1])); // 非零个数
  if (!out0 || !pCnt) {
    std::cout << "[dumpOutput] output buffer is null" << std::endl;
    return;
  }
  const int32_t count = *pCnt;

  std::cout << "count: " << count << std::endl;
  std::cout << "Output:" << std::endl;

  if (mParams.rowOrder) {
    // 行主序交替：(row0,col0,row1,col1,...)
    for (int32_t i = 0; i < count; ++i) {
      const int32_t r = out0[2 * i];
      const int32_t c = out0[2 * i + 1];
      std::cout << r << " " << c << std::endl;
    }
  } else {
    // 列主序分段：[rows...][cols...]
    const int32_t* rows = out0;
    const int32_t* cols = out0 + count;
    for (int32_t i = 0; i < count; ++i) {
      std::cout << rows[i] << " " << cols[i] << std::endl;
    }
  }
}
