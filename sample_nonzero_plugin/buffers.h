
#pragma once
#include "NvInfer.h"
#include "common.h"
#include <cuda_runtime_api.h>
#include <unordered_map>

namespace samplesCommon {

inline size_t elementSize(nvinfer1::DataType t) {
  using nvinfer1::DataType;
  switch (t) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kINT8:
    return 1;
  case DataType::kINT32:
    return 4;
#if NV_TENSORRT_MAJOR >= 8
  case DataType::kBOOL:
    return 1;
#endif
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 9)
  case DataType::kBF16:
    return 2;
#endif
  default:
    return 0;
  }
}

class BufferManager {
public:
  BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                const std::vector<int64_t>& ioVolumes)
      : mEngine(std::move(engine)) {
    init(ioVolumes);
  }
  ~BufferManager() {
    destroy();
  }

  void* getHostBuffer(const std::string& name) const {
    auto it = mTensors.find(name);
    return (it == mTensors.end()) ? nullptr : it->second.h;
  }
  void* getDeviceBuffer(const char* name) const {
    auto it = mTensors.find(name);
    return (it == mTensors.end()) ? nullptr : it->second.d;
  }
  void* getDeviceBuffer(const std::string& name) const {
    return getDeviceBuffer(name.c_str());
  }

  void copyInputToDeviceAsync(cudaStream_t stream) {
    for (auto& kv : mTensors) {
      auto& t = kv.second;
      if (t.isInput && t.h && t.d && t.bytes) {
        CHECK(cudaMemcpyAsync(t.d, t.h, t.bytes, cudaMemcpyHostToDevice, stream));
      }
    }
  }

  void copyOutputToHostAsync(cudaStream_t stream) {
    for (auto& kv : mTensors) {
      auto& t = kv.second;
      if (!t.isInput && t.h && t.d && t.bytes) {
        CHECK(cudaMemcpyAsync(t.h, t.d, t.bytes, cudaMemcpyDeviceToHost, stream));
      }
    }
  }

private:
  struct TensorMem {
    void* h{ nullptr };
    void* d{ nullptr };
    size_t bytes{ 0 };
    bool isInput{ false };
  };

  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  std::unordered_map<std::string, TensorMem> mTensors;

  void destroy() {
    for (auto& kv : mTensors) {
      if (kv.second.d)
        cudaFree(kv.second.d);
      if (kv.second.h)
        cudaFreeHost(kv.second.h);
    }
    mTensors.clear();
  }

  void init(const std::vector<int64_t>& ioVolumes) {
    const int n = mEngine->getNbIOTensors();
    if (static_cast<int>(ioVolumes.size()) != n) {
      std::cout << "[Buffer] ioVolumes size (" << ioVolumes.size() << ") != IOTensors (" << n
                << "). Will try best-effort." << std::endl;
    }

    for (int i = 0; i < n; ++i) {
      const char* name = mEngine->getIOTensorName(i);
      auto mode = mEngine->getTensorIOMode(name);
      auto dtype = mEngine->getTensorDataType(name);
      size_t esz = elementSize(dtype);
      if (esz == 0) {
        std::cout << "[Buffer] Unsupported DataType: " << name << std::endl;
        continue;
      }

      int64_t vol = 0;
      if (i < static_cast<int>(ioVolumes.size())) {
        vol = ioVolumes[i];
      } else {
        auto d = mEngine->getTensorShape(name);
        vol = 1;
        for (int k = 0; k < d.nbDims; ++k)
          vol *= d.d[k];
      }
      if (vol <= 0) {
        std::cout << "[Buffer] Non-positive volume for " << name << ", set to 1." << std::endl;
        vol = 1;
      }

      TensorMem tm;
      tm.isInput = (mode == nvinfer1::TensorIOMode::kINPUT);
      tm.bytes = static_cast<size_t>(vol) * esz;
      CHECK(cudaMallocHost(&tm.h, tm.bytes));
      CHECK(cudaMalloc(&tm.d, tm.bytes));
      std::memset(tm.h, 0, tm.bytes);
      CHECK(cudaMemset(tm.d, 0, tm.bytes));

      mTensors.emplace(name, tm);
    }
  }
};

} // namespace samplesCommon