#pragma once
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace samplesCommon {

// Logger 实现，含 TRT Logger + 简易测试报告
class TRTLogger final : public nvinfer1::ILogger {
public:
  explicit TRTLogger(Severity sev = Severity::kINFO) : mSeverity(sev) {}
  void setReportableSeverity(Severity sev) {
    mSeverity = sev;
  }
  Severity getReportableSeverity() const {
    return mSeverity;
  }

  void log(Severity severity, const char* msg) noexcept override {
    if (severity > mSeverity || msg == nullptr)
      return;
    std::ostream& os = (severity >= Severity::kERROR) ? std::cerr : std::cout;
    os << "[TRT][" << toString(severity) << "] " << msg << std::endl;
  }

private:
  static const char* toString(Severity s) {
    switch (s) {
    case Severity::kINTERNAL_ERROR:
      return "FATAL";
    case Severity::kERROR:
      return "ERROR";
    case Severity::kWARNING:
      return "WARN";
    case Severity::kINFO:
      return "INFO";
    case Severity::kVERBOSE:
      return "VERB";
    default:
      return "LOG";
    }
  }
  Severity mSeverity;
};

class Logger {
public:
  Logger() : mTrtLogger(nvinfer1::ILogger::Severity::kINFO) {}

  nvinfer1::ILogger& getTRTLogger() {
    return mTrtLogger;
  }

  struct TestHandle {
    int id{ 0 };
    std::string name;
  };
  TestHandle defineTest(const std::string& name, int /*argc*/, char* /*argv*/[]) {
    static int s_id = 1;
    TestHandle h;
    h.id = s_id++;
    h.name = name;
    return h;
  }
  void reportTestStart(const TestHandle& h) {
    std::cout << "[TEST] Start: " << h.name << std::endl;
  }
  int reportPass(const TestHandle& h) {
    std::cout << "[TEST] PASS : " << h.name << std::endl;
    return EXIT_SUCCESS;
  }
  int reportFail(const TestHandle& h) {
    std::cerr << "[TEST] FAIL : " << h.name << std::endl;
    return EXIT_FAILURE;
  }

private:
  TRTLogger mTrtLogger;
};

// 全局 logger 实例
extern Logger gLogger;

// 便捷日志流
struct LogStream {
  explicit LogStream(std::ostream& o) : os(&o) {}
  template <typename T>
  LogStream& operator<<(const T& v) {
    (*os) << v;
    return *this;
  }
  LogStream& operator<<(std::ostream& (*pf)(std::ostream&)) {
    (*os) << pf;
    return *this;
  }
  std::ostream* os;
};
extern LogStream gLogInfo;
extern LogStream gLogWarning;
extern LogStream gLogError;

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (!obj)
      return;
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
    // TensorRT 10+：接口都有虚析构，直接 delete
    delete obj;
#else
    obj->destroy();
#endif
  }
};

std::string locateFile(const std::string& name, const std::vector<std::string>& dirs);
bool readPGMFile(const std::string& path, uint8_t* data, int h, int w);
inline bool checkCudaImpl(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    gLogError << "CUDA error " << static_cast<int>(status) << " (" << cudaGetErrorString(status)
              << ") at " << file << ":" << line << std::endl;
    return false;
  }
  return true;
}

} // namespace samplesCommon

#define CHECK(call)                                                                                \
  do {                                                                                             \
    if (!::samplesCommon::checkCudaImpl((call), __FILE__, __LINE__)) {                             \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#define ASSERT_TRUE(cond)                                                                          \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      ::samplesCommon::gLogError << "Assertion failed: " #cond << " at " << __FILE__ << ":"        \
                                 << __LINE__ << std::endl;                                         \
      std::terminate();                                                                            \
    }                                                                                              \
  } while (0)
