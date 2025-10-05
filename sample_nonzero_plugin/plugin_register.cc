#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin_register.h"
#include "sample_nonzero_plugin.h"
#include <mutex>
#include <vector>

using namespace nvinfer1;

class ThreadSafeLoggerFinder {
private:
  ILoggerFinder* mLoggerFinder{ nullptr };
  std::mutex mMutex;

public:
  ThreadSafeLoggerFinder() = default;

  //! Set the logger finder.
  void setLoggerFinder(ILoggerFinder* finder) {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr) {
      mLoggerFinder = finder;
    }
  }

  //! Get the logger.
  ILogger* getLogger() noexcept {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr) {
      return mLoggerFinder->findLogger();
    }
    return nullptr;
  }
};

ThreadSafeLoggerFinder gLoggerFinder;

ILogger* getPluginLogger() {
  return gLoggerFinder.getLogger();
}

extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
  nbCreators = 1;
  static NonZeroPluginCreator nonZeroPluginCreator;
  static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = { &nonZeroPluginCreator };
  return kPLUGIN_CREATOR_LIST;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder) {
  gLoggerFinder.setLoggerFinder(finder);
}
