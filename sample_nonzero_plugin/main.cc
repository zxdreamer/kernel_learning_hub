
#include "NvInfer.h"
#include "argsParser.h"
#include "buffers.h"
#include "build_engine.h"
#include "common.h"
#include <cuda_runtime_api.h>

std::string const kSAMPLE_NAME = "TensorRT.sample_non_zero_plugin";

samplesCommon::NonZeroParams initializeSampleParams(samplesCommon::Args const& args) {
  samplesCommon::NonZeroParams params;
  // Use default directories if user hasn't provided directory paths
  if (args.dataDirs.empty()) {
    params.dataDirs.push_back("data/mnist/");
  } else {
    params.dataDirs = args.dataDirs;
  }

  params.inputTensorNames.push_back("Input");
  params.outputTensorNames.push_back("Output0");
  params.outputTensorNames.push_back("Output1");
  params.fp16 = args.runInFp16;
  params.rowOrder = args.rowOrder;
  params.engineFile = args.engineFile;

  return params;
}

void printHelpInfo() {
  std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or "
               "--datadir=<path to data directory>]"
            << std::endl;
  std::cout << "--help          Display help information" << std::endl;
  std::cout << "--datadir       Specify path to a data directory, overriding "
               "the default. This option can be used "
               "multiple times to add multiple directories. If no data "
               "directories are given, the default is to use "
               "(data/samples/mnist/, data/mnist/)"
            << std::endl;
  std::cout << "--fp16          Run in FP16 mode." << std::endl;
  std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv) {
  samplesCommon::Args args;
  bool argsOK = samplesCommon::parseArgs(args, argc, argv);
  if (!argsOK) {
    std::cout << "Invalid arguments" << std::endl;
    printHelpInfo();
    return EXIT_FAILURE;
  }
  if (args.help) {
    printHelpInfo();
    return EXIT_SUCCESS;
  }

  auto sampleTest = samplesCommon::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

  samplesCommon::gLogger.reportTestStart(sampleTest);

  SampleNonZeroPlugin sample(initializeSampleParams(args));

  if (!sample.build()) {
    return samplesCommon::gLogger.reportFail(sampleTest);
  }
  if (!sample.infer()) {
    return samplesCommon::gLogger.reportFail(sampleTest);
  }

  return samplesCommon::gLogger.reportPass(sampleTest);
}
