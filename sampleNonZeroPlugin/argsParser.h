// argsParser.h
#pragma once

#include "NvInferPlugin.h"
#include <cstring>
#include <string>
#include <vector>

namespace samplesCommon {

struct Args {
  bool help{ false };
  bool runInFp16{ false };
  bool rowOrder{ true };
  std::vector<std::string> dataDirs;
  std::string engineFile;
};

// 支持：
//   -h / --help
//   --fp16
//   --columnOrder       （把 rowOrder 置为 false）
//   -d <dir> / --datadir=<dir>  （可多次）
//   --engine=<path>
inline bool parseArgs(Args& a, int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    if (s == "-h" || s == "--help") {
      a.help = true;
      return true;
    } else if (s == "--fp16") {
      a.runInFp16 = true;
    } else if (s == "--columnOrder") {
      a.rowOrder = false;
    } else if (s == "-d") {
      if (i + 1 < argc)
        a.dataDirs.emplace_back(argv[++i]);
      else
        return false;
    } else if (s.rfind("--datadir=", 0) == 0) {
      a.dataDirs.emplace_back(s.substr(strlen("--datadir=")));
    } else if (s.rfind("--engine=", 0) == 0) {
      a.engineFile = s.substr(strlen("--engine="));
    } else {
    }
  }
  return true;
}

} // namespace samplesCommon
