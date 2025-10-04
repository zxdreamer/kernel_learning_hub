#include "common.h"
#include <cerrno>
#include <sys/stat.h>

namespace samplesCommon {

Logger gLogger{};
LogStream gLogInfo{ std::cout };
LogStream gLogWarning{ std::cout };
LogStream gLogError{ std::cerr };

static bool fileExists(const std::string& p) {
  struct stat st {};
  return ::stat(p.c_str(), &st) == 0 && (st.st_mode & S_IFREG);
}

std::string locateFile(const std::string& name, const std::vector<std::string>& dirs) {
  if (fileExists(name))
    return name;

  for (auto const& d : dirs) {
    std::string sep = (!d.empty() && (d.back() == '/' || d.back() == '\\')) ? "" : "/";
    std::string path = d + sep + name;
    if (fileExists(path))
      return path;
  }
  return name;
}

static void skipSpacesAndComments(std::istream& is) {
  while (true) {
    int c = is.peek();
    if (c == '#') {
      std::string dummy;
      std::getline(is, dummy);
      continue;
    }
    if (std::isspace(c)) {
      is.get();
      continue;
    }
    break;
  }
}

bool readPGMFile(const std::string& path, uint8_t* data, int h, int w) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    gLogError << "Failed to open PGM: " << path << std::endl;
    return false;
  }

  std::string magic;
  ifs >> magic;
  if (magic != "P5" && magic != "P2") {
    gLogError << "Unsupported PGM magic: " << magic << " (expect P5 or P2)" << std::endl;
    return false;
  }

  skipSpacesAndComments(ifs);
  int W = 0, H = 0, maxval = 255;
  ifs >> W;
  skipSpacesAndComments(ifs);
  ifs >> H;
  skipSpacesAndComments(ifs);
  ifs >> maxval;
  if (W != w || H != h) {
    gLogWarning << "PGM size mismatch. Expect (" << w << "," << h << ") but got (" << W << "," << H
                << "). "
                << "Will read and crop/clip as needed." << std::endl;
  }
  if (maxval <= 0 || maxval > 65535) {
    gLogError << "Invalid maxval in PGM: " << maxval << std::endl;
    return false;
  }

  ifs.get();

  const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::vector<uint8_t> tmp;
  tmp.resize(static_cast<size_t>(W) * static_cast<size_t>(H));

  auto scaleToByte = [&](uint32_t v) -> uint8_t {
    if (maxval <= 255)
      return static_cast<uint8_t>(v);
    return static_cast<uint8_t>((v * 255u) / static_cast<uint32_t>(maxval));
  };

  if (magic == "P5") {
    if (maxval <= 255) {
      ifs.read(reinterpret_cast<char*>(tmp.data()), tmp.size());
    } else {
      // 16-bit big-endian per PGM spec
      for (size_t i = 0; i < tmp.size(); ++i) {
        unsigned char bytes[2];
        ifs.read(reinterpret_cast<char*>(bytes), 2);
        uint16_t v = (static_cast<uint16_t>(bytes[0]) << 8) | bytes[1];
        tmp[i] = scaleToByte(v);
      }
    }
  } else { // P2 ASCII
    for (size_t i = 0; i < tmp.size(); ++i) {
      int v = 0;
      ifs >> v;
      tmp[i] = scaleToByte(static_cast<uint32_t>(v));
    }
  }

  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      int srcX = std::min(std::max(xx, 0), W - 1);
      int srcY = std::min(std::max(yy, 0), H - 1);
      data[yy * w + xx] = tmp[srcY * W + srcX];
    }
  }
  return true;
}

} // namespace samplesCommon
