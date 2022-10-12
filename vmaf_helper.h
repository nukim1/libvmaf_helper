#pragma once

#include <libvmaf/libvmaf.h>
#include <vector>
#include <string>
#include <cstdint>

enum VmafMetricFlag
{
  MT_FLAG_VMAF = 0x00,
  MT_FLAG_PSNR = 0x01,
  MT_FLAG_PSNR_HVS = 0x02,
  MT_FLAG_SSIM = 0x04,
  MT_FLAG_MS_SSIM = 0x08,
  // MT_FLAG_CIEDE = 0x10, too slow
  MT_FLAG_CAMBI = 0x20
};

struct VmafHelperConfig
{
  int picWidth;
  int picHeight;
  int bitDepth;
  VmafPixelFormat pixelFormat;
  int numThreads;
  const char *vmafModelPath;
  int metricFlag;
};

struct VmafHelperResult
{
  std::string metricName;
  std::vector<double> scorePerFrame;
  double meanScore;
  double harmonicMeanScore;
  double minScore;
  double maxScore;
};

class VmafHelper
{
public:
  VmafHelper() = default;
  VmafHelper(const VmafHelperConfig& cfg) { Create(cfg); }
  ~VmafHelper() { Destroy(); }

  void Create(const VmafHelperConfig& cfg);
  void Destroy();
  void PutFrame(const uint8_t* refPtr, const uint8_t* distPtr);
  void Flush();
  std::vector<VmafHelperResult> GetResult();

private:
  VmafConfiguration _vmafConfig;
  VmafContext *_vmafContext{nullptr};
  VmafModel *_vmafModel{nullptr};
  VmafModelConfig _vmafModelConfig;
  
  VmafHelperConfig _thisConfig;
  std::vector<std::string> _extraFeatures;
  uint32_t _frameCount;
};
