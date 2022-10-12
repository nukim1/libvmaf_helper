#include "vmaf_helper.h"

#include <exception>
#include <sstream>
#include <iostream>
#include <memory.h>

static int ChromaWidth(int lumaWidth, VmafPixelFormat format)
{
  if(format == VMAF_PIX_FMT_YUV444P)
    return lumaWidth;
  else if(format == VMAF_PIX_FMT_YUV420P || format == VMAF_PIX_FMT_YUV422P) 
    return (lumaWidth + 1) >> 1;
  else
    return 0;
}

static int ChromaHeight(int lumaHeight, VmafPixelFormat format)
{
  if(format == VMAF_PIX_FMT_YUV444P || format == VMAF_PIX_FMT_YUV422P)
    return lumaHeight;
  else if(format == VMAF_PIX_FMT_YUV420P)
    return (lumaHeight + 1) >> 1;
  else  
    return 0;
}


void VmafHelper::Create(const VmafHelperConfig& cfg)
{
  Destroy();

  _vmafConfig = {};
  _vmafConfig.log_level = VMAF_LOG_LEVEL_ERROR;
  _vmafConfig.n_threads = cfg.numThreads;
  
  if(vmaf_init(&_vmafContext, _vmafConfig) != 0)
  {
    throw std::runtime_error("vmaf_init() failed!\n");
  }

  if (vmaf_model_load_from_path(&_vmafModel, &_vmafModelConfig, cfg.vmafModelPath) != 0)
  {
    std::ostringstream err;
    err << "Unable to load vmaf model file: " << cfg.vmafModelPath << std::endl;
    throw std::runtime_error(err.str());
  }

  if (vmaf_use_features_from_model(_vmafContext, _vmafModel) != 0)
  {
    throw std::runtime_error("problem loading feature extractors from model!\n");
  }

  const char *vFeatureNames[] = {"psnr", "psnr_hvs", "float_ssim", "float_ms_ssim", "ciede", "cambi"};
  
  for(int i = 0; i < sizeof(vFeatureNames)/sizeof(*vFeatureNames); i++)
  {
    if((cfg.metricFlag & (1 << i)) == 0)
      continue;
    
    if(vmaf_use_feature(_vmafContext, vFeatureNames[i], nullptr) != 0)
    {
      std::cerr << "problem loading feature extractor: " << vFeatureNames[i] << std::endl;
      continue;
    }
     
    VmafMetricFlag metricFlag = VmafMetricFlag(1 << i);
    if(metricFlag == MT_FLAG_PSNR)
    {
      _extraFeatures.emplace_back(std::string{vFeatureNames[i]} + "_y");
      _extraFeatures.emplace_back(std::string{vFeatureNames[i]} + "_cb");
      _extraFeatures.emplace_back(std::string{vFeatureNames[i]} + "_cr");
    }
    else 
      _extraFeatures.emplace_back(std::string{vFeatureNames[i]});
  }

  _frameCount = 0;
  _thisConfig = cfg;
}

void VmafHelper::Destroy()
{
  if (_vmafModel)
  {
    vmaf_model_destroy(_vmafModel);
    _vmafModel = nullptr;
  }
  if (_vmafContext)
  {
    vmaf_close(_vmafContext);
    _vmafContext = nullptr;
  }
}

void VmafHelper::PutFrame(const uint8_t* refPtr, const uint8_t* distPtr)
{
  VmafPicture refPic;
  VmafPicture distPic;

  if (vmaf_picture_alloc(&refPic, _thisConfig.pixelFormat, _thisConfig.bitDepth, _thisConfig.picWidth, _thisConfig.picHeight) != 0)
  {
    std::cerr << "problem occured when allocating picture\n";
    return;
  }
  if (vmaf_picture_alloc(&distPic, _thisConfig.pixelFormat, _thisConfig.bitDepth, _thisConfig.picWidth, _thisConfig.picHeight) != 0)
  {
    std::cerr << "problem occured when allocating picture\n";
    return;
  }

  auto pixFmt = _thisConfig.pixelFormat;
  int w[3], h[3];
  ptrdiff_t stride[3];

  w[0] = _thisConfig.picWidth;
  h[0] = _thisConfig.picHeight;
  w[1] = w[2] = ChromaWidth(w[0], pixFmt);
  h[1] = h[2] = ChromaHeight(h[0], pixFmt);
  stride[0] = w[0] * ((_thisConfig.bitDepth + 7) >> 3);
  stride[1] = stride[2] = w[1] * ((_thisConfig.bitDepth + 7) >> 3);
  
  for(int i = 0; i < 3; i++)
  {
    for (int y = 0; y < h[i]; y++)
    {
      memcpy((uint8_t*)refPic.data[i] + refPic.stride[i] * y, refPtr, stride[i]);
      memcpy((uint8_t*)distPic.data[i] + distPic.stride[i] * y, distPtr, stride[i]);
      refPtr += stride[i];
      distPtr += stride[i];
    }
  }

  if (vmaf_read_pictures(_vmafContext, &refPic, &distPic, _frameCount++) != 0)
  {
    std::cerr << "vmaf_read_pictures() failed!\n";
    return;
  } 
}

void VmafHelper::Flush()
{
  if (vmaf_read_pictures(_vmafContext, nullptr, nullptr, 0) != 0)
  {
    std::cerr << "vmaf_read_pictures() flush failed!\n";
    return;
  }
}

std::vector<VmafHelperResult> VmafHelper::GetResult()
{
  std::vector<VmafHelperResult> dst;

  Flush();
  
  dst.resize(1 + _extraFeatures.size());
  
  for(auto &elem : dst)
    elem.scorePerFrame.resize(_frameCount);

  dst[0].metricName = "vmaf";
  if (vmaf_score_pooled(_vmafContext, _vmafModel, VMAF_POOL_METHOD_MEAN, &dst[0].meanScore, 0, _frameCount - 1) != 0)
  {
    std::cerr << "vmaf_score_pooled() failed!\n";
  }
  if (vmaf_score_pooled(_vmafContext, _vmafModel, VMAF_POOL_METHOD_HARMONIC_MEAN, &dst[0].harmonicMeanScore, 0, _frameCount - 1) != 0)
  {
    std::cerr << "vmaf_score_pooled() failed!\n";
  }
  if (vmaf_score_pooled(_vmafContext, _vmafModel, VMAF_POOL_METHOD_MIN, &dst[0].minScore, 0, _frameCount - 1) != 0)
  {
    std::cerr << "vmaf_score_pooled() failed!\n";
  }
  if (vmaf_score_pooled(_vmafContext, _vmafModel, VMAF_POOL_METHOD_MAX, &dst[0].maxScore, 0, _frameCount - 1) != 0)
  {
    std::cerr << "vmaf_score_pooled() failed!\n";
  }

  for(int j = 0 ; j < _frameCount; j++)
  {
    if(vmaf_score_at_index(_vmafContext, _vmafModel, &dst[0].scorePerFrame[j], j) != 0)
      std::cerr << "vmaf_score_at_index() failed!\n";
  }

  for(int i = 1; i <= _extraFeatures.size(); i++)
  {
    dst[i].metricName = _extraFeatures[i-1];

    if (vmaf_feature_score_pooled(_vmafContext, dst[i].metricName.c_str(), VMAF_POOL_METHOD_MEAN, &dst[i].meanScore, 0, _frameCount - 1) != 0)
      std::cerr << "vmaf_feature_score_pooled() failed!\n";

    if (vmaf_feature_score_pooled(_vmafContext, dst[i].metricName.c_str(), VMAF_POOL_METHOD_HARMONIC_MEAN, &dst[i].harmonicMeanScore, 0, _frameCount - 1) != 0)
      std::cerr << "vmaf_feature_score_pooled() failed!\n";

    if (vmaf_feature_score_pooled(_vmafContext, dst[i].metricName.c_str(), VMAF_POOL_METHOD_MIN, &dst[i].minScore, 0, _frameCount - 1) != 0)
      std::cerr << "vmaf_feature_score_pooled() failed!\n";

    if (vmaf_feature_score_pooled(_vmafContext, dst[i].metricName.c_str(), VMAF_POOL_METHOD_MAX, &dst[i].maxScore, 0, _frameCount - 1) != 0)
      std::cerr << "vmaf_feature_score_pooled() failed!\n";

    for (int j = 0; j < _frameCount; j++)
    {
      if (vmaf_feature_score_at_index(_vmafContext, dst[i].metricName.c_str(), &dst[i].scorePerFrame[j], j) != 0)
        std::cerr << "vmaf_feature_score_at_index() failed!\n";
    }
  }

  return dst;
}
