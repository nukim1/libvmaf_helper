Very simple libvmaf container class

* usage *

VmafHelper vmaf;
VmafHelperConfig vmafCfg;

vmafCfg.bitDepth = 8;
vmafCfg.metricFlag = MT_FLAG_PSNR; // or 0
vmafCfg.numThreads = 16;
vmafCfg.picWidth = 1920;
vmafCfg.picHeight = 1080;
vmafCfg.pixelFormat = VMAF_PIX_FMT_YUV420P;
vmafCfg.vmafModelPath = "vmaf_v0.6.1.json";

vmaf.Create(vmafCfg);

vmaf.PutFrame(pRefFrame, pDistFrame);
auto result = vmaf.GetResult();
