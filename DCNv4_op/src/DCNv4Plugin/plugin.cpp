#include "dcnv4_plugin.h"
#include "NvInferPlugin.h"

class DCNv4PluginCreator : public nvinfer1::IPluginCreator {
public:
  const char* getPluginName() const noexcept override   { return "DCNv4"; }
  const char* getPluginVersion() const noexcept override{ return "1"; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return nullptr; }
  nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) noexcept override {
    // Override via ONNX attributes
    return new DCNv4Plugin(3,3,1,1,1,1,1,1,1,1.f,1,false,1,1,false);
  }
  nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, size_t length) noexcept override {
    return new DCNv4Plugin(data, length);
  }
  void setPluginNamespace(const char*) noexcept override {}
  const char* getPluginNamespace() const noexcept override { return ""; }
};

REGISTER_TENSORRT_PLUGIN(DCNv4PluginCreator);
