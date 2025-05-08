#ifndef DCNV4_PLUGIN_H
#define DCNV4_PLUGIN_H

#include "NvInfer.h"
#include <string>

class DCNv4Plugin : public nvinfer1::IPluginV2DynamicExt {
public:

  DCNv4Plugin(int kernel_h, int kernel_w,
              int stride_h,  int stride_w,
              int pad_h,     int pad_w,
              int dilation_h,int dilation_w,
              int groups,    int group_channels,
              float offset_scale,
              int im2col_step,
              bool remove_center,
              int d_stride,
              int block_thread,
              bool softmax);

  DCNv4Plugin(const void* data, size_t length);

  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::DimsExprs getOutputDimensions(int           index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int                         nbInputs,
                                          nvinfer1::IExprBuilder&     expr) noexcept override;

  bool supportsFormatCombination(int                               pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int                               nbInputs,
                                 int                               nbOutputs) noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int                               nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int                               nbOutputs) const noexcept override { return 0; }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const*                inputs,
              void* const*                      outputs,
              void*                             workspace,
              cudaStream_t                      stream) noexcept override;

  size_t getSerializationSize() const noexcept override;
  void   serialize(void* buffer) const noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int                                     nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int                                     nbOutputs) noexcept override {}

  const char* getPluginType()    const noexcept override { return "DCNv4"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  void        destroy()          noexcept override { delete this; }
  void        setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
  const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

  nvinfer1::DataType getOutputDataType(int                              index,
                                       const nvinfer1::DataType*       inputTypes,
                                       int                              nbInputs) const noexcept override {
    return inputTypes[0];
  }

private:
  int   kernel_h_, kernel_w_;
  int   stride_h_, stride_w_;
  int   pad_h_, pad_w_;
  int   dilation_h_, dilation_w_;
  int   groups_, group_channels_;
  float offset_scale_;
  int   im2col_step_;
  bool  remove_center_;
  int   d_stride_, block_thread_;
  bool  softmax_;
  std::string mNamespace;
};

#endif
