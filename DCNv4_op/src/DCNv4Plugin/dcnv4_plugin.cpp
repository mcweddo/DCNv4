#include "dcnv4_plugin.h"
#include "dcnv4_cuda.h"
#include <cuda_runtime_api.h>
#include <cstring>

DCNv4Plugin::DCNv4Plugin(int kh, int kw,
                         int sh, int sw,
                         int ph, int pw,
                         int dh, int dw,
                         int g,  int gc,
                         float os,
                         int ics,
                         bool rc,
                         int ds, int bt,
                         bool sm)
    : kernel_h_(kh), kernel_w_(kw),
      stride_h_(sh), stride_w_(sw),
      pad_h_(ph), pad_w_(pw),
      dilation_h_(dh), dilation_w_(dw),
      groups_(g), group_channels_(gc),
      offset_scale_(os),
      im2col_step_(ics),
      remove_center_(rc),
      d_stride_(ds),
      block_thread_(bt),
      softmax_(sm) {}

DCNv4Plugin::DCNv4Plugin(const void* data, size_t length) {
  // simply memcpy back fields in the same order as serialize()
  std::memcpy(this, data, length);
}

nvinfer1::DimsExprs DCNv4Plugin::getOutputDimensions(int index,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& expr) noexcept {
  // input dims: [N, C, H, W]
  // output dims: same N, C, H, W (deform conv preserves spatial dims)
  nvinfer1::DimsExprs out(inputs[0]);
  return out;
}

bool DCNv4Plugin::supportsFormatCombination(int pos,
    const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  // only linear format, float or half
  auto const& desc = inOut[pos];
  return desc.format == nvinfer1::TensorFormat::kLINEAR &&
         (desc.type == nvinfer1::DataType::kFLOAT ||
          desc.type == nvinfer1::DataType::kHALF);
}

int DCNv4Plugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

  // input order: [value, p_offset]  (DCNv4 packs both offsets+mask)
  // but the Python wrapper splits weight/bias elsewhere; in our plugin we
  // assume the ONNX node packs them in the same order you emitted them.

  const at::Half*   maybe_half = nullptr;  // example for FP16
  const float*      value   = reinterpret_cast<const float*>(inputs[0]);
  const float*      offset  = reinterpret_cast<const float*>(inputs[1]);
  float*            out     = reinterpret_cast<float*>(outputs[0]);

  // plugin fields:
  int kh = kernel_h_, kw = kernel_w_;
  int sh = stride_h_, sw = stride_w_;
  int ph = pad_h_,    pw = pad_w_;
  int dh = dilation_h_, dw = dilation_w_;
  int g  = groups_,    gc = group_channels_;
  float os = offset_scale_;
  int ics = im2col_step_;
  int rc  = remove_center_ ? 1 : 0;
  int ds  = d_stride_, bt = block_thread_;
  bool sm = softmax_;

  // **Forward** wrapper from DCNv4_op/src/cuda/dcnv4_cuda.cu
  dcnv4_cuda_forward(
    /* value        */ value,
    /* p_offset     */ offset,
    /* kernel_h, w  */ kh, kw,
    /* stride_h, w  */ sh, sw,
    /* pad_h,   w   */ ph, pw,
    /* dilation_h,w */ dh, dw,
    /* group        */ g,
    /* group_ch     */ gc,
    /* offset_scale */ os,
    /* im2col_step  */ ics,
    /* remove_center*/ rc,
    /* d_stride     */ ds,
    /* block_thread */ bt,
    /* softmax      */ sm
  );

  return cudaGetLastError() != cudaSuccess;
}

size_t DCNv4Plugin::getSerializationSize() const noexcept {
  return sizeof(*this);
}

void DCNv4Plugin::serialize(void* buffer) const noexcept {
  std::memcpy(buffer, this, sizeof(*this));
}

nvinfer1::IPluginV2DynamicExt* DCNv4Plugin::clone() const noexcept {
  auto* p = new DCNv4Plugin(kernel_h_, kernel_w_,
                            stride_h_, stride_w_,
                            pad_h_, pad_w_,
                            dilation_h_, dilation_w_,
                            groups_, group_channels_,
                            offset_scale_,
                            im2col_step_,
                            remove_center_,
                            d_stride_, block_thread_,
                            softmax_);
  p->setPluginNamespace(getPluginNamespace());
  return p;
}
