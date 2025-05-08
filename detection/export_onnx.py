#!/usr/bin/env python3
# export_flashinternimage_onnx.py

import os, sys, argparse
import torch, onnx, onnxsim
from mmcv import Config
from torch.onnx import register_custom_op_symbolic
from mmdet.apis import init_detector


# def _symbolic_dcnv4(g, value, p_offset,
#                     kh, kw, sh, sw, ph, pw, dh, dw,
#                     group, group_ch, offset_scale, im2col_step,
#                     remove_center, d_stride, block_thread, softmax):
#     return g.op(
#         "custom_ops::DCNv4",
#         value, p_offset,
#         kernel_h_i=kh, kernel_w_i=kw,
#         stride_h_i=sh, stride_w_i=sw,
#         pad_h_i=ph, pad_w_i=pw,
#         dilation_h_i=dh, dilation_w_i=dw,
#         group_i=group, group_channels_i=group_ch,
#         offset_scale_f=offset_scale,
#         im2col_step_i=im2col_step,
#         remove_center_i=remove_center,
#         d_stride_i=d_stride,
#         block_thread_i=block_thread,
#         softmax_i=softmax
#     )
#
# register_custom_op_symbolic("DCNv4.ext::dcnv4_cuda_forward", _symbolic_dcnv4, 16)


parser = argparse.ArgumentParser(
    description="Export FlashInternImage+DCNv4 to ONNX"
)
parser.add_argument("--config",     required=True,  help="MMDet config .py")
parser.add_argument("--checkpoint", required=True,  help="Model .pth")
parser.add_argument("--output-onnx",default="model.onnx",
                    help="ONNX output path")
parser.add_argument("--output-sim", default="model_simpl.onnx",
                    help="Simplified ONNX output path")
parser.add_argument("--batch-size", type=int, default=1,
                    help="Batch size (overridden by config's img_scale if found)")
parser.add_argument("--height",     type=int, default=640,
                    help="Fallback height (if config has no img_scale)")
parser.add_argument("--width",      type=int, default=640,
                    help="Fallback width (if config has no img_scale)")
parser.add_argument("--opset",      type=int, default=16,
                    help="ONNX opset version")
parser.add_argument("--device",     default="cuda:0",
                    help="Export device")
parser.add_argument("--no-simplify",action="store_true",
                    help="Skip onnx-simplifier")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Prepare PYTHONPATH & imports for custom code
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DETECT_DIR   = os.path.join(PROJECT_ROOT, "detection")
DCNV4_DIR    = os.path.join(PROJECT_ROOT, "DCNv4_op", "python")

sys.path[:0] = [DETECT_DIR, DCNV4_DIR]

import DCNv4.ext      # load the PyTorch DCNv4 extension
import mmdet_custom   # register FlashInternImage model

cfg = Config.fromfile(args.config)
model = init_detector(cfg, args.checkpoint, device=args.device)
model.eval()

img_scale = None
for t in cfg.data.test.pipeline:
    if t.get("type") == "MultiScaleFlipAug":
        img_scale = t.get("img_scale", None)
        break

if img_scale is None:
    if args.height is None or args.width is None:
        raise ValueError(
            "No img_scale in config AND --height/--width not provided"
        )
    H, W = args.height, args.width
else:
    if isinstance(img_scale, (list, tuple)) and isinstance(img_scale[0], (list, tuple)):
        W, H = img_scale[0]
    else:
        W, H = img_scale

print(f"Exporting with input size (WxH) = ({W} x {H})")

B = args.batch_size
dummy = torch.randn(B, 3, H, W, device=args.device)

print(f">>> torch.onnx.export → {args.output_onnx}")
torch.onnx.export(
    model, dummy, args.output_onnx,
    opset_version=args.opset,
    input_names=["input"],
    output_names=["detections"],
    dynamic_axes={
        "input":     {0: "batch", 2: "height", 3: "width"},
        "detections":{0: "batch"},
    },
    do_constant_folding=True,
    custom_opsets={"custom_ops": args.opset},
)

if not args.no_simplify:
    print(f">>> Simplifying to {args.output_sim}")
    model_onnx = onnx.load(args.output_onnx)
    model_simp, ok = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=True,
        input_shapes={"input": [B, 3, H, W]}
    )
    if not ok:
        print("ONNX simplifier check failed")
        sys.exit(1)
    onnx.save(model_simp, args.output_sim)
    print("Simplified ONNX saved.")

print("Export finished.")
