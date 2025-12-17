#!/usr/bin/env python3
"""
csrnet_head_count_fix.py

CSRNet inference with:
 - multi-scale inference fusion (fixes near-camera missing heads)
 - optional linear calibration (fixes systematic over/under-count)
 - optional ROI mask
 - FP16 support and GPU memory-safe operations

Usage example:
python csrnet_head_count_fix.py --video ppl.mp4 --weights csrnet_best.pth --output out_fixed.mp4 \
  --csv counts_fixed.csv --scales 1.0 1.5 --calib calib_gt.csv --roi_mask audience_mask.png --fp16
"""
import os, time, argparse, csv
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ----------------------------
# CSRNet model (same as before)
# ----------------------------
class CSRNet(nn.Module):
    def __init__(self, load_vgg_weights=True):
        super().__init__()
        self.frontend = self._make_layers([64,64,'M',128,128,'M',256,256,256,'M',512,512,512], in_ch=3)
        self.backend = self._make_layers([512,512,512,256,128,64], in_ch=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if load_vgg_weights:
            self._init_vgg()

    def _make_layers(self, cfg, in_ch=3, dilation=False):
        layers = []
        d = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=d, dilation=d), nn.ReLU(inplace=True)]
                in_ch = v
        return nn.Sequential(*layers)

    def _init_vgg(self):
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            vgg_features = list(vgg.features.children())[:23]
            vgg_seq = nn.Sequential(*vgg_features)
            vgg_state = vgg_seq.state_dict()
            frontend_state = self.frontend.state_dict()
            matched = {k:v for k,v in vgg_state.items() if k in frontend_state and frontend_state[k].shape==v.shape}
            frontend_state.update(matched)
            self.frontend.load_state_dict(frontend_state)
            print("[INFO] Loaded VGG16 frontend weights (partial).")
        except Exception as e:
            print("[WARN] Could not load VGG weights:", e)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

# ----------------------------
# utils
# ----------------------------
def build_transform(resize_shorter=512):
    return transforms.Compose([
        transforms.Resize(resize_shorter),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def preprocess_frame(frame_bgr, resize_shorter, device, fp16):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    transform = build_transform(resize_shorter)
    t = transform(pil).unsqueeze(0).to(device)
    if fp16:
        t = t.half()
    return t

def overlay_heatmap(frame_bgr, density_map, alpha=0.45):
    h,w = frame_bgr.shape[:2]
    dm = density_map.copy()
    denom = (dm.max() - dm.min()) if (dm.max() - dm.min())>1e-8 else 1.0
    dm_norm = (dm - dm.min())/denom
    dm_uint8 = (dm_norm*255).astype('uint8')
    cmap = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_JET)
    if cmap.shape[:2] != (h,w):
        cmap = cv2.resize(cmap, (w,h))
    return cv2.addWeighted(frame_bgr, 1-alpha, cmap, alpha, 0)

def load_roi_mask(path, frame_shape):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("ROI mask not found:", path)
    # resize mask to frame
    mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask>127).astype(np.uint8)  # binary
    return mask

def fit_linear_calibration(preds, gts):
    # preds, gts are lists or arrays
    if len(preds) < 2:
        return 1.0, 0.0
    a,b = np.polyfit(preds, gts, 1)
    return float(a), float(b)

# ----------------------------
# Multi-scale inference function
# ----------------------------
def multi_scale_density(model, frame, device, scales=(1.0,1.5), resize_base=512, fp16=False):
    """
    Run model at different input scales and fuse density maps.
    Returns fused density map in original frame size (H,W).
    """
    H, W = frame.shape[:2]
    densities = []
    for s in scales:
        # compute shorter-side resize for this scale
        resize = max(128, int(resize_base * s))
        inp = preprocess_frame(frame, resize_shorter=resize, device=device, fp16=fp16)
        with torch.no_grad():
            # optionally use autocast for fp16
            if fp16:
                with torch.cuda.amp.autocast():
                    out = model(inp)
            else:
                out = model(inp)
            dens = out.squeeze(0).squeeze(0).cpu().numpy()
            # resize to original frame size
            dens_resized = cv2.resize(dens, (W, H), interpolation=cv2.INTER_LINEAR)
            densities.append(dens_resized)
        # free GPU mem aggressively
        try:
            del out, inp
            torch.cuda.empty_cache()
        except Exception:
            pass
    # simple average fusion
    fused = np.mean(np.stack(densities, axis=0), axis=0)
    return fused

# ----------------------------
# main runner
# ----------------------------
def run_video(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device is None else (torch.device(args.device) if args.device else torch.device('cpu')))
    print("[INFO] Using device:", device)
    model = CSRNet(load_vgg_weights=True)
    ckpt = torch.load(args.weights, map_location='cpu')
    # support common checkpoint formats
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        state = {k.replace('module.',''):v for k,v in ckpt.items()}
    else:
        state = ckpt
    model_state = model.state_dict()
    filtered = {k:v for k,v in state.items() if k in model_state and model_state[k].shape==v.shape}
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    if args.fp16:
        model.half()
    model.eval()
    print("[INFO] Model loaded.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (W,H))

    # load ROI mask if provided
    roi_mask = None
    if args.roi_mask:
        roi_mask = load_roi_mask(args.roi_mask, (H,W))
        print("[INFO] Loaded ROI mask")

    # calibration (if provided) - read calib CSV with frame_idx,gt_count
    calib_a, calib_b = 1.0, 0.0
    if args.calib:
        gt_map = {}
        with open(args.calib, 'r') as f:
            r = csv.reader(f)
            for row in r:
                if len(row) < 2: continue
                try:
                    fi = int(row[0]); g = float(row[1])
                    gt_map[fi] = g
                except:
                    continue
        print(f"[INFO] Loaded {len(gt_map)} calibration frames")
    else:
        gt_map = {}

    results = []
    frame_idx = 0
    # precompute scales list
    scales = args.scales if args.scales else [1.0, 1.5]
    # smoothing
    smoothed = None
    alpha = args.alpha

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            fused = multi_scale_density(model, frame, device, scales=scales, resize_base=args.resize, fp16=args.fp16)
            # apply ROI mask if given
            if roi_mask is not None:
                fused = fused * roi_mask

            raw_pred = float(fused.sum())

            # if calibration frames provided, accumulate for fit
            if frame_idx in gt_map:
                # store for offline fit later
                if 'calib_preds' not in locals():
                    calib_preds = []
                    calib_gts = []
                calib_preds.append(raw_pred)
                calib_gts.append(gt_map[frame_idx])

            # if we have already fit calibration coefficients (once), apply
            if 'calib_a' in locals():
                pred_cal = calib_a * raw_pred + calib_b
            else:
                pred_cal = raw_pred

            # smoothing
            if smoothed is None:
                smoothed = pred_cal
            else:
                smoothed = alpha * pred_cal + (1.0 - alpha) * smoothed

            # write overlay
            overlay = overlay_heatmap(frame, fused, alpha=0.45)
            cv2.putText(overlay, f"Raw: {raw_pred:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(overlay, f"Smoothed: {smoothed:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if writer:
                writer.write(overlay)

            results.append((frame_idx, float(raw_pred), float(smoothed)))

            # progress + occasional cache clearing
            if frame_idx % max(1, int(fps*5)) == 0:
                print(f"[INFO] Frame {frame_idx}/{total}  raw={raw_pred:.2f}  smoothed={smoothed:.2f}  VRAM used(MB): {torch.cuda.memory_allocated(device)/1024**2 if device.type=='cuda' else 0:.1f}")

            # Periodically, after collecting calibration pairs, fit linear map
            if ('calib_preds' in locals()) and (len(calib_preds) >= args.calib_min and 'calib_a' not in locals()):
                calib_a, calib_b = fit_linear_calibration(calib_preds, calib_gts)
                # store as locals accessible above
                calib_a_val, calib_b_val = calib_a, calib_b
                calib_a, calib_b = calib_a_val, calib_b_val
                print(f"[INFO] Fitted calibration: pred*{calib_a:.4f} + {calib_b:.4f}")

    cap.release()
    if writer:
        writer.release()

    # save CSV
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_idx','raw_pred','smoothed'])
            for fi, rp, sm in results:
                w.writerow([fi, f"{rp:.3f}", f"{sm:.3f}"])
        print(f"[INFO] Saved counts to: {args.csv}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=str, required=True)
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--output', type=str, default='out_fixed.mp4')
    p.add_argument('--csv', type=str, default='counts_fixed.csv')
    p.add_argument('--resize', type=int, default=512, help='base resize shorter side')
    p.add_argument('--scales', nargs='+', type=float, default=[1.0,1.5], help='multi-scale factors (e.g. 0.75 1.0 1.25 1.5)')
    p.add_argument('--fp16', action='store_true', help='use fp16 mixed precision')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--calib', type=str, default=None, help='csv with frame_idx,gt_count for linear calibration (optional)')
    p.add_argument('--calib-min', dest='calib_min', type=int, default=10, help='min frames to fit calibration')
    p.add_argument('--roi_mask', type=str, default=None, help='path to binary ROI mask (optional)')
    p.add_argument('--alpha', type=float, default=0.4, help='EMA smoothing alpha')
    args = p.parse_args()
    run_video(args)
