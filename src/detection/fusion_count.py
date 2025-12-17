# #!/usr/bin/env python3
# """
# fusion_count.py — YOLO + CSRNet fusion (robust) with:
#  - safe handling of CSRNet outputs (avoid cv2.resize crash)
#  - flexible preprocessing calling
#  - density calibration and masking
#  - clearer overlay labels
# """

# import os
# import cv2
# import csv
# import numpy as np
# import torch
# from ultralytics import YOLO
# import inspect
# import warnings

# # Try to import CSRNet and preprocess helpers from csrnet_head_count.py
# try:
#     from csrnet_head_count import CSRNet, preprocess_frame_bgr as _csr_preproc
#     _preproc_name = "preprocess_frame_bgr"
# except Exception:
#     try:
#         from csrnet_head_count import CSRNet, preprocess_frame as _csr_preproc
#         _preproc_name = "preprocess_frame"
#     except Exception:
#         _csr_preproc = None
#         _preproc_name = None

# # Fallback local preprocess if none found
# if _csr_preproc is None:
#     from torchvision import transforms
#     from PIL import Image

#     def _local_build_transform(resize_shorter=512):
#         return transforms.Compose([
#             transforms.Resize(resize_shorter),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#         ])

#     def _csr_preproc(frame_bgr, resize_shorter=512, device='cpu', fp16=False):
#         img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         pil = Image.fromarray(img_rgb)
#         t = _local_build_transform(resize_shorter)(pil).unsqueeze(0).to(device)
#         if fp16:
#             t = t.half()
#         return t
#     _preproc_name = "local_preproc"

# # Helper to call preprocess function with correct signature
# def call_preproc(frame, resize_shorter, device, fp16):
#     sig = inspect.signature(_csr_preproc)
#     kwargs = {}
#     for name in sig.parameters:
#         if name in ('frame_bgr', 'frame'):
#             continue
#         if name == 'resize_shorter':
#             kwargs['resize_shorter'] = resize_shorter
#         elif name == 'device':
#             kwargs['device'] = device
#         elif name == 'fp16':
#             kwargs['fp16'] = fp16
#     return _csr_preproc(frame, **kwargs)

# # Density calibrator (estimate per-head mass)
# class DensityCalibrator:
#     def __init__(self):
#         self.values = []
#         self.scale = None
#     def update(self, density_sum, det_count):
#         if det_count >= 1 and det_count <= 12:
#             self.values.append(density_sum / max(det_count, 1))
#     def finalize(self):
#         if len(self.values) == 0:
#             self.scale = 1000.0
#         else:
#             self.scale = float(np.median(self.values))
#     def convert(self, density_sum):
#         if self.scale is None:
#             self.finalize()
#         return density_sum / self.scale

# # Safe conversion of CSRNet output to 2D numpy density map
# def csrnet_output_to_density(out_tensor, target_wh):
#     """
#     out_tensor: torch.Tensor (model output)
#     target_wh: (W, H) target size to resize to
#     Returns: 2D numpy array shape (H, W)
#     """
#     # Ensure tensor on CPU
#     try:
#         t = out_tensor.detach().cpu()
#     except Exception:
#         # If out_tensor is already numpy or unexpected, try conversions
#         if isinstance(out_tensor, np.ndarray):
#             arr = out_tensor
#             if arr.ndim == 2:
#                 return cv2.resize(arr, target_wh, interpolation=cv2.INTER_LINEAR)
#             elif arr.ndim >= 3:
#                 arr2 = arr.squeeze()
#                 if arr2.ndim == 2:
#                     return cv2.resize(arr2, target_wh, interpolation=cv2.INTER_LINEAR)
#         # fallback: zero map
#         return np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)

#     # t is tensor; convert shape possibilities to 2D
#     arr = None
#     # possible shapes: (1,1,H,W), (1,H,W), (H,W), (B,1,H,W)
#     if t.ndim == 4:
#         # take first batch, first channel
#         if t.size(0) >= 1 and t.size(1) >= 1:
#             arr = t[0,0,:,:].numpy()
#         else:
#             arr = t.squeeze().numpy() if t.numel()>0 else None
#     elif t.ndim == 3:
#         # maybe (1,H,W)
#         if t.size(0) == 1:
#             arr = t[0,:,:].numpy()
#         else:
#             arr = t.squeeze().numpy()
#     elif t.ndim == 2:
#         arr = t.numpy()
#     else:
#         arr = t.squeeze().numpy() if t.numel()>0 else None

#     if arr is None or arr.size == 0:
#         # fallback zero map and warn
#         warnings.warn("CSRNet output was empty or unexpected; using zero density map.")
#         return np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)

#     # resize to target_wh = (W,H)
#     try:
#         resized = cv2.resize(arr, target_wh, interpolation=cv2.INTER_LINEAR)
#     except Exception as e:
#         warnings.warn(f"cv2.resize failed on CSRNet output: {e}; returning zeros.")
#         resized = np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)
#     return resized

# # Main fusion runner
# def run_fusion(video_path, yolo_path, csr_path, output_path, csv_path,
#                tile_w=600, tile_h=400, overlap=0.6,
#                conf=0.30, nms=0.45, dilate=40, fp16=True,
#                debug_heat=False, resize_shorter=512):

#     print("[INFO] Loading YOLO:", yolo_path)
#     yolo = YOLO(yolo_path)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("[INFO] Device:", device)

#     # Load CSRNet
#     csr = CSRNet(load_vgg_weights=True)
#     if not os.path.exists(csr_path):
#         raise FileNotFoundError("CSRNet weights not found: " + csr_path)
#     ckpt = torch.load(csr_path, map_location='cpu')
#     if isinstance(ckpt, dict) and 'model' in ckpt:
#         state = ckpt['model']
#     elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
#         state = {k.replace('module.',''):v for k,v in ckpt.items()}
#     else:
#         state = ckpt
#     model_state = csr.state_dict()
#     filtered = {k:v for k,v in state.items() if k in model_state and model_state[k].shape==v.shape}
#     model_state.update(filtered)
#     csr.load_state_dict(model_state, strict=False)
#     csr.to(device)
#     if fp16:
#         try:
#             csr.half()
#         except Exception:
#             pass
#     csr.eval()
#     print("[INFO] CSRNet loaded. Preproc helper:", _preproc_name)

#     # Video IO
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open video: " + str(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     print(f"[INFO] Video: {video_path} ({W}x{H}) frames={total} fps={fps}")

#     writer = None
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         writer = cv2.VideoWriter(output_path, fourcc, fps, (W,H))

#     # prepare tile coords
#     step_x = max(1, int(tile_w * (1 - overlap)))
#     step_y = max(1, int(tile_h * (1 - overlap)))
#     tile_coords = []
#     for y in range(0, max(1, H - tile_h + 1), step_y):
#         for x in range(0, max(1, W - tile_w + 1), step_x):
#             x2 = min(W, x + tile_w); y2 = min(H, y + tile_h)
#             tile_coords.append((x, y, x2, y2))
#     if len(tile_coords) == 0:
#         tile_coords = [(0,0,W,H)]

#     frame_idx = 0
#     results = []
#     calibrator = DensityCalibrator()

#     with torch.no_grad():
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_idx += 1

#             # YOLO tiled detection
#             all_boxes = []
#             all_scores = []
#             for (x1,y1,x2,y2) in tile_coords:
#                 crop = frame[y1:y2, x1:x2]
#                 if crop.size == 0:
#                     continue
#                 res = yolo(crop)[0]
#                 for det in res.boxes:
#                     conf_score = float(det.conf[0])
#                     if conf_score < conf:
#                         continue
#                     xyxy = det.xyxy[0].cpu().numpy().tolist()
#                     bx1 = xyxy[0] + x1; by1 = xyxy[1] + y1
#                     bx2 = xyxy[2] + x1; by2 = xyxy[3] + y1
#                     all_boxes.append([bx1,by1,bx2,by2])
#                     all_scores.append(conf_score)

#             # NMS across full frame
#             if len(all_boxes) > 0:
#                 boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
#                 scores_t = torch.tensor(all_scores, dtype=torch.float32)
#                 keep = torch.ops.torchvision.nms(boxes_t, scores_t, nms).cpu().numpy().tolist()
#                 boxes_keep = boxes_t[keep].cpu().numpy().tolist()
#                 scores_keep = scores_t[keep].cpu().numpy().tolist()
#             else:
#                 boxes_keep = []
#                 scores_keep = []

#             # filter by small size
#             filtered_boxes = []
#             filtered_scores = []
#             for b,s in zip(boxes_keep, scores_keep):
#                 w = b[2]-b[0]; h = b[3]-b[1]
#                 if w < 6 or h < 6:
#                     continue
#                 filtered_boxes.append(b); filtered_scores.append(s)
#             det_count = len(filtered_boxes)

#             # CSRNet density: preprocess and infer
#             try:
#                 inp = call_preproc(frame, resize_shorter, device, fp16)
#             except Exception as e:
#                 warnings.warn(f"Preprocess call failed: {e}; using fallback local preproc.")
#                 # fallback local build
#                 from torchvision import transforms
#                 from PIL import Image
#                 def _local_build_transform(resize_shorter=512):
#                     return transforms.Compose([transforms.Resize(resize_shorter),
#                                                transforms.ToTensor(),
#                                                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
#                 pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 inp = _local_build_transform(resize_shorter)(pil).unsqueeze(0).to(device)
#                 if fp16:
#                     inp = inp.half()

#             if fp16 and inp.dtype != torch.half:
#                 try:
#                     inp = inp.half()
#                 except Exception:
#                     pass

#             out = csr(inp)

#             # convert CSRNet output safely to density map (H,W)
#             dens_resized = csrnet_output_to_density(out, (W, H))

#             # mask density inside detector boxes (with dilation)
#             mask = np.ones_like(dens_resized, dtype=np.uint8)
#             for b in filtered_boxes:
#                 x1,y1,x2,y2 = [int(round(v)) for v in b]
#                 x1d = max(0, x1 - dilate)
#                 y1d = max(0, y1 - dilate)
#                 x2d = min(W-1, x2 + dilate)
#                 y2d = min(H-1, y2 + dilate)
#                 mask[y1d:y2d, x1d:x2d] = 0

#             density_remaining = dens_resized * mask
#             try:
#                 density_sum = float(np.sum(density_remaining))
#             except Exception:
#                 density_sum = 0.0

#             # update calibrator and compute calibrated density
#             calibrator.update(density_sum, det_count)
#             calibrated_density = calibrator.convert(density_sum)

#             final_count = det_count + calibrated_density

#             # visualization
#             vis = frame.copy()
#             for b,s in zip(filtered_boxes, filtered_scores):
#                 x1,y1,x2,y2 = [int(round(v)) for v in b]
#                 cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
#                 cv2.putText(vis, f"{s:.2f}", (x1, max(6,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

#             dm = density_remaining.copy()
#             denom = (dm.max() - dm.min()) if (dm.max() - dm.min())>1e-8 else 1.0
#             dm_norm = (dm - dm.min())/denom
#             dm_uint8 = (dm_norm*255).astype('uint8')
#             heat = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_JET)
#             overlay = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)

#             label = f"DETECT: {det_count}   DENSITY: {calibrated_density:.2f}   FINAL: {final_count:.2f}"
#             cv2.putText(overlay, label, (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

#             if writer:
#                 writer.write(overlay)

#             results.append((frame_idx, det_count, float(calibrated_density), float(final_count)))

#             if debug_heat and frame_idx == 5:
#                 cv2.imwrite("debug_heatmap_frame5.jpg", overlay)

#             if frame_idx % max(1, int(fps*2)) == 0:
#                 print(f"[INFO] Frame {frame_idx}/{total}  det={det_count}  dens={calibrated_density:.2f}  final={final_count:.2f}")

#     cap.release()
#     if writer:
#         writer.release()

#     # write CSV
#     if csv_path:
#         with open(csv_path, "w", newline="") as f:
#             w = csv.writer(f)
#             w.writerow(['frame_idx','det_count','density_norm','final_count'])
#             for row in results:
#                 w.writerow(row)
#         print("[INFO] Saved CSV:", csv_path)

#     print("[DONE] Output:", output_path)


# # CLI
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument('--video', required=True)
#     p.add_argument('--yolo', required=True)
#     p.add_argument('--csr-weights', required=True)
#     p.add_argument('--output', default='out_fusion.mp4')
#     p.add_argument('--csv', default='fusion_counts.csv')
#     p.add_argument('--tile-w', type=int, default=600)
#     p.add_argument('--tile-h', type=int, default=400)
#     p.add_argument('--overlap', type=float, default=0.6)
#     p.add_argument('--conf', type=float, default=0.30)
#     p.add_argument('--nms', type=float, default=0.45)
#     p.add_argument('--dilate', type=int, default=40)
#     p.add_argument('--fp16', action='store_true')
#     p.add_argument('--debug-heat', action='store_true')
#     p.add_argument('--resize', type=int, default=512)
#     args = p.parse_args()

#     # pass resize_shorter param through
#     resize_shorter = args.resize

#     run_fusion(
#         args.video, args.yolo, args.csr_weights,
#         args.output, args.csv,
#         args.tile_w, args.tile_h, args.overlap,
#         args.conf, args.nms, args.dilate,
#         args.fp16, args.debug_heat, resize_shorter
#     )


#!/usr/bin/env python3
"""
fusion_count.py — YOLO + CSRNet fusion (robust).

Features:
 - tiled YOLO detection (ultralytics)
 - class filtering (--yolo-class)
 - CSRNet density estimation with safe output handling
 - mask density inside detected boxes (dilate)
 - automatic calibration using first N (--calib-frames) frames then freeze
 - writes CSV with raw and calibrated density and final counts
 - options: --fp16, --debug-heat
"""

import os
import cv2
import csv
import numpy as np
import torch
from ultralytics import YOLO
import inspect
import warnings

# Try to import CSRNet and preprocess helpers from your csrnet_head_count.py
try:
    from csrnet_head_count import CSRNet, preprocess_frame_bgr as _csr_preproc
    _preproc_name = "preprocess_frame_bgr"
except Exception:
    try:
        from csrnet_head_count import CSRNet, preprocess_frame as _csr_preproc
        _preproc_name = "preprocess_frame"
    except Exception:
        _csr_preproc = None
        _preproc_name = None

# Fallback local preprocess if none found
if _csr_preproc is None:
    from torchvision import transforms
    from PIL import Image

    def _local_build_transform(resize_shorter=512):
        return transforms.Compose([
            transforms.Resize(resize_shorter),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _csr_preproc(frame_bgr, resize_shorter=512, device='cpu', fp16=False):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        t = _local_build_transform(resize_shorter)(pil).unsqueeze(0).to(device)
        if fp16:
            t = t.half()
        return t
    _preproc_name = "local_preproc"

# Helper to call preprocess function with correct signature
def call_preproc(frame, resize_shorter, device, fp16):
    sig = inspect.signature(_csr_preproc)
    kwargs = {}
    for name in sig.parameters:
        if name in ('frame_bgr', 'frame'):
            continue
        if name == 'resize_shorter':
            kwargs['resize_shorter'] = resize_shorter
        elif name == 'device':
            kwargs['device'] = device
        elif name == 'fp16':
            kwargs['fp16'] = fp16
    return _csr_preproc(frame, **kwargs)

# Density calibrator (estimate per-head mass)
class DensityCalibrator:
    def __init__(self):
        self.values = []
        self.scale = None
    def update(self, density_sum, det_count):
        if det_count >= 1 and det_count <= 12:
            self.values.append(density_sum / max(det_count, 1))
    def finalize(self):
        if len(self.values) == 0:
            # fallback conservative scale
            self.scale = 1000.0
        else:
            self.scale = float(np.median(self.values))
    def convert(self, density_sum):
        if self.scale is None:
            self.finalize()
        return density_sum / self.scale

# Safe conversion of CSRNet output to 2D numpy density map
def csrnet_output_to_density(out_tensor, target_wh):
    """
    out_tensor: torch.Tensor (model output)
    target_wh: (W, H) target size to resize to
    Returns: 2D numpy array shape (H, W)
    """
    # try detach->cpu
    try:
        t = out_tensor.detach().cpu()
    except Exception:
        # fallback if numpy already
        if isinstance(out_tensor, np.ndarray):
            arr = out_tensor
            arr2 = arr.squeeze()
            if arr2.ndim == 2:
                return cv2.resize(arr2, target_wh, interpolation=cv2.INTER_LINEAR)
        return np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)

    arr = None
    if t.ndim == 4:
        # (B,C,H,W) -> take first batch, first channel
        if t.size(0) >= 1 and t.size(1) >= 1:
            arr = t[0,0,:,:].numpy()
        else:
            arr = t.squeeze().numpy() if t.numel()>0 else None
    elif t.ndim == 3:
        # maybe (1,H,W)
        if t.size(0) == 1:
            arr = t[0,:,:].numpy()
        else:
            arr = t.squeeze().numpy()
    elif t.ndim == 2:
        arr = t.numpy()
    else:
        arr = t.squeeze().numpy() if t.numel()>0 else None

    if arr is None or arr.size == 0:
        warnings.warn("CSRNet output empty/unexpected; using zero density map.")
        return np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)

    try:
        resized = cv2.resize(arr, target_wh, interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        warnings.warn(f"cv2.resize failed on CSRNet output: {e}; returning zeros.")
        resized = np.zeros((target_wh[1], target_wh[0]), dtype=np.float32)
    return resized

# Main fusion runner
def run_fusion(video_path, yolo_path, csr_path, output_path, csv_path,
               tile_w=600, tile_h=400, overlap=0.6,
               conf=0.30, nms=0.45, dilate=40, fp16=True,
               debug_heat=False, resize_shorter=512,
               yolo_class=0, calib_frames=50):

    print("[INFO] Loading YOLO:", yolo_path)
    yolo = YOLO(yolo_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device:", device)

    # Load CSRNet
    csr = CSRNet(load_vgg_weights=True)
    if not os.path.exists(csr_path):
        raise FileNotFoundError("CSRNet weights not found: " + csr_path)
    ckpt = torch.load(csr_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        state = {k.replace('module.',''):v for k,v in ckpt.items()}
    else:
        state = ckpt
    model_state = csr.state_dict()
    filtered = {k:v for k,v in state.items() if k in model_state and model_state[k].shape==v.shape}
    model_state.update(filtered)
    csr.load_state_dict(model_state, strict=False)
    csr.to(device)
    if fp16:
        try:
            csr.half()
        except Exception:
            pass
    csr.eval()
    print("[INFO] CSRNet loaded. Preproc helper:", _preproc_name)

    # Video IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video: {video_path} ({W}x{H}) frames={total} fps={fps}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W,H))

    # prepare tile coords
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))
    tile_coords = []
    for y in range(0, max(1, H - tile_h + 1), step_y):
        for x in range(0, max(1, W - tile_w + 1), step_x):
            x2 = min(W, x + tile_w); y2 = min(H, y + tile_h)
            tile_coords.append((x, y, x2, y2))
    if len(tile_coords) == 0:
        tile_coords = [(0,0,W,H)]

    frame_idx = 0
    results = []
    calibrator = DensityCalibrator()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO tiled detection
            all_boxes = []
            all_scores = []
            for (x1,y1,x2,y2) in tile_coords:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                res = yolo(crop)[0]
                for det in res.boxes:
                    conf_score = float(det.conf[0])
                    if conf_score < conf:
                        continue
                    # class filtering (Ultralytics uses det.cls)
                    cls_id = None
                    try:
                        cls_id = int(det.cls[0].cpu().item())
                    except Exception:
                        pass
                    if cls_id is not None and cls_id != yolo_class:
                        continue
                    xyxy = det.xyxy[0].cpu().numpy().tolist()
                    bx1 = xyxy[0] + x1; by1 = xyxy[1] + y1
                    bx2 = xyxy[2] + x1; by2 = xyxy[3] + y1
                    all_boxes.append([bx1,by1,bx2,by2])
                    all_scores.append(conf_score)

            # NMS across full frame
            if len(all_boxes) > 0:
                boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
                scores_t = torch.tensor(all_scores, dtype=torch.float32)
                keep = torch.ops.torchvision.nms(boxes_t, scores_t, nms).cpu().numpy().tolist()
                boxes_keep = boxes_t[keep].cpu().numpy().tolist()
                scores_keep = scores_t[keep].cpu().numpy().tolist()
            else:
                boxes_keep = []
                scores_keep = []

            # filter by small size
            filtered_boxes = []
            filtered_scores = []
            for b,s in zip(boxes_keep, scores_keep):
                w = b[2]-b[0]; h = b[3]-b[1]
                if w < 6 or h < 6:
                    continue
                filtered_boxes.append(b); filtered_scores.append(s)
            det_count = len(filtered_boxes)

            # CSRNet density: preprocess and infer
            try:
                inp = call_preproc(frame, resize_shorter, device, fp16)
            except Exception as e:
                warnings.warn(f"Preprocess call failed: {e}; using fallback local preproc.")
                from torchvision import transforms
                from PIL import Image
                def _local_build_transform(resize_shorter=512):
                    return transforms.Compose([transforms.Resize(resize_shorter),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inp = _local_build_transform(resize_shorter)(pil).unsqueeze(0).to(device)
                if fp16:
                    inp = inp.half()

            if fp16 and inp.dtype != torch.half:
                try:
                    inp = inp.half()
                except Exception:
                    pass

            out = csr(inp)

            # convert CSRNet output safely to density map (H,W)
            dens_resized = csrnet_output_to_density(out, (W, H))

            # mask density inside detector boxes (with dilation)
            mask = np.ones_like(dens_resized, dtype=np.uint8)
            for b in filtered_boxes:
                x1,y1,x2,y2 = [int(round(v)) for v in b]
                x1d = max(0, x1 - dilate)
                y1d = max(0, y1 - dilate)
                x2d = min(W-1, x2 + dilate)
                y2d = min(H-1, y2 + dilate)
                mask[y1d:y2d, x1d:x2d] = 0

            density_remaining = dens_resized * mask
            try:
                raw_density_sum = float(np.sum(density_remaining))
            except Exception:
                raw_density_sum = 0.0

            # calibration: collect for first calib_frames frames, then freeze
            if frame_idx <= calib_frames:
                calibrator.update(raw_density_sum, det_count)
                if frame_idx == calib_frames:
                    calibrator.finalize()
            if calibrator.scale is None and frame_idx > calib_frames:
                calibrator.finalize()

            calibrated_density = calibrator.convert(raw_density_sum)
            final_count = det_count + calibrated_density

            # visualization
            vis = frame.copy()
            for b,s in zip(filtered_boxes, filtered_scores):
                x1,y1,x2,y2 = [int(round(v)) for v in b]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{s:.2f}", (x1, max(6,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            dm = density_remaining.copy()
            denom = (dm.max() - dm.min()) if (dm.max() - dm.min())>1e-8 else 1.0
            dm_norm = (dm - dm.min())/denom
            dm_uint8 = (dm_norm*255).astype('uint8')
            heat = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)

            label = f"DETECT: {det_count}  RAW: {raw_density_sum:.1f}  CAL: {calibrated_density:.2f}  FINAL: {final_count:.2f}"
            cv2.putText(overlay, label, (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            if writer:
                writer.write(overlay)

            results.append((frame_idx, det_count, raw_density_sum, float(calibrated_density), float(final_count)))

            if debug_heat and frame_idx == 5:
                cv2.imwrite("debug_heatmap_frame5.jpg", overlay)

            if frame_idx % max(1, int(fps*2)) == 0:
                print(f"[INFO] Frame {frame_idx}/{total}  det={det_count}  raw={raw_density_sum:.1f}  cal={calibrated_density:.2f}  final={final_count:.2f}")

    cap.release()
    if writer:
        writer.release()

    # write CSV
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(['frame_idx','det_count','raw_density','density_norm','final_count'])
            for row in results:
                w.writerow(row)
        print("[INFO] Saved CSV:", csv_path)

    print("[DONE] Output:", output_path)


# CLI
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--yolo', required=True)
    p.add_argument('--yolo-class', type=int, default=0, help='class id in YOLO to count (default 0 = person)')
    p.add_argument('--csr-weights', required=True)
    p.add_argument('--output', default='out_fusion.mp4')
    p.add_argument('--csv', default='fusion_counts.csv')
    p.add_argument('--tile-w', type=int, default=600)
    p.add_argument('--tile-h', type=int, default=400)
    p.add_argument('--overlap', type=float, default=0.6)
    p.add_argument('--conf', type=float, default=0.30)
    p.add_argument('--nms', type=float, default=0.45)
    p.add_argument('--dilate', type=int, default=40)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--debug-heat', action='store_true')
    p.add_argument('--resize', type=int, default=512)
    p.add_argument('--calib-frames', type=int, default=50, help='number of frames to build calibration scale then freeze')
    args = p.parse_args()

    resize_shorter = args.resize

    run_fusion(
        args.video, args.yolo, args.csr_weights,
        args.output, args.csv,
        args.tile_w, args.tile_h, args.overlap,
        args.conf, args.nms, args.dilate,
        args.fp16, args.debug_heat, resize_shorter,
        yolo_class=args.yolo_class, calib_frames=args.calib_frames
    )
