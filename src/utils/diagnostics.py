# diagnose_density.py
import cv2, numpy as np, torch, argparse
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from csrnet_head_count import CSRNet, build_transform  # reuse CSRNet & transform

def preprocess(frame, resize_shorter=512, device='cpu'):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    t = build_transform(resize_shorter)(pil).unsqueeze(0).to(device)
    return t

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--frame', type=int, default=1)
    p.add_argument('--resize', type=int, default=512)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fi = max(1, min(args.frame, total))
    # seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi-1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("cannot read frame")

    model = CSRNet(load_vgg_weights=True)
    ckpt = torch.load(args.weights, map_location='cpu')
    # load state dict heuristically (same as in script)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    ms = model.state_dict()
    filtered = {k:v for k,v in state.items() if k in ms and ms[k].shape==v.shape}
    ms.update(filtered); model.load_state_dict(ms, strict=False)
    model.to(device).eval()

    inp = preprocess(frame, resize_shorter=args.resize, device=device)
    with torch.no_grad():
        out = model(inp)
    dens = out.squeeze().cpu().numpy()
    print("raw density shape:", dens.shape, "sum:", dens.sum())
    # resize to original frame
    dens_resized = cv2.resize(dens, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    print("resized sum:", dens_resized.sum(), "min/max:", dens_resized.min(), dens_resized.max(), "mean:", dens_resized.mean())

    # save visualization
    dm = dens_resized.copy()
    dm_norm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
    heat = (dm_norm*255).astype('uint8')
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
    outp = Path('diag_frame_{:04d}.jpg'.format(fi))
    cv2.imwrite(str(outp), overlay[:,:,::-1]) # save RGB->BGR
    print("Saved overlay to", outp)

    # plot with matplotlib for quick inspection
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.title('frame'); plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.subplot(1,2,2); plt.title('density heatmap'); plt.imshow(dens_resized, cmap='jet'); plt.colorbar(); plt.axis('off')
    plt.show()

if __name__=='__main__':
    main()
