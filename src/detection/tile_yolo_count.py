# tile_yolo_count.py
import cv2, numpy as np, argparse, torch
from torchvision.ops import nms
from ultralytics import YOLO

def tile_frame(frame, tile_w, tile_h, overlap):
    H, W = frame.shape[:2]
    step_x = int(tile_w * (1 - overlap))
    step_y = int(tile_h * (1 - overlap))
    tiles = []
    for y in range(0, max(1, H - tile_h + 1), max(1, step_y)):
        for x in range(0, max(1, W - tile_w + 1), max(1, step_x)):
            x2 = min(W, x + tile_w); y2 = min(H, y + tile_h)
            tiles.append((x, y, x2, y2))
    # ensure last row/col included
    if tiles == []:
        tiles = [(0,0,W,H)]
    return tiles

def merge_detections(boxes, scores, iou_thr=0.45):
    # boxes: Nx4 (x1,y1,x2,y2) in numpy
    if len(boxes)==0:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_thr).numpy().tolist()
    boxes_keep = boxes_t[keep].numpy().tolist()
    scores_keep = scores_t[keep].numpy().tolist()
    return boxes_keep, scores_keep

def run_video(args):
    model = YOLO(args.yolo)  # ultralytics
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); writer = cv2.VideoWriter(args.output, fourcc, fps, (W,H))
    tiles = tile_frame(np.zeros((H,W,3),dtype=np.uint8), args.tile_w, args.tile_h, args.overlap)
    frame_idx = 0
    results = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        all_boxes = []
        all_scores = []
        for (x1,y1,x2,y2) in tiles:
            tile = frame[y1:y2, x1:x2]
            # resize tile so detector sees heads at reasonable sizes if desired:
            # not required if your head model works at original tile size
            res = model(tile)[0]  # ultralytics returns batch, get first
            for det in res.boxes:
                conf = float(det.conf[0])
                if conf < args.conf: continue
                xyxy = det.xyxy[0].cpu().numpy().tolist()  # x1,y1,x2,y2 relative to tile
                # map to full frame coords
                bx1 = xyxy[0] + x1; by1 = xyxy[1] + y1
                bx2 = xyxy[2] + x1; by2 = xyxy[3] + y1
                all_boxes.append([bx1,by1,bx2,by2])
                all_scores.append(conf)
        # merge across tiles with NMS
        boxes_keep, scores_keep = merge_detections(all_boxes, all_scores, iou_thr=args.nms)
        # optional filter by box size (remove tiny/huge boxes)
        filtered = []
        for b,s in zip(boxes_keep, scores_keep):
            w = b[2]-b[0]; h = b[3]-b[1]
            if w < args.min_box or h < args.min_box: continue
            if w > args.max_box or h > args.max_box:  # optionally ignore extremely large boxes
                pass
            filtered.append((b,s))
        # draw
        for b,s in filtered:
            x1,y1,x2,y2 = map(int,b)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{s:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        count = len(filtered)
        results.append((frame_idx, count))
        cv2.putText(frame, f"Count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        if writer: writer.write(frame)
        if frame_idx % 50 == 0: print(f"[INFO] Frame {frame_idx}, count={count}")
    cap.release()
    if writer: writer.release()
    # save CSV
    import csv
    with open(args.csv,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame','count'])
        for fi,c in results: w.writerow([fi,c])
    print("[DONE] Saved", args.csv)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--yolo', required=True)
    p.add_argument('--tile-w', type=int, default=800)
    p.add_argument('--tile-h', type=int, default=600)
    p.add_argument('--overlap', type=float, default=0.5)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--nms', type=float, default=0.45)
    p.add_argument('--min-box', type=int, default=12)
    p.add_argument('--max-box', type=int, default=2000)
    p.add_argument('--output', type=str, default='out_tile_yolo.mp4')
    p.add_argument('--csv', type=str, default='tile_counts.csv')
    args = p.parse_args()
    run_video(args)
