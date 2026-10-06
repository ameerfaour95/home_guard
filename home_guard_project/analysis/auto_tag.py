"""Auto-tag v2: two detectors + tiles + fusion + tracking with gap filling.

    py -m home_guard_project.analysis.auto_tag SRC_DIR OUT_DIR [--fps 5] [--device 0|cpu] [--only CLIP ...] [--frames 225,240]

Per sampled frame, three sources:
  A  YOLO11x on the whole frame (imgsz 1280, test-time augmentation)
  B  RT-DETR-x on the whole frame
  C  YOLO11x on 2x2 overlapping tiles (small / far objects)
Boxes are fused per class (at most one box per source in a cluster, so two people
standing close stay two boxes); fused score = sum of the sources' scores / 3.
Then an IoU tracker links boxes over time, short gaps (<= 1 s) are filled, parked
(static) objects are held through gaps, and 1-2 frame blips are dropped.

Drawing: solid class color = sure (score >= 0.35); YELLOW = unsure; thin box with
"~" = filled in from neighbouring frames.  Labels hold every kept box; frames/*.json
keeps score, sources and status per box for later edits.
"""
import argparse, csv, glob, json, os, subprocess, collections
import cv2, numpy as np
from ultralytics import YOLO, RTDETR

COCO_TO_OURS = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 14: 6, 15: 7, 16: 8}
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog"]
COLORS = [(0, 220, 0), (255, 160, 0), (0, 140, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (200, 200, 0), (180, 105, 255), (128, 0, 255)]
YELLOW = (0, 230, 255)
VEHICLES = {2, 3, 4, 5}
FFMPEG = r"C:\Users\ameer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
SURE = 0.35


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def run(model, img, conf, imgsz, device, augment=False):
    r = model.predict(img, conf=conf, imgsz=imgsz, device=device, iou=0.75, augment=augment,
                      classes=list(COCO_TO_OURS), verbose=False)[0]
    return [(COCO_TO_OURS[int(c)], float(s), list(map(float, b)))
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def tiles_detect(model, img, device):
    H, W = img.shape[:2]; out = []
    tw, th = int(W * 0.62), int(H * 0.62)
    for x0 in (0, W - tw):
        for y0 in (0, H - th):
            for c, s, (x1, y1, x2, y2) in run(model, img[y0:y0+th, x0:x0+tw], 0.15, 1280, device):
                gx1, gy1, gx2, gy2 = x1 + x0, y1 + y0, x2 + x0, y2 + y0
                # a box cut by an inner tile edge belongs to another tile / the full-frame pass
                cut = (x1 < 2 and x0 > 0) or (y1 < 2 and y0 > 0) or (x2 > tw - 2 and x0 + tw < W) or (y2 > th - 2 and y0 + th < H)
                if not cut:
                    out.append((c, s, [gx1, gy1, gx2, gy2]))
    # tiles overlap: NMS within this source
    out.sort(key=lambda d: -d[1]); keep = []
    for d in out:
        if all(not (k[0] == d[0] and iou(k[2], d[2]) > 0.5) for k in keep):
            keep.append(d)
    return keep


def nms(dets, thr=0.7):
    dets = sorted(dets, key=lambda d: -d[1]); keep = []
    for d in dets:
        if all(not (k[0] == d[0] and iou(k[2], d[2]) > thr) for k in keep):
            keep.append(d)
    return keep


def inside(a, b):
    """fraction of box a that lies inside box b"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])
    return ix * iy / area if area > 0 else 0.0


def fuse(sources, n_sources=3, thr=0.4):
    """Weighted box fusion; one box per source per cluster."""
    sources = [nms(s, 0.6) for s in sources]  # TTA / RT-DETR give several boxes for one object
    allb = sorted(((s, c, b, si) for si, dets in enumerate(sources) for c, s, b in dets), key=lambda x: -x[0])
    clusters = []
    for s, c, b, si in allb:
        best, bi = None, 0.0
        for cl in clusters:
            if cl["c"] != c or si in cl["src"]:
                continue
            v = iou(cl["box"], b)
            if v > thr and v > bi:
                best, bi = cl, v
        if best is None:
            clusters.append(dict(c=c, src={si: (s, b)}, box=list(b)))
        else:
            best["src"][si] = (s, b)
            w = np.array([v[0] for v in best["src"].values()]); bb = np.array([v[1] for v in best["src"].values()])
            best["box"] = list((bb * w[:, None]).sum(0) / w.sum())
    out = []
    for cl in clusters:
        top = max(v[0] for v in cl["src"].values()); n = len(cl["src"])
        if n == 1 and top < 0.3:
            continue  # a single weak vote is noise
        score = min(1.0, top + 0.1 * (n - 1)) if n > 1 else top * 0.85
        out.append(dict(c=cl["c"], score=round(score, 3), box=cl["box"], src="".join("ABC"[i] for i in sorted(cl["src"]))))
    # cross-class duplicates (car vs truck on the same object): keep the stronger
    out.sort(key=lambda d: -d["score"]); keep = []
    for d in out:
        if any(k["c"] in VEHICLES and d["c"] in VEHICLES and iou(k["box"], d["box"]) > 0.7 for k in keep):
            continue
        keep.append(d)
    # a "group" box drawn around 2+ objects of its class, and weak part-boxes inside a strong box
    final = []
    for d in keep:
        members = [k for k in keep if k is not d and k["c"] == d["c"] and inside(k["box"], d["box"]) > 0.7 and k["score"] >= 0.5]
        if len(members) >= 2:
            continue
        if d["score"] < 0.5 and any(k is not d and k["c"] == d["c"] and k["score"] > d["score"] and inside(d["box"], k["box"]) > 0.85 for k in keep):
            continue
        final.append(d)
    return final


def track(per_frame, max_gap):
    """per_frame: list of det lists -> tracks {tid: {fi: det}}"""
    tracks, last, tid = {}, {}, 0
    for fi, dets in enumerate(per_frame):
        cand = []
        for t, (lfi, ld) in last.items():
            if fi - lfi > max_gap + 1:
                continue
            for di, d in enumerate(dets):
                if d["c"] == ld["c"] or (d["c"] in VEHICLES and ld["c"] in VEHICLES):
                    v = iou(ld["box"], d["box"])
                    if v > 0.2:
                        cand.append((v, t, di))
        cand.sort(reverse=True); used_t, used_d = set(), set()
        for v, t, di in cand:
            if t in used_t or di in used_d:
                continue
            used_t.add(t); used_d.add(di); tracks[t][fi] = dets[di]; last[t] = (fi, dets[di])
        for di, d in enumerate(dets):
            if di not in used_d:
                tid += 1; tracks[tid] = {fi: d}; last[tid] = (fi, d)
    return tracks


def smooth(tracks, n_frames, max_gap):
    out = {}
    for t, fr in tracks.items():
        idx = sorted(fr); scores = [fr[i]["score"] for i in idx]
        n_sure = sum(1 for v in scores if v >= SURE)
        if n_sure < max(2, 0.3 * len(idx)):
            continue                                  # weak or blip track: never filled in, never kept
        cls = collections.Counter(fr[i]["c"] for i in idx).most_common(1)[0][0]
        boxes = np.array([fr[i]["box"] for i in idx]); diag = np.hypot(*(boxes[:, 2:] - boxes[:, :2]).mean(0))
        ctr = (boxes[:, :2] + boxes[:, 2:]) / 2; motion = np.linalg.norm(ctr.max(0) - ctr.min(0)) / max(diag, 1)
        static = motion < 0.25 and n_sure >= 5 and len(idx) >= max(4, 0.3 * (idx[-1] - idx[0] + 1))
        new = {}
        for i in idx:
            d = dict(fr[i]); d["c"] = cls; d["status"] = "sure" if d["score"] >= SURE else "unsure"; new[i] = d
        for a, b in zip(idx, idx[1:]):
            if 1 < b - a <= (n_frames if static else max_gap + 1):
                for k in range(a + 1, b):
                    w = (k - a) / (b - a)
                    box = [(1 - w) * x + w * y for x, y in zip(fr[a]["box"], fr[b]["box"])]
                    new[k] = dict(c=cls, score=round(min(fr[a]["score"], fr[b]["score"]), 3), box=box, src="~", status="filled")
        if static:  # a parked object does not vanish at the clip edges either
            for k in list(range(0, idx[0])) + list(range(idx[-1] + 1, n_frames)):
                if abs(k - (idx[0] if k < idx[0] else idx[-1])) <= max_gap * 3:
                    new[k] = dict(c=cls, score=new[idx[0] if k < idx[0] else idx[-1]]["score"], box=list(fr[idx[0] if k < idx[0] else idx[-1]]["box"]), src="~", status="filled")
        out[t] = new
    return out


def draw(img, items, label_n, ts):
    H, W = img.shape[:2]; f = img.copy()
    for tid, d in items:
        x1, y1, x2, y2 = map(int, d["box"])
        col = COLORS[d["c"]] if d["status"] == "sure" else YELLOW if d["status"] == "unsure" else COLORS[d["c"]]
        th = 3 if d["status"] == "sure" else 2 if d["status"] == "unsure" else 1
        cv2.rectangle(f, (x1, y1), (x2, y2), col, th)
        lab = f"{NAMES[d['c']]} #{tid}" + ("?" if d["status"] == "unsure" else "~" if d["status"] == "filled" else "")
        fs = 0.5 if W < 500 else 0.75
        cv2.putText(f, lab, (x1, max(16, y1 - 5)), 0, fs, (0, 0, 0), 4); cv2.putText(f, lab, (x1, max(16, y1 - 5)), 0, fs, col, 2)
    cv2.rectangle(f, (0, 0), (min(W, 470), 34), (0, 0, 0), -1)
    cv2.putText(f, f"frame {label_n:04d}   ({ts:.1f}s)", (8, 25), 0, 0.75, (255, 255, 255), 2)
    return f


def tag_clip(models, path, out, fps_out, device, only_frames=None):
    yolo, detr = models
    clip = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path); nfps = cap.get(cv2.CAP_PROP_FPS) or 25.0; step = max(1, round(nfps / fps_out))
    frames, natives, n = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n % step == 0 and (not only_frames or n in only_frames):
            frames.append(f); natives.append(n)
        n += 1
    cap.release()
    per = []
    for f in frames:
        A = run(yolo, f, 0.10, 1280, device, augment=True)
        B = nms(run(detr, f, 0.15, 1280, device))
        C = tiles_detect(yolo, f, device)
        per.append(fuse([A, B, C]))
    max_gap = int(round(fps_out))  # 1 s
    tr = smooth(track(per, max_gap), len(frames), max_gap)
    for d in ("images", "labels", "frames"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    stats = collections.Counter(); per_class_tracks = collections.defaultdict(set)
    for fi, (f, nat) in enumerate(zip(frames, natives)):
        H, W = f.shape[:2]; name = f"{clip}_f{nat:04d}"
        items = sorted(((t, fr[fi]) for t, fr in tr.items() if fi in fr), key=lambda x: x[0])
        cv2.imwrite(os.path.join(out, "images", name + ".jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(out, "labels", name + ".txt"), "w") as fh:
            for t, d in items:
                x1, y1, x2, y2 = d["box"]
                fh.write(f"{d['c']} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                stats[d["status"]] += 1; per_class_tracks[NAMES[d["c"]]].add(t)
        cv2.imwrite(os.path.join(out, "frames", f"frame_{nat:04d}.jpg"), draw(f, items, nat, nat / nfps), [cv2.IMWRITE_JPEG_QUALITY, 85])
        json.dump([dict(track=t, cls=NAMES[d["c"]], box=[round(v, 1) for v in d["box"]], score=d["score"], sources=d["src"], status=d["status"]) for t, d in items],
                  open(os.path.join(out, "frames", f"frame_{nat:04d}.json"), "w"))
    if not only_frames:
        fr = sorted(glob.glob(os.path.join(out, "frames", "frame_*.jpg")))
        h, w = cv2.imread(fr[0]).shape[:2]; w -= w % 2; h -= h % 2
        raw = os.path.join(out, "tmp.avi"); vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), fps_out / 2, (w, h))
        for p in fr:
            vw.write(cv2.imread(p)[:h, :w])
        vw.release()
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, clip + ".mp4")], check=True)
        os.remove(raw)
    return dict(clip=clip, video_fps=round(nfps, 2), frames_tagged=len(frames), every_nth_frame=step,
                boxes_sure=stats["sure"], boxes_unsure=stats["unsure"], boxes_filled=stats["filled"],
                tracks=" ".join(f"{k}:{len(v)}" for k, v in sorted(per_class_tracks.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=5); ap.add_argument("--device", default="0")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--frames")
    a = ap.parse_args()
    models = (YOLO(r"C:\Users\ameer\Ameer\home_guard_data\models\yolo11x.pt"), RTDETR(r"C:\Users\ameer\Ameer\home_guard_data\models\rtdetr-x.pt"))
    clips = sorted(glob.glob(os.path.join(a.src, "**", "clips", "**", "*.mp4"), recursive=True))
    if a.only:
        clips = [c for c in clips if any(o in c for o in a.only)]
    only_frames = set(map(int, a.frames.split(","))) if a.frames else None
    rows = []
    for i, p in enumerate(clips, 1):
        cid = os.path.splitext(os.path.basename(p))[0]
        info = tag_clip(models, p, os.path.join(a.out, f"{i:02d}_{cid}"), a.fps, a.device, only_frames)
        rows.append(dict(n=i, source="dataset_uca" if "dataset_uca" in p else "dataset_smarthome", **info, claude_note="", what_to_fix=""))
        print(f"[{i}/{len(clips)}] {cid} {info['frames_tagged']} frames | sure {info['boxes_sure']} unsure {info['boxes_unsure']} filled {info['boxes_filled']} | {info['tracks']}", flush=True)
    if not only_frames:
        with open(os.path.join(a.out, "review.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("DONE", len(rows))


if __name__ == "__main__":
    main()
