"""One-time migration (2026-10-06): build home_guard_data/dataset, one folder where every
clip is stored once and YOLO and the VLM read from the same files.

Layout: clips/ crops/ meta/ <source>/<clip>; annotations/clips.jsonl (one row per clip:
description, alert, source, who tagged it, review flags) + label_studio/ originals;
yolo/images|labels + frames.txt + hard_negatives.txt (no train/val split yet, by the
owner's choice); vlm/clips.jsonl; review/.  Paths below are the laptop paths it ran with.

Copied (never moved) from:
  home_guard_yolo/s3_tagging_full/   full S3 tagging/ mirror (exports, clips, meta, crops, model responses)
  home_guard_yolo/tagging/           vehicle-fixed exports + review lists
  home_guard_yolo/analysis/          human descriptions (vlm_training.jsonl per batch)
  home_guard_yolo/dataset_v1/        fixed YOLO frames + labels
  home_guard_yolo/claude_tagging/    10 clips auto-tagged by Claude (pending the owner's review)
"""
import csv, glob, json, os, re, shutil, sys, collections
sys.stdout.reconfigure(encoding="utf-8")

Y = r"C:\Users\ameer\Ameer\home_guard_yolo"
D = r"C:\Users\ameer\Ameer\home_guard_data\dataset"
S3_ROOT = "s3://security-camera-project-v1/home_guard_dataset"
BATCHES = ["ameer_house_batch_1", "ameer_house_batch_2", "uca_dataset_batch", "smarthome_dataset_batch"]
MIRROR = os.path.join(Y, "s3_tagging_full")


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)


def source_of(batch, clip_id):
    if batch.startswith("uca"):
        return "uca"
    if batch.startswith("smarthome"):
        return "smarthome"
    return "external" if clip_id.startswith("external_") else "house"


def index(root, suffix, folder=None):
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*" + suffix), recursive=True):
        if folder and os.sep + folder + os.sep not in p:
            continue
        out.setdefault(os.path.basename(p)[: -len(suffix)], p)
    return out


review = collections.defaultdict(list)
with open(os.path.join(Y, "review_for_humans.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        review[r["clip"]].append(r["why"])

rows, missing = [], collections.Counter()
for b in BATCHES:
    bdir = os.path.join(MIRROR, b)
    clips, crops, metas = index(bdir, ".mp4", "clips"), index(bdir, ".mp4", "vlm_crops"), index(bdir, ".meta.json", "meta")
    resp = index(bdir, ".model_raw.txt")
    tasks = {}
    for t in json.load(open(os.path.join(bdir, b + ".json"), encoding="utf-8")):
        url = t["data"].get("video_url") or t["data"].get("video", "")
        tasks[os.path.basename(url.split("?")[0])[:-4]] = t
    # annotations: originals + vehicle-fixed export
    cp(os.path.join(bdir, b + ".json"), os.path.join(D, "annotations", "label_studio", b + ".json"))
    cp(os.path.join(Y, "tagging", b, b + ".vehicles.json"), os.path.join(D, "annotations", "label_studio", b + ".vehicle_fixed.json"))
    cp(os.path.join(Y, "tagging", b, b + ".vehicles.csv"), os.path.join(D, "annotations", "fixes", b + ".vehicle_tracks.csv"))
    for line in open(os.path.join(Y, "analysis", b, "vlm_training.jsonl"), encoding="utf-8"):
        v = json.loads(line)
        cid = v["clip_id"]; src = source_of(b, cid)
        if cid not in clips:
            missing["clip"] += 1; print("MISSING clip", b, cid); continue
        rec = dict(clip_id=cid, source=src, batch=b, ls_task_id=tasks.get(cid, {}).get("id"),
                   camera=v.get("camera_name"), date=v.get("date"), duration_sec=v.get("duration_sec"),
                   split=None)
        rec["clip"] = f"clips/{src}/{cid}.mp4"; cp(clips[cid], os.path.join(D, *rec["clip"].split("/")))
        if cid in crops:
            rec["crop"] = f"crops/{src}/{cid}.mp4"; cp(crops[cid], os.path.join(D, *rec["crop"].split("/")))
        else:
            rec["crop"] = None; missing["crop"] += 1
        if cid in metas:
            rec["meta"] = f"meta/{src}/{cid}.meta.json"; cp(metas[cid], os.path.join(D, *rec["meta"].split("/")))
        else:
            rec["meta"] = None; missing["meta"] += 1
        if cid in resp:
            cp(resp[cid], os.path.join(D, "annotations", "model_responses", src, cid + ".model_raw.txt"))
        desc = (v.get("description") or "").strip()
        ALERT = re.compile(r"\s*\[(alert|alet)\]\s*", re.I)  # "[alet]" is a tagger typo for "[alert]"
        rec.update(description=ALERT.sub(" ", desc).strip(), alert=bool(ALERT.search(desc)),
                   description_by="human (Label Studio)", boxes_by="human (Label Studio), fixed 2026-10-06",
                   num_persons=v.get("num_persons"), num_cars=v.get("num_cars"),
                   needs_check="; ".join(review.get(cid, [])) or None)
        rows.append(rec)

# YOLO: every frame + label in one place (no train/val split yet), hard negatives listed separately
train_lines = open(os.path.join(Y, "dataset_v1", "train.txt")).read().split()
val_lines = open(os.path.join(Y, "dataset_v1", "val.txt")).read().split()
reps = collections.Counter(train_lines + val_lines)
frames_all, hard_neg = [], []
for line in sorted(reps):
    rel = line.replace("./", "", 1)                       # images/train/<cam>/<name>.jpg
    parts = rel.split("/")                                # drop the train|val level
    flat_img = "/".join(["images"] + parts[2:])
    src_img = os.path.join(Y, "dataset_v1", *parts)
    src_lab = os.path.join(Y, "dataset_v1", "labels", *parts[1:])[:-4] + ".txt"
    cp(src_img, os.path.join(D, "yolo", *flat_img.split("/")))
    cp(src_lab, os.path.join(D, "yolo", "labels", *parts[2:])[:-4] + ".txt")
    frames_all.append(flat_img)
    if reps[line] > 1:
        hard_neg.append(flat_img)
with open(os.path.join(D, "yolo", "frames.txt"), "w") as f:
    f.writelines(x + chr(10) for x in frames_all)
with open(os.path.join(D, "yolo", "hard_negatives.txt"), "w") as f:
    f.writelines(x + chr(10) for x in hard_neg)
names = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog"]
with open(os.path.join(D, "yolo", "classes.txt"), "w") as f:
    f.writelines(f"{i} {n}" + chr(10) for i, n in enumerate(names))
yolo_frames = collections.Counter()
for line in frames_all:
    yolo_frames[os.path.basename(line).rsplit("_f", 1)[0]] += 1
for r in rows:
    r["yolo_frames"] = yolo_frames.get(r["clip_id"], 0)

# Claude-tagged clips: boxes pending the owner's review, description left empty
ct = os.path.join(Y, "claude_tagging")
if os.path.exists(os.path.join(ct, "review.csv")):
    srcs = index(os.path.join(ct, "_source"), ".mp4", "clips"); smeta = index(os.path.join(ct, "_source"), ".meta.json", "meta")
    with open(os.path.join(ct, "review.csv"), encoding="utf-8-sig") as f:
        crows = list(csv.DictReader(f))
    for r in crows:
        cid = r["clip"]; src = "uca" if "dataset_uca" in srcs[cid] else "smarthome"
        folder = glob.glob(os.path.join(ct, f"*_{cid}"))[0]
        rec = dict(clip_id=cid, source=src, batch="claude_tagged_2026-10-06", ls_task_id=None, camera=None, date=None,
                   duration_sec=None, split=None, clip=f"clips/{src}/{cid}.mp4", crop=None,
                   meta=f"meta/{src}/{cid}.meta.json" if cid in smeta else None)
        cp(srcs[cid], os.path.join(D, *rec["clip"].split("/")))
        if rec["meta"]:
            cp(smeta[cid], os.path.join(D, *rec["meta"].split("/")))
        for sub in ("images", "labels"):
            for p in glob.glob(os.path.join(folder, sub, "*")):
                cp(p, os.path.join(D, "yolo", "pending_review", sub, os.path.basename(p)))
        for p in glob.glob(os.path.join(folder, "frames", "*")) + glob.glob(os.path.join(folder, "*.mp4")):
            cp(p, os.path.join(D, "review", "claude_tagged", os.path.basename(folder), os.path.relpath(p, folder)))
        rec.update(description="", alert=None, description_by=None, boxes_by="Claude (YOLO11x + ByteTrack), not reviewed yet",
                   num_persons=None, num_cars=None, needs_check="owner review of Claude's boxes", yolo_frames=int(r["frames_tagged"]))
        rows.append(rec)
    with open(os.path.join(D, "review", "claude_tagged", "review.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)

# review list for human re-checks
cp(os.path.join(Y, "review_for_humans.csv"), os.path.join(D, "review", "needs_human_check.csv"))

# master table + VLM list (no train/val split yet)
os.makedirs(os.path.join(D, "vlm"), exist_ok=True)
with open(os.path.join(D, "annotations", "clips.jsonl"), "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
with open(os.path.join(D, "vlm", "clips.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        if r["description"]:
            f.write(json.dumps(dict(clip_id=r["clip_id"], source=r["source"], video=r["clip"], crop=r["crop"],
                                    video_s3_path=f"{S3_ROOT}/{r['clip']}", vlm_crop_s3_path=f"{S3_ROOT}/{r['crop']}" if r["crop"] else None,
                                    description=r["description"], alert=r["alert"], camera=r["camera"],
                                    duration_sec=r["duration_sec"]), ensure_ascii=False) + chr(10))
c = collections.Counter(r["source"] for r in rows)
print("yolo frames:", len(frames_all), "hard negatives:", len(hard_neg))
print("clips:", len(rows), dict(c)); print("missing:", dict(missing))
print("alerts:", sum(1 for r in rows if r["alert"]), "with description:", sum(1 for r in rows if r["description"]))
