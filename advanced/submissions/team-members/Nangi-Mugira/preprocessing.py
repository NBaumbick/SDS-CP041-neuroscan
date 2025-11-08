import os, random, math, shutil, pathlib, csv
from collections import Counter, defaultdict
from PIL import Image
import numpy as np

# ========= EDIT THESE =========
BALANCED_DIR = r"Data/archive/brain_tumor_just_split"   # contains balanced_data/yes and balanced_data/no
OUT_DIR      = r"Data/archive/brain_tumor_prepped"    # will create train/val/test subfolders here
TARGET_SIZE  = (64, 64)                  # (width,height) to resize images to
SPLITS       = {"train":0.70, "val":0.15, "test":0.15}
SEED         = 42
SAVE_PLOTS   = False  # set True to save simple histograms as PNGs

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")

def list_images(folder):
    p = pathlib.Path(folder)
    return [str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in EXTS]

yes_dir = os.path.join(BALANCED_DIR, "yes")
no_dir  = os.path.join(BALANCED_DIR, "no")
yes_files = list_images(yes_dir)
no_files  = list_images(no_dir)

if not yes_files or not no_files:
    raise RuntimeError("No images found. Check BALANCED_DIR/yes and BALANCED_DIR/no paths.")

# 1) Inspect sizes and write a small report
def get_size(path):
    try:
        with Image.open(path) as im:
            return im.size  # (w,h)
    except Exception:
        return None

sizes = []
bad   = []
for f in yes_files + no_files:
    sz = get_size(f)
    if sz is None:
        bad.append(f)
    else:
        sizes.append(sz)

size_counts = Counter(sizes)
min_w = min(s[0] for s in sizes)
min_h = min(s[1] for s in sizes)
max_w = max(s[0] for s in sizes)
max_h = max(s[1] for s in sizes)

report_dir = os.path.join(OUT_DIR, "_reports")
os.makedirs(report_dir, exist_ok=True)
with open(os.path.join(report_dir, "size_report.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width","height","count"])
    for (w0,h0), c in sorted(size_counts.items(), key=lambda x: (-x[1], x[0])):
        w.writerow([w0,h0,c])

print(f"[INFO] Size report saved to {os.path.join(report_dir, 'size_report.csv')}")
print(f"[INFO] Min size: ({min_w},{min_h})  Max size: ({max_w},{max_h})  Unique sizes: {len(size_counts)}")
if bad:
    print(f"[WARN] {len(bad)} images could not be opened and will be skipped.")

# 2) Resize/standardize to TARGET_SIZE and save into a staging folder
stage_dir = os.path.join(OUT_DIR, "_stage_all")  # temporary
for cls in ["yes","no"]:
    os.makedirs(os.path.join(stage_dir, cls), exist_ok=True)

def resize_save(src_path, dst_path, size):
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = im.resize(size, Image.BILINEAR)
            im.save(dst_path, quality=95)
            return True
    except Exception:
        return False

def batch_resize(files, cls_name):
    ok = 0
    for i, src in enumerate(files, 1):
        dst = os.path.join(stage_dir, cls_name, os.path.basename(src))
        # avoid collisions
        base, ext = os.path.splitext(dst)
        c = 1
        while os.path.exists(dst):
            dst = f"{base}__{c}{ext}"
            c += 1
        if resize_save(src, dst, TARGET_SIZE):
            ok += 1
    print(f"[INFO] Resized {ok}/{len(files)} images for class '{cls_name}'.")

batch_resize(yes_files, "yes")
batch_resize(no_files,  "no")

# 3) Pixel intensity distribution (summary + optional hist plots)
def intensity_stats(sample_paths, sample_max=500):
    sample_paths = random.sample(sample_paths, min(len(sample_paths), sample_max))
    vals = []
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"), dtype=np.uint8)  # grayscale for simplicity
                vals.append(arr.flatten())
        except Exception:
            pass
    if not vals:
        return None
    x = np.concatenate(vals)
    return {
        "min": int(x.min()),
        "max": int(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "n_pixels": int(x.size)
    }

stats_yes = intensity_stats(list_images(os.path.join(stage_dir,"yes")))
stats_no  = intensity_stats(list_images(os.path.join(stage_dir,"no")))

with open(os.path.join(report_dir, "intensity_summary.txt"), "w") as f:
    f.write(f"TARGET_SIZE: {TARGET_SIZE}\n")
    f.write(f"YES stats: {stats_yes}\n")
    f.write(f"NO  stats: {stats_no}\n")
print(f"[INFO] Intensity summary saved to {os.path.join(report_dir, 'intensity_summary.txt')}")

if SAVE_PLOTS:
    import matplotlib.pyplot as plt
    def save_hist(paths, out_png, sample_max=500):
        paths = random.sample(paths, min(len(paths), sample_max))
        allv = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("L"), dtype=np.uint8)
                    allv.append(arr.flatten())
            except:
                pass
        if not allv:
            return
        x = np.concatenate(allv)
        plt.figure()
        plt.hist(x, bins=32)
        plt.title(os.path.basename(out_png).replace("_"," ").replace(".png",""))
        plt.xlabel("Pixel intensity (0-255)")
        plt.ylabel("Frequency")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

    save_hist(list_images(os.path.join(stage_dir,"yes")), os.path.join(report_dir, "hist_yes.png"))
    save_hist(list_images(os.path.join(stage_dir,"no")),  os.path.join(report_dir, "hist_no.png"))
    print(f"[INFO] Saved histograms to {report_dir}")

# 4) Split into train/val/test (stratified)
for sp in ["train","val","test"]:
    for cls in ["yes","no"]:
        os.makedirs(os.path.join(OUT_DIR, sp, cls), exist_ok=True)

def stratified_split(filepaths):
    random.shuffle(filepaths)
    n = len(filepaths)
    n_tr = math.floor(SPLITS["train"]*n)
    n_vl = math.floor(SPLITS["val"]*n)
    return {
        "train": filepaths[:n_tr],
        "val":   filepaths[n_tr:n_tr+n_vl],
        "test":  filepaths[n_tr+n_vl:]
    }

yes_paths = list_images(os.path.join(stage_dir, "yes"))
no_paths  = list_images(os.path.join(stage_dir, "no"))

ys = stratified_split(yes_paths)
ns = stratified_split(no_paths)

for sp in ["train","val","test"]:
    # YES
    for src in ys[sp]:
        dst = os.path.join(OUT_DIR, sp, "yes", os.path.basename(src))
        shutil.copy2(src, dst)
    # NO
    for src in ns[sp]:
        dst = os.path.join(OUT_DIR, sp, "no", os.path.basename(src))
        shutil.copy2(src, dst)

# final counts
def count_dir(d):
    return len([f for f in pathlib.Path(d).iterdir() if f.is_file() and f.suffix.lower() in EXTS])

final = defaultdict(dict)
for sp in ["train","val","test"]:
    final[sp]["yes"] = count_dir(os.path.join(OUT_DIR, sp, "yes"))
    final[sp]["no"]  = count_dir(os.path.join(OUT_DIR, sp, "no"))

print("[DONE] Prepped data at:", OUT_DIR)
for sp in ["train","val","test"]:
    print(f"  {sp}: yes={final[sp]['yes']}  no={final[sp]['no']}  total={final[sp]['yes']+final[sp]['no']}")
print(f"[NOTE] Images are resized to {TARGET_SIZE}. Normalize to [0,1] in your Keras pipeline (see below).")
