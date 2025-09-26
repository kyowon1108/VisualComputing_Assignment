# --- scripts/make_rgb_he_metrics.py ---
import os, json, csv, hashlib, argparse, pathlib
import numpy as np, cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_png(path, img):
    ensure_dir(pathlib.Path(path).parent)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])

def to_gray_u8(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return g

def rgb_channel_equalize(rgb):
    # equalize per R,G,B channel deterministically
    r,g,b = [rgb[...,i] for i in range(3)]
    r = cv2.equalizeHist(r.astype(np.uint8))
    g = cv2.equalizeHist(g.astype(np.uint8))
    b = cv2.equalizeHist(b.astype(np.uint8))
    out = np.stack([r,g,b], axis=-1)
    return out

def deltaE_maps(orig_rgb, proc_rgb):
    # inputs uint8 RGB; convert to float 0..1
    o = (orig_rgb/255.0).astype(np.float32)
    p = (proc_rgb/255.0).astype(np.float32)
    o_lab = rgb2lab(o)
    p_lab = rgb2lab(p)
    # full ΔE2000
    d_full = deltaE_ciede2000(o_lab, p_lab)
    # chroma-only: use original L*, processed a*,b* with original L*
    p_chroma = p_lab.copy()
    p_chroma[...,0] = o_lab[...,0]
    d_chroma = deltaE_ciede2000(o_lab, p_chroma)
    return d_full.astype(np.float32), d_chroma.astype(np.float32)

def save_map_gray(path, arr, vmin=None, vmax=None):
    # normalize 0..255 for stable saving without matplotlib
    a = arr.astype(np.float32)
    if vmin is None: vmin = float(np.min(a))
    if vmax is None: vmax = float(np.max(a))
    if vmax <= vmin: vmax = vmin + 1.0
    a = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    a = (a * 255.0).round().astype(np.uint8)
    rgb = np.stack([a,a,a], axis=-1)
    imwrite_png(path, rgb)

def ssim_map(orig_gray, proc_gray):
    # data_range=255 for uint8
    score, full = ssim(orig_gray, proc_gray, data_range=255, full=True)
    return score, (full * 255.0).round().astype(np.uint8)

def sha256(path):
    h=hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda:f.read(1<<16), b""):
            h.update(ch)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="images/he_dark_indoor.jpg")
    parser.add_argument("--out_base", default="results/he/result_rgb_he.png")
    parser.add_argument("--metrics_dir", default="results/he_metrics_fixed")
    parser.add_argument("--stats_csv", default="results/he_metrics_fixed/he_metrics_stats.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"

    ensure_dir("results/he"); ensure_dir(args.metrics_dir)

    # 1) Build RGB-HE result (or reuse)
    orig = imread_rgb(args.src)
    if (not os.path.exists(args.out_base)) or args.force:
        rgbhe = rgb_channel_equalize(orig)
        imwrite_png(args.out_base, rgbhe)
        # for compatibility with other scripts
        compat = "results/he/result_rgb_global.png"
        if not os.path.exists(compat):
            imwrite_png(compat, rgbhe)
        made_base = True
    else:
        rgbhe = imread_rgb(args.out_base); made_base = False

    # 2) Metrics (diff/ssim/deltaE/chroma)
    og = to_gray_u8(orig); pg = to_gray_u8(rgbhe)
    diff = cv2.absdiff(og, pg).astype(np.float32)
    save_map_gray(f"{args.metrics_dir}/diff_rgb_he.png", diff, vmin=0, vmax=255)

    ssim_score, ssim_full = ssim_map(og, pg)
    imwrite_png(f"{args.metrics_dir}/ssim_rgb_he.png", np.stack([ssim_full]*3, axis=-1))

    d_full, d_chr = deltaE_maps(orig, rgbhe)
    save_map_gray(f"{args.metrics_dir}/deltaE_rgb_he.png", d_full, vmin=0, vmax=50)
    save_map_gray(f"{args.metrics_dir}/deltaE_chroma_rgb_he.png", d_chr, vmin=0, vmax=50)

    # 3) Update stats CSV (append-or-merge)
    row = {
        "name":"rgb_he",
        "deltaE_mean": float(np.mean(d_full)),
        "deltaE_max":  float(np.max(d_full)),
        "deltaE_chroma_mean": float(np.mean(d_chr)),
        "ssim_score": float(ssim_score),
        "diff_mean":  float(np.mean(diff)),
    }
    # merge-safe write
    rows=[]; existed=False
    if os.path.exists(args.stats_csv):
        try:
            with open(args.stats_csv, newline="") as f:
                r=csv.DictReader(f); rows=list(r)
        except Exception:
            rows=[]
    # replace rgb_he if present
    out_rows=[]
    for r in rows:
        if r.get("name","")== "rgb_he":
            out_rows.append({**r, **row}); existed=True
        else:
            out_rows.append(r)
    if not existed:
        out_rows.append(row)
    # ensure field order
    fields=["name","deltaE_mean","deltaE_max","deltaE_chroma_mean","ssim_score","diff_mean"]
    ensure_dir(pathlib.Path(args.stats_csv).parent)
    with open(args.stats_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)

    # 4) Rebuild simple 3-up collages if the other two methods exist
    # (We will just check presence; if missing, skip silently.)
    def exists(p): return os.path.exists(p)
    trio = {
      "diff":   ["diff_rgb_he.png","diff_y_he.png","diff_clahe.png"],
      "ssim":   ["ssim_rgb_he.png","ssim_y_he.png","ssim_clahe.png"],
      "deltaE": ["deltaE_rgb_he.png","deltaE_y_he.png","deltaE_clahe.png"]
    }
    for kind, files in trio.items():
        paths = [f"{args.metrics_dir}/{f}" for f in files]
        if all(exists(p) for p in paths):
            imgs = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in paths]
            h = max(i.shape[0] for i in imgs); w = sum(i.shape[1] for i in imgs)
            canvas = np.zeros((h, w, 3), np.uint8)
            x=0
            for im in imgs:
                h2,w2=im.shape[:2]
                canvas[:h2, x:x+w2] = im; x+=w2
            out = f"{args.metrics_dir}/{kind}_collage.png"
            imwrite_png(out, canvas)

    result = {
      "task":"make_rgb_he_metrics",
      "base_image": args.out_base,
      "made_base": made_base,
      "artifacts":[
        f"{args.metrics_dir}/diff_rgb_he.png",
        f"{args.metrics_dir}/ssim_rgb_he.png",
        f"{args.metrics_dir}/deltaE_rgb_he.png",
        f"{args.metrics_dir}/deltaE_chroma_rgb_he.png"
      ]
    }
    print(json.dumps(result))
if __name__ == "__main__":
    main()
# --- end of file ---