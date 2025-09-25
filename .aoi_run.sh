# ==========================
# ALL-IN-ONE SUBMISSION FLOW
# ==========================
# 설정값(필요 시 수정)
BRANCH="feature/presentation-guide"
DO_CAPTURE_AUDIT=true      # 촬영 메타데이터 & 640×480 레터박스 생성
DO_MAKE_SLIDES=true        # slides/pdf 스크립트가 있으면 실행
DO_PACKAGE=true            # 제출 zip 생성
DO_CLEANUP=false           # results/ 불필요 파일 정리(화이트리스트 방식) ※ 기본 비활성
DO_PUSH=true               # git 커밋 & 푸시

set -euo pipefail
mkdir -p .aoi_tmp && rm -f .aoi_tmp/*.json

# 0) Git 동기화 + 의존성
git checkout "${BRANCH}"
git pull --rebase origin "${BRANCH}" || true

python3 -m pip install -U pip wheel >/dev/null 2>&1 || true
if [ -f requirements.txt ]; then
  python3 -m pip install -r requirements.txt >/dev/null 2>&1 || true
else
  python3 -m pip install opencv-python numpy scikit-image pillow matplotlib reportlab imageio imageio-ffmpeg tqdm exifread >/dev/null 2>&1 || true
fi

# 0-1) (선택) Git LFS (있으면 가볍게 설정)
if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
  if [ ! -f .gitattributes ] || ! grep -q '\\.mp4' .gitattributes 2>/dev/null; then
    git lfs track "*.mp4" "*.gif" || true
    echo "*.gif binary" >> .gitattributes || true
  fi
fi

# 0-2) 환경/버전 요약 JSON
python3 - <<'PY' > .aoi_tmp/env_sync.json
import json, subprocess, platform
def sh(*c):
    try: return subprocess.check_output(c, text=True).strip()
    except: return ""
try:
    import cv2, numpy as np, skimage
    v = {"opencv": cv2.__version__, "numpy": np.__version__, "skimage": skimage.__version__}
except Exception:
    v = {"opencv": "", "numpy": "", "skimage": ""}
j = {
  "task": "env_sync",
  "git": {"branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
         "commit": sh("git", "rev-parse", "--short", "HEAD"),
         "dirty": bool(sh("git", "status", "--porcelain"))},
  "versions": {"python": platform.python_version(), **v}
}
print(json.dumps(j, ensure_ascii=False))
PY

# 1) 촬영 메타데이터 & 640×480 레터박스
if [ "${DO_CAPTURE_AUDIT}" = true ]; then
python3 - <<'PY' > .aoi_tmp/capture_audit.json
import os, csv
from pathlib import Path
from PIL import Image, ExifTags
SRC=Path("images"); DST=Path("images_640x480"); DST.mkdir(exist_ok=True, parents=True)
csvp=Path("docs"); csvp.mkdir(exist_ok=True, parents=True)
csvf=csvp/"capture_metadata.csv"; mdf=csvp/"capture_metadata.md"
def get_exif(img):
    d={}
    try:
        ex=img.getexif()
        for k,v in ex.items():
            d[ExifTags.TAGS.get(k,k)]=v
    except: pass
    return d
rows=[["filename","width","height","ISO","ShutterSpeed","FNumber","WhiteBalance","FocalLength"]]
count=0
for p in sorted(SRC.glob("*.*")):
    if p.suffix.lower() not in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}: continue
    im=Image.open(p).convert("RGB"); ex=get_exif(im); w,h=im.size
    rows.append([p.name,w,h, ex.get("ISOSpeedRatings",""), ex.get("ShutterSpeedValue",""),
                 ex.get("FNumber",""), ex.get("WhiteBalance",""), ex.get("FocalLength","")])
    target=(640,480); r=min(target[0]/w, target[1]/h); nw,nh=int(w*r), int(h*r)
    im2=im.resize((nw,nh), Image.BICUBIC)
    canvas=Image.new("RGB", target, (0,0,0))
    canvas.paste(im2, ((target[0]-nw)//2,(target[1]-nh)//2))
    canvas.save(DST/p.name); count+=1
with open(csvf,"w",newline="") as f: csv.writer(f).writerows(rows)
with open(mdf,"w") as f:
    f.write("| filename | width | height | ISO | Shutter | FNumber | WB | Focal |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows[1:]: f.write("| "+" | ".join(map(str,r))+" |\n")
print({'task':'capture_audit','artifacts':['docs/capture_metadata.csv','docs/capture_metadata.md','images_640x480/'],'count_images':count})
PY
fi

# 2) 슬라이드/리포트 갱신(스크립트 있으면 실행)
if [ "${DO_MAKE_SLIDES}" = true ]; then
  [ -f scripts/make_slide_figs.py ] && python3 scripts/make_slide_figs.py || true
  [ -f scripts/make_pdf.py ] && python3 scripts/make_pdf.py || true
fi
python3 - <<'PY' > .aoi_tmp/slides_pdf_refresh.json
import json, os
arts = ["results/slides/he_summary.png","results/slides/otsu_summary.png","docs/final_report.pdf"]
print(json.dumps({"task":"slides_pdf_refresh","found":{a:os.path.exists(a) for a in arts}}, ensure_ascii=False))
PY

# 3) 제출 번들 패키징(zip)
if [ "${DO_PACKAGE}" = true ]; then
python3 - <<'PY' > .aoi_tmp/package.json
import os, zipfile, json, hashlib
from pathlib import Path
want = [
 "docs/final_report.pdf",
 "docs/capture_metadata.md",
 "results/slides/he_summary.png",
 "results/slides/otsu_summary.png",
 "results/video/he_sweep.mp4",
 "results/video/otsu_sweep.mp4",
 "results/he/compare_he_contact_sheet.png",
 "results/otsu/compare_otsu_contact_sheet.png",
]
for d in ["results/he_metrics","results/otsu_metrics","results/ablation"]:
    if os.path.isdir(d):
        for root,_,files in os.walk(d):
            for f in files: want.append(os.path.join(root,f))
want = [w for w in want if os.path.exists(w)]
Path("dist").mkdir(exist_ok=True, parents=True)
out = Path("dist/submission_bundle.zip")
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for w in want: z.write(w,w)
h = hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else ""
sz = out.stat().st_size if out.exists() else 0
print(json.dumps({"task":"package","artifact":str(out),"count":len(want),"size_bytes":sz,"sha256":h}, ensure_ascii=False))
PY
fi

# 4) 안전 정리(화이트리스트 기반)
if [ "${DO_CLEANUP}" = true ]; then
cat > keep_list.txt <<'EOF'
results/slides/he_summary.png
results/slides/otsu_summary.png
results/he/compare_he_contact_sheet.png
results/otsu/compare_otsu_contact_sheet.png
results/he_metrics/he_metrics_collage.png
results/otsu_metrics/compare_panel.png
results/video/he_sweep.mp4
results/video/otsu_sweep.mp4
results/ablation/ablation_he_top3.json
results/ablation/ablation_otsu_top3.json
docs/final_report.pdf
EOF

python3 - <<'PY' > .aoi_tmp/cleanup.json
import os, json
keep=set()
if os.path.exists("keep_list.txt"):
  with open("keep_list.txt") as f:
    for line in f:
      p=line.strip()
      if p and not p.startswith("#") and os.path.exists(p): keep.add(os.path.abspath(p))
deleted=0
for root,_,files in os.walk("results"):
  for fn in files:
    p=os.path.abspath(os.path.join(root,fn))
    if p not in keep and os.path.exists(p):
      os.remove(p); deleted+=1
empties=[]
for root,dirs,_ in os.walk("results", topdown=False):
  for d in dirs:
    dp=os.path.join(root,d)
    if not os.listdir(dp):
      os.rmdir(dp); empties.append(os.path.relpath(dp))
print(json.dumps({"task":"cleanup_apply","deleted_files":deleted,"removed_empty_dirs":len(empties)}, ensure_ascii=False))
PY
fi

# 5) 커밋 & 푸시
if [ "${DO_PUSH}" = true ]; then
  git add -A
  git commit -m "chore(submission): capture metadata + slides/report refresh + bundle" || true
  git push origin "${BRANCH}" || true
fi

# 6) 최종 통합 JSON 출력
python3 - <<'PY'
import json, glob, os

def load(p):
    try:
        with open(p,'r',encoding='utf-8') as f: return json.load(f)
    except: return None
res = {}
for name in ["env_sync","capture_audit","slides_pdf_refresh","package","cleanup","cleanup_apply","push"]:
    matches = glob.glob(f".aoi_tmp/{name}*.json")
    if matches:
        res[name] = load(matches[0])
    else:
        res[name] = None
import subprocess

def sh(*c):
    try: return subprocess.check_output(c, text=True).strip()
    except: return ""
res["push"] = res.get("push") or {"task":"push","branch":sh("git","rev-parse","--abbrev-ref","HEAD"),"commit":sh("git","rev-parse","--short","HEAD")}
print(json.dumps({"all_in_one":res}, ensure_ascii=False))
PY
