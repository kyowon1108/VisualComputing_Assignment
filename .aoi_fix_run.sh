# ================================
# REPAIR & FINALIZE ALL-IN-ONE RUN
# ================================
# 원하는 설정
BRANCH="feature/presentation-guide"
DO_CAPTURE_AUDIT=false      # 이미 수행했으므로 기본 false (필요하면 true)
DO_MAKE_SLIDES=true         # slides/pdf 갱신 재시도
DO_PACKAGE=true             # 제출 zip 재생성
DO_PUSH=true                # 커밋/푸시

set -euo pipefail
mkdir -p .aoi_fix && rm -f .aoi_fix/*.json

# 0) 브랜치 전환 & 미커밋 정리(안전 커밋) → rebase 동기화
git checkout "${BRANCH}"
git add -A
git commit -m "chore(wip): auto-save before rebase" || true
git pull --rebase origin "${BRANCH}" || true

# 1) 의존성 설치(누락 패키지 보강)
python3 -m pip install -U pip wheel >/dev/null 2>&1 || true
# 핵심 과목 스택 고정 폭
python3 -m pip install \
  "opencv-python>=4.10,<4.13" \
  "numpy>=2.0,<2.2" \
  "scikit-image>=0.25.0,<0.26" \
  "reportlab>=4.2,<5" \
  pillow matplotlib imageio imageio-ffmpeg exifread tqdm >/dev/null 2>&1 || true

# 버전 확인 JSON
python3 - <<'PY' > .aoi_fix/env.json
import json, platform
try:
  import cv2, numpy as np, skimage
  v = {"opencv": cv2.__version__, "numpy": np.__version__, "skimage": skimage.__version__}
except Exception as e:
  v = {"opencv":"(import failed)","numpy":"(import failed)","skimage":"(import failed)"}
print(json.dumps({"task":"env","python":platform.python_version(), **v}, ensure_ascii=False))
PY

# 2) (선택) 촬영 메타/640x480 재실행
if [ "${DO_CAPTURE_AUDIT}" = true ]; then
python3 - <<'PY' > .aoi_fix/capture.json
import os, csv
from pathlib import Path
from PIL import Image, ExifTags
SRC=Path("images"); DST=Path("images_640x480"); DST.mkdir(exist_ok=True, parents=True)
csvp=Path("docs"); csvp.mkdir(exist_ok=True, parents=True)
csvf=csvp/"capture_metadata.csv"; mdf=csvp/"capture_metadata.md"
def exif(img):
  d={}
  try:
    ex=img.getexif()
    for k,v in ex.items(): d[ExifTags.TAGS.get(k,k)]=v
  except: pass
  return d
rows=[["filename","width","height","ISO","ShutterSpeed","FNumber","WhiteBalance","FocalLength"]]
n=0
for p in sorted(SRC.glob("*.*")):
  if p.suffix.lower() not in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}: continue
  im=Image.open(p).convert("RGB"); e=exif(im); w,h=im.size
  rows.append([p.name,w,h,e.get("ISOSpeedRatings",""),e.get("ShutterSpeedValue",""),e.get("FNumber",""),e.get("WhiteBalance",""),e.get("FocalLength","")])
  T=(640,480); r=min(T[0]/w, T[1]/h); nw,nh=int(w*r), int(h*r)
  from PIL import Image as _I
  im2=im.resize((nw,nh), _I.BICUBIC)
  canvas=_I.new("RGB", T, (0,0,0)); canvas.paste(im2, ((T[0]-nw)//2,(T[1]-nh)//2)); canvas.save(DST/p.name); n+=1
with open(csvf,"w",newline="") as f: csv.writer(f).writerows(rows)
with open(mdf,"w") as f:
  f.write("| filename | width | height | ISO | Shutter | FNumber | WB | Focal |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
  for r in rows[1:]: f.write("| "+" | ".join(map(str,r))+" |\n")
print({"task":"capture_audit","count":n,"artifacts":["docs/capture_metadata.csv","docs/capture_metadata.md","images_640x480/"]})
PY
fi

# 3) 슬라이드/리포트 갱신 재시도
[ -f scripts/make_slide_figs.py ] && python3 scripts/make_slide_figs.py || true
[ -f scripts/make_pdf.py ] && python3 scripts/make_pdf.py || true
python3 - <<'PY' > .aoi_fix/slides.json
import os, json
found = {k: os.path.exists(k) for k in [
  "results/slides/he_summary.png",
  "results/slides/otsu_summary.png",
  "docs/final_report.pdf"
]}
print(json.dumps({"task":"slides_pdf_refresh","found":found}, ensure_ascii=False))
PY

# 4) 제출 번들 재패키징
python3 - <<'PY' > .aoi_fix/package.json
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

# 5) 커밋 & 푸시
if [ "${DO_PUSH}" = true ]; then
  git add -A
  git commit -m "fix(build): install skimage/reportlab; regenerate slides/pdf; repackage submission" || true
  git push origin "${BRANCH}" || true
fi

# 6) 최종 통합 JSON
python3 - <<'PY'
import json, subprocess, os

def sh(*c):
  try: return subprocess.check_output(c, text=True).strip()
  except: return ""

out = {
  "task":"repair_finalize",
  "git":{"branch": sh("git","rev-parse","--abbrev-ref","HEAD"),
         "commit": sh("git","rev-parse","--short","HEAD"),
         "dirty": bool(sh("git","status","--porcelain"))},
}
# merge sub-jsons
for f in ["env.json","capture.json","slides.json","package.json"]:
  p=os.path.join(".aoi_fix",f)
  if os.path.exists(p):
    try:
      with open(p,"r",encoding="utf-8") as fh: out.update(json.load(fh))
    except Exception as e:
      out.setdefault("errors",[]).append({"file":f,"msg":str(e)})
print(json.dumps(out, ensure_ascii=False))
PY
