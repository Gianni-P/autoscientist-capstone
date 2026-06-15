import os, glob
from PIL import Image

# Pick one image
d = "/home/gdp/autoscientist/data/nih_chestxray14/images_001/images"
files = glob.glob(os.path.join(d, "*.png"))
if not files:
    print("no images")
    raise SystemExit(1)
p = files[0]
sz = os.path.getsize(p)
img = Image.open(p)
print(f"sample: {os.path.basename(p)}")
print(f"file size: {sz/1024:.1f} KB")
print(f"image dims: {img.size}, mode: {img.mode}")
# Median/avg size across ~200 images
sizes = [os.path.getsize(f) for f in files[:200]]
print(f"avg size over 200: {sum(sizes)/len(sizes)/1024:.1f} KB")
print(f"min/max: {min(sizes)/1024:.1f} / {max(sizes)/1024:.1f} KB")
