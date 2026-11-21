"""raw/me_cert.png: the APPROVED picture (face, beard, frame, hands untouched) continued down to the feet
with the plain thobe + sandals from a full-body render of the same outfit."""
from PIL import Image
import numpy as np
src = open('tools/process_assets.py').read()
exec(src.split('manifest = {}')[0])
A = np.asarray(keep_largest(chroma(Image.open('raw/me_approved.jpg'), "green", nogreen=True))).astype(np.float32)
B = keep_largest(chroma(Image.open('raw/me_cert_body.jpg'), "green", nogreen=True))
def agal(a):
    al = a[..., 3] > 128; dk = al & (a[..., :3].max(-1) < 60)
    ys, xs = np.where(dk[:120]); return xs.min(), xs.max(), ys.min()
ax0, ax1, ay = agal(A)
b0 = np.asarray(B).astype(np.float32); bx0, bx1, by = agal(b0)
s = (ax1 - ax0) / (bx1 - bx0)                       # same head size
B = B.resize((round(B.width * s), round(B.height * s)), Image.LANCZOS); b = np.asarray(B).astype(np.float32)
bx0, bx1, by = agal(b)
seamA = A.shape[0] - 55                               # a plain-thobe row near the bottom of the approved picture
seamB = by + (seamA - ay)
def span(a, y):
    r = np.where(a[y, :, 3] > 128)[0]; return r.min(), r.max()
la, ra = span(A, seamA); lb, rb = span(b, seamB)
kx = (ra - la) / (rb - lb)
H = seamA + (b.shape[0] - seamB)
W = int(max(A.shape[1], (b.shape[1]) * kx + abs(la - lb * kx)) + 40)
out = np.zeros((H, W, 4), np.float32)
offA = 20; out[:A.shape[0], offA:offA + A.shape[1]] = A
# bottom part: B rows from seamB-BL, stretched horizontally so the thobe width/centre match at the seam
BL = 50
low = Image.fromarray(b[seamB - BL:].astype(np.uint8), 'RGBA')
low = np.asarray(low.resize((round(low.width * kx), low.height), Image.LANCZOS)).astype(np.float32)
dx = int(round(offA + la - lb * kx)); y0 = seamA - BL
h = min(low.shape[0], H - y0); w = low.shape[1]
x_a, x_b = max(0, dx), min(W, dx + w)
seg = low[:h, x_a - dx:x_b - dx]
t = np.clip((np.arange(h) - 0) / BL, 0, 1)[:, None, None]   # 0 at top of blend → 1 below seam
cur = out[y0:y0 + h, x_a:x_b]
top_valid = np.arange(h)[:, None] < (A.shape[0] - y0)
t = np.where(top_valid[..., None], t, 1.0)
out[y0:y0 + h, x_a:x_b] = cur * (1 - t) + seg * t
res = Image.fromarray(out.clip(0, 255).astype(np.uint8), 'RGBA')
res.save('raw/me_cert.png')   # already cut out (RGBA)
print('cert composite', res.size, 'scale', round(s, 3), 'kx', round(kx, 3))
