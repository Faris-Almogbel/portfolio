"""Make assets/p_winapps.webp: the countdown monitor with its screen showing a small desktop of app windows.

The new screen is drawn flat, then warped onto the monitor's (slightly angled) screen so it sits inside the bezel.
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np, cv2
from scipy import ndimage

a = Image.open('assets/p_meeting.webp').convert('RGBA')
A = np.asarray(a).astype(int)
W, H = a.size

# the screen: largest dark area, reduced to its four corners
lum = A[..., :3].mean(-1)
dark = (lum < 70) & (A[..., 3] > 200)
lab, n = ndimage.label(dark)
k = int(np.argmax(ndimage.sum(dark, lab, range(1, n + 1)))) + 1
hull = cv2.convexHull(np.column_stack(np.where(lab == k))[:, ::-1].astype(np.int32))
quad = cv2.approxPolyDP(hull, 6, True).reshape(-1, 2).astype(np.float32)
c0 = quad.mean(0)
quad = c0 + (quad - c0) * 0.965                       # sit just inside the bezel
s = quad.sum(1); d_ = np.diff(quad, axis=1).ravel()
tl, br, tr, bl = quad[np.argmin(s)], quad[np.argmax(s)], quad[np.argmin(d_)], quad[np.argmax(d_)]

# flat screen content
FW, FH = 1000, 600
c = Image.new('RGBA', (FW, FH))
d = ImageDraw.Draw(c)
for yy in range(FH):
    t = yy / FH
    d.line([(0, yy), (FW, yy)], fill=(int(16 + 12 * t), int(70 - 34 * t), int(84 - 24 * t), 255))
glow = Image.new('L', (FW, FH)); ImageDraw.Draw(glow).ellipse([420, -80, 1120, 520], fill=120)
c = Image.composite(Image.new('RGBA', c.size, (58, 166, 160, 255)), c, glow.filter(ImageFilter.GaussianBlur(120)))
d = ImageDraw.Draw(c)

def win(x, y, w, h, body, title):
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=body, outline=(255, 255, 255, 70), width=3)
    d.rounded_rectangle([x, y, x + w, y + 50], radius=18, fill=title)
    d.rectangle([x, y + 30, x + w, y + 50], fill=title)
    for i, cc in enumerate([(255, 255, 255, 180), (255, 255, 255, 130), (255, 255, 255, 95)]):
        cx = x + w - 34 - i * 30
        d.ellipse([cx - 8, y + 17, cx + 8, y + 33], fill=cc)

win(70, 60, 470, 330, (234, 237, 239, 255), (30, 122, 120, 255))
for i, wd in enumerate([360, 290, 390, 250, 330]):
    d.rounded_rectangle([110, 100 + i * 46 + 40, 110 + wd, 100 + i * 46 + 56], radius=8, fill=(158, 170, 178, 255))
fx, fy = 410, 200
win(fx, fy, 520, 300, (20, 28, 38, 255), (30, 122, 120, 255))
f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 118)
txt = "15:00"; tw = d.textlength(txt, font=f); tx = fx + 260 - tw / 2; ty = fy + 78
gl = Image.new('RGBA', c.size); ImageDraw.Draw(gl).text((tx, ty), txt, font=f, fill=(90, 255, 140, 210))
c.alpha_composite(gl.filter(ImageFilter.GaussianBlur(12))); d = ImageDraw.Draw(c)
d.text((tx, ty), txt, font=f, fill=(150, 255, 180, 255))
d.rounded_rectangle([fx + 50, fy + 250, fx + 470, fy + 266], radius=8, fill=(58, 78, 88, 255))
d.rounded_rectangle([fx + 50, fy + 250, fx + 300, fy + 266], radius=8, fill=(90, 255, 140, 255))
d.rectangle([0, FH - 54, FW, FH], fill=(10, 22, 30, 235))
for i, cc in enumerate([(63, 166, 162), (236, 238, 240), (90, 255, 140), (240, 180, 80)]):
    cx = FW // 2 - 90 + i * 60
    d.rounded_rectangle([cx - 20, FH - 42, cx + 20, FH - 12], radius=8, fill=cc + (255,))
# screen depth: darker edges, like light falling off inside the glass
vig = Image.new('L', (FW, FH), 0); ImageDraw.Draw(vig).rectangle([0, 0, FW, FH], outline=255, width=60)
c = Image.composite(Image.new('RGBA', c.size, (6, 12, 18, 255)), c, vig.filter(ImageFilter.GaussianBlur(40)).point(lambda v: int(v * .55)))

# warp the flat screen onto the monitor
src = np.float32([[0, 0], [FW, 0], [FW, FH], [0, FH]])
dst = np.float32([tl, tr, br, bl])
S = 4
M = cv2.getPerspectiveTransform(src, dst * S)
big = cv2.warpPerspective(np.asarray(c), M, (W * S, H * S), flags=cv2.INTER_AREA)
mask = cv2.warpPerspective(np.full((FH, FW), 255, np.uint8), M, (W * S, H * S), flags=cv2.INTER_LINEAR)
scr = Image.fromarray(big).resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(.45))
msk = Image.fromarray(mask).resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(.6))
out = Image.composite(scr, a, msk)

# glass: a soft diagonal sheen and a faint top reflection, only on the screen
sh = Image.new('RGBA', (W, H)); sd = ImageDraw.Draw(sh)
sd.polygon([tuple(tl), (tl[0] + (tr[0] - tl[0]) * .42, tl[1] + (tr[1] - tl[1]) * .42), (bl[0] + (br[0] - bl[0]) * .12, bl[1] + (br[1] - bl[1]) * .12), tuple(bl)], fill=(255, 255, 255, 22))
sh = sh.filter(ImageFilter.GaussianBlur(3))
out = Image.composite(Image.alpha_composite(out, sh), out, msk)
out.save('assets/p_winapps.webp', quality=90)
