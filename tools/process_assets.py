"""Turn raw/*.png (stitched Gemini renders) into assets/*.webp.

Chroma-keys green (#00FF00-ish) and magenta backgrounds, despills edges,
splits sprite sheets into separate objects, and composites the real
KIDANA logo onto the blank plaque of the KIDANA building.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "raw", ROOT / "assets"
OUT.mkdir(exist_ok=True)
BRAND = ROOT / "brand"
LOGO = BRAND / "kidana_logo.png"


def keyness(a, key):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if key == "green":
        return g - np.maximum(r, b)
    if key == "blue":
        return b - np.maximum(r, g)
    return np.minimum(r, b) - g  # magenta


def chroma(img, key, fill=False, fill_max=0.0015):
    """Key out a flat green/magenta background.

    Matte from 'key-ness', then every edge pixel takes the colour of the
    nearest solid foreground pixel, so no green/magenta can bleed through.
    """
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    k = keyness(a, key)
    bgmask = k > np.percentile(k, 60)
    K = np.median(a[bgmask], axis=0)
    kK = float(keyness(K[None, None, :], key)[0, 0])
    t = k / max(kK, 1)
    hi, lo = {"green": (0.60, 0.30), "blue": (0.42, 0.16)}.get(key, (0.34, 0.10))
    alpha = np.clip((hi - t) / (hi - lo), 0, 1)
    alpha = ndimage.median_filter(alpha, size=3)
    alpha = ndimage.grey_erosion(alpha, size=(2, 2))
    alpha = ndimage.gaussian_filter(alpha, 0.6)
    holes = None
    if fill:
        # each object's enclosed "key-coloured" areas (glowing green digits) stay
        lab, n = ndimage.label(alpha > 0.5)
        filled = np.zeros_like(alpha, bool)
        for i, sl in enumerate(ndimage.find_objects(lab)):
            filled[sl] |= ndimage.binary_fill_holes(lab[sl] == i + 1)
        holes = filled & (alpha <= 0.5)
        hl, hn = ndimage.label(holes)
        if hn:
            hs = ndimage.sum(holes, hl, range(1, hn + 1))
            lum = a.mean(2)
            keep_ids = []
            for j, sl in enumerate(ndimage.find_objects(hl)):
                if hs[j] >= alpha.size * fill_max:
                    continue
                y0, x0 = max(sl[0].start - 4, 0), max(sl[1].start - 4, 0)
                sub = hl[y0:sl[0].stop + 4, x0:sl[1].stop + 4] == j + 1
                ring = ndimage.binary_dilation(sub, iterations=3) & ~sub
                if lum[y0:sl[0].stop + 4, x0:sl[1].stop + 4][ring].mean() < 150:
                    keep_ids.append(j + 1)  # hole inside a dark screen -> glowing digits
            holes = holes & np.isin(hl, keep_ids)
        holes = ndimage.binary_dilation(holes, iterations=1) & filled
        alpha = np.where(holes, 1.0, alpha)
    solid = alpha > 0.97
    solid = ndimage.binary_erosion(solid, iterations=1)
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    F = a[iy, ix]
    F = np.where(solid[..., None], a, F)
    # light global despill for any key tint that is left inside
    r, g, b = F[..., 0], F[..., 1], F[..., 2]
    if key == "green":
        F[..., 1] = np.minimum(g, np.maximum(r, b) + 40)
    elif key == "blue":
        near = ndimage.binary_dilation(alpha < 0.9, iterations=4) | (b > g)
        F[..., 2] = np.where(near, np.minimum(b, g * 0.7), np.minimum(b, np.maximum(r, g) + 20))
        F[..., 0] = np.where(near, np.minimum(r, g * 1.2 + 6), r)
    else:
        sp = np.clip(np.minimum(r, b) - g, 0, None)
        r, b = r - sp, b - sp
        near = ndimage.binary_dilation(alpha < 0.9, iterations=5)
        F[..., 0] = np.where(near, np.minimum(r, g * 1.05 + 4), r)
        F[..., 2] = np.where(near, np.minimum(b, g * 0.75), b)
    if holes is not None:
        F = np.where(holes[..., None], a, F)
    rgba = np.dstack([F, alpha * 255]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def trim(im, pad=6):
    al = np.asarray(im)[..., 3]
    ys, xs = np.where(al > 8)
    if not len(xs):
        return im
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad + 1, im.width), min(ys.max() + pad + 1, im.height)
    return im.crop((x0, y0, x1, y1))


def components(im, n_expected, merge=18, min_frac=0.002, rows=1):
    al = np.asarray(im)[..., 3] > 40
    grown = ndimage.binary_dilation(al, iterations=merge)
    lab, n = ndimage.label(grown)
    objs = ndimage.find_objects(lab)
    area = al.size
    items = []
    for i, sl in enumerate(objs):
        m = (lab[sl] == i + 1) & al[sl]
        if m.sum() < area * min_frac:
            continue
        items.append((sl, i + 1))
    if len(items) != n_expected:
        raise SystemExit(f"expected {n_expected} objects, found {len(items)}")
    # sort into rows then left->right
    items.sort(key=lambda t: t[0][0].start)
    per = n_expected // rows
    rowsorted = []
    for r in range(rows):
        chunk = items[r * per:(r + 1) * per]
        chunk.sort(key=lambda t: t[0][1].start)
        rowsorted += chunk
    res = []
    arr = np.asarray(im).copy()
    for sl, lid in rowsorted:
        sub = arr[sl].copy()
        mask = lab[sl] == lid
        sub[..., 3] = np.where(mask, sub[..., 3], 0)
        res.append(trim(Image.fromarray(sub, "RGBA")))
    return res


def keep_largest(im, merge=25):
    al = np.asarray(im)[..., 3] > 40
    lab, n = ndimage.label(ndimage.binary_dilation(al, iterations=merge))
    if n > 1:
        sizes = ndimage.sum(al, lab, range(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        arr = np.asarray(im).copy()
        arr[..., 3] = np.where(lab == keep, arr[..., 3], 0)
        im = Image.fromarray(arr, "RGBA")
    return trim(im)


def save(im, name, q=84, maxside=None):
    if maxside and max(im.size) > maxside:
        im = im.copy()
        im.thumbnail((maxside, maxside), Image.LANCZOS)
    im.save(OUT / f"{name}.webp", "WEBP", quality=q, method=6)
    return im.size


def kidana_logo(b):
    """Find the blank gold plaque and stamp the real KIDANA icon on it."""
    if not LOGO or not LOGO.exists():
        return b
    a = np.asarray(b).astype(np.float32)
    r, g, bl, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    gold = (al > 200) & (r > 150) & (r - bl > 70) & (g > 100) & (r - g > 15) & (r - g < 90)
    gold = ndimage.binary_opening(gold, iterations=2)
    lab, n = ndimage.label(gold)
    if not n:
        print("  ! plaque not found")
        return b
    sizes = ndimage.sum(gold, lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d = min(x1 - x0, y1 - y0)
    logo = Image.open(LOGO).convert("RGBA")
    la = np.asarray(logo)[..., 3]
    cols = np.where(la.max(0) > 0)[0]
    icon = logo.crop((2560, 0, logo.width, logo.height))
    icon = trim(icon, 0)
    # recolour to warm white
    ia = np.asarray(icon).copy()
    ia[..., 0], ia[..., 1], ia[..., 2] = 255, 248, 236
    icon = Image.fromarray(ia, "RGBA")
    w = int(d * 0.66)
    h = int(icon.height * w / icon.width)
    icon = icon.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    # slight transparency so it reads painted-on
    ia = np.asarray(icon).copy(); ia[..., 3] = (ia[..., 3] * 0.92).astype(np.uint8)
    icon = Image.fromarray(ia, "RGBA")
    out = b.copy()
    out.alpha_composite(icon, (int(cx - w / 2), int(cy - h / 2)))
    print(f"  logo stamped on plaque at ({cx:.0f},{cy:.0f}) d={d}")
    return out


def uqu_logo(b, label):
    """Replace the painted emblem on the gate's plaque with the real UQU icon."""
    icon_p = BRAND / "uqu_icon.png"
    if not icon_p.exists():
        return b
    a = np.asarray(b).astype(np.float32)
    r, g, bl, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    teal = (al > 200) & (g - r > 40) & (bl - r > 30)
    lab, n = ndimage.label(ndimage.binary_closing(teal, iterations=3))
    if not n:
        print("  ! uqu emblem not found", label); return b
    sizes = ndimage.sum(teal, lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    h, w = y1 - y0, x1 - x0
    # the painted crown sits above the teal shape: include it in the patch
    py0, py1 = max(0, int(y0 - h * .42)), int(y1 + h * .06)
    px0, px1 = int(x0 - w * .08), int(x1 + w * .08)
    # plaque colour = median of light pixels just around the patch
    ring = a[max(0, py0 - 6):py1 + 6, max(0, px0 - 6):px1 + 6]
    light = ring[(ring[..., :3].mean(2) > 200) & (ring[..., 3] > 200)]
    col = np.median(light[:, :3], axis=0) if len(light) else np.array([238, 234, 224])
    out = a.copy()
    patch = out[py0:py1, px0:px1]
    noise = np.random.default_rng(1).normal(0, 2.2, patch[..., :3].shape)
    patch[..., :3] = col + noise
    out = Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")
    icon = trim(Image.open(icon_p).convert("RGBA"), 0)
    ih = int((py1 - py0) * .92)
    iw = int(icon.width * ih / icon.height)
    if iw > (px1 - px0) * 1.1:
        iw = int((px1 - px0) * 1.1); ih = int(icon.height * iw / icon.width)
    icon = icon.resize((iw, ih), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.5))
    out.alpha_composite(icon, (int((px0 + px1) / 2 - iw / 2), int((py0 + py1) / 2 - ih / 2)))
    print(f"  uqu logo placed on {label} at ({px0},{py0})-({px1},{py1})")
    return out


def wasl_logo(b):
    """Put the Wasl wordmark on the cream band of the bus, under its windows."""
    lp = BRAND / "wasl_logo.png"
    if not lp.exists():
        return b
    a = np.asarray(b).astype(np.int32)
    r, g, bl, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    teal = (al > 200) & (g - r > 40) & (bl - r > 40) & (g > 90)
    teal = ndimage.binary_opening(teal, iterations=2)
    lab, n = ndimage.label(teal)
    if not n:
        print("  ! bus not found"); return b
    objs = ndimage.find_objects(lab)
    ys = [o[0] for o in objs]; xs = [o[1] for o in objs]
    by0 = min(y.start for y in ys); bx0 = min(x.start for x in xs); bx1 = max(x.stop for x in xs)
    band_h = max(y.stop for y in ys) - by0
    lh = int(band_h * .8)
    logo = trim(Image.open(lp).convert("RGBA"), 0)
    lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    cx = bx0 + (bx1 - bx0) * .30
    cy = by0 - band_h * .55
    out = b.copy()
    out.alpha_composite(logo, (int(cx - lw / 2), int(cy - lh / 2)))
    print(f"  wasl logo on bus at ({cx:.0f},{cy:.0f}) h={lh}")
    return out


def fill_holes(im):
    """Re-fill transparent holes inside an object (e.g. green digits on a screen)."""
    a = np.asarray(im).copy()
    solid = a[..., 3] > 128
    filled = ndimage.binary_fill_holes(solid)
    holes = filled & ~solid
    a[..., 3] = np.where(holes, 255, a[..., 3])
    return Image.fromarray(a, "RGBA"), holes


def tile_floor(im):
    """Blend left/right edges so the texture tiles horizontally."""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    bw = w // 8
    left, right = a[:, :bw].copy(), a[:, -bw:].copy()
    t = np.linspace(0, 1, bw)[None, :, None]
    out = a[:, :w - bw].copy()
    # left edge starts as the continuation of the (cropped) right edge
    out[:, :bw] = right * (1 - t) + left * t
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


manifest = {}

def rec(name, size):
    manifest[name] = {"w": size[0], "h": size[1]}
    print(f"{name:14s} {size}")


# backgrounds
for n in ["sky-day", "sky-sunset", "sky-night"]:
    rec(n, save(Image.open(RAW / f"{n}.png").convert("RGB"), n, 82))
rec("wall", save(Image.open(RAW / "wall.png").convert("RGB"), "wall", 82))
rec("map", save(Image.open(RAW / "map.png").convert("RGB"), "map", 84))
fl = tile_floor(Image.open(RAW / "floor.png"))
rec("floor", save(fl, "floor", 80))

# single cut-outs
rec("me", save(keep_largest(chroma(Image.open(RAW / "character2.png"), "green")), "me", 88))
rec("uqu", save(uqu_logo(keep_largest(chroma(Image.open(RAW / "uqu.png"), "green")), "uqu"), "uqu", 86))
kb = keep_largest(chroma(Image.open(RAW / "kidana.png"), "green"))
rec("kidana", save(kidana_logo(kb), "kidana", 86))
rec("wasl", save(wasl_logo(keep_largest(chroma(Image.open(RAW / "wasl.png"), "green"))), "wasl", 86))

# sheets
props = components(chroma(Image.open(RAW / "props.png"), "magenta"), 4)
for n, im in zip(["_palm_old", "_acacia_old", "frame", "table"], props):
    if not n.startswith("_"):
        rec(n, save(im, n, 86))
rec("palm", save(keep_largest(chroma(Image.open(RAW / "palm1.png"), "blue")), "palm", 86))
rec("acacia", save(keep_largest(chroma(Image.open(RAW / "acacia1.png"), "blue")), "acacia", 86))
proj = components(chroma(Image.open(RAW / "projects2.png"), "green", fill=True), 3, merge=40)
for n, im in zip(["p_securefile", "p_washroom", "p_meeting"], proj):
    rec(n, save(im, n, 86))
lm = components(chroma(Image.open(RAW / "landmarks.png"), "green"), 8, merge=6, rows=2)
names = ["m_uqu", "m_kidana", "m_wasl", "m_gallery", "m_bench", "m_mail", "ic_globe", "ic_map"]
for n, im in zip(names, lm):
    if n == "m_kidana":
        im = kidana_logo(im)
    if n == "m_uqu":
        im = uqu_logo(im, "m_uqu")
    if n == "m_bench":
        im = keep_largest(chroma(Image.open(RAW / "bench2.png"), "green", fill=True, fill_max=0.00015))
    rec(n, save(im, n, 86, maxside=520))

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)

# contact sheet on a checkerboard for eyeballing
items = [k for k in manifest if not k.startswith("sky") and k not in ("wall", "map", "floor")]
cell = 260
cols = 6
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGBA", (cols * cell, rows * (cell + 20)), (255, 255, 255, 255))
chk = Image.new("RGBA", (cell, cell))
cp = chk.load()
for y in range(cell):
    for x in range(cell):
        cp[x, y] = (200, 200, 200, 255) if (x // 16 + y // 16) % 2 else (235, 235, 235, 255)
for i, n in enumerate(items):
    im = Image.open(OUT / f"{n}.webp").convert("RGBA")
    im.thumbnail((cell - 10, cell - 10))
    c = chk.copy()
    c.alpha_composite(im, ((cell - im.width) // 2, (cell - im.height) // 2))
    sheet.paste(c, ((i % cols) * cell, (i // cols) * (cell + 20)))
(ROOT / "tools").mkdir(exist_ok=True)
sheet.convert("RGB").save(ROOT / "tools" / "preview.png")
print("ok")
