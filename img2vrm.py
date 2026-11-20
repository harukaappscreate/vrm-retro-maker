# SPDX-License-Identifier: MIT
"""Turn a picture of a character (front view, e.g. an AI image) into a voxel VRM.

    picture -> cut out (plain background or alpha) -> pixel art (32-128 px tall, 16-256 colours)
    -> joints (guessed from the silhouette, or given) -> arms turned out to a T-pose
    -> each pixel becomes a column of voxels, thicker towards the middle of the body ("puff")
    -> one skinned mesh with a palette texture -> VRM 1.0 or VRM 0.x bytes

Only Pillow and NumPy.  Every step is a separate function so a tool can show and fix the result
(the joints especially) before the VRM is written.
"""
import io
import json
import struct

import numpy as np
from PIL import Image

NEAREST = 9728
SIDES = (0, 1)                 # 0 = left side of the picture (the character's right), 1 = right side of the picture


# ====================================================================== 1. cut out
def cut_out(img, tol=40, work=480, holes=True):
    """RGBA picture, background made transparent (alpha 0 / 255).
    - A picture that already has transparency keeps it.
    - A bright, saturated background (green / magenta "screen"): every pixel close to that colour goes,
      wherever it is (also between a ponytail and the head), and the screen colour is taken off the edges.
    - Any other plain background: what touches the border and is close to the border's usual colour goes;
      with `holes`, closed-in areas of almost exactly that colour go too (gaps under arms, between hair and head).
      Turn `holes` off for white clothes on a white background.
    The edge is decided again at full size and one pixel is taken off it, so no light fringe stays."""
    im = img.convert("RGBA")
    a = np.asarray(im)
    if (a[..., 3] < 128).mean() > 0.02:
        return im
    s = min(1.0, work / max(im.size))
    small = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BOX)
    rgb = np.asarray(small)[..., :3].astype(np.int32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(-1))
    near = dist < tol
    screen = bg.max() - bg.min() > 90                             # a coloured screen, not white / grey / paper
    if screen:
        fill = near
    else:
        fill = np.zeros_like(near)
        fill[0], fill[-1], fill[:, 0], fill[:, -1] = near[0], near[-1], near[:, 0], near[:, -1]
        while True:                                               # grow from the border through similar pixels
            g = fill.copy()
            g[1:] |= fill[:-1]
            g[:-1] |= fill[1:]
            g[:, 1:] |= fill[:, :-1]
            g[:, :-1] |= fill[:, 1:]
            g &= near
            if (g == fill).all():
                break
            fill = g
        if holes:                                                 # closed-in pockets of the background colour
            strict = (dist < tol * 0.5) & ~fill
            min_px = max(12, int(strict.size * 0.0003))           # bigger than an eye
            big = keep_big(strict, min_px=min_px)
            if big.sum() >= min_px:
                fill |= big
    # full size: re-decide the pixels near the edge by their own colour, then take one pixel off the figure
    soft = np.asarray(Image.fromarray(np.where(fill, 0, 255).astype(np.uint8)).resize(im.size, Image.BILINEAR))
    full = a[..., :3].astype(np.int32)
    fdist = np.sqrt(((full - bg) ** 2).sum(-1))
    solid = soft >= 128
    edge = (soft > 0) & (soft < 255)
    solid[edge] = fdist[edge] >= tol
    e = solid.copy()
    e[1:] &= solid[:-1]
    e[:-1] &= solid[1:]
    e[:, 1:] &= solid[:, :-1]
    e[:, :-1] &= solid[:, 1:]
    out = a.copy()
    out[..., 3] = np.where(e, 255, 0)
    if screen:                                                    # take the screen colour off the figure's edge
        k = int(np.argmax(bg))
        others = [c for c in range(3) if c != k]
        band = e & ~(np.roll(e, 2, 0) & np.roll(e, -2, 0) & np.roll(e, 2, 1) & np.roll(e, -2, 1))
        cap = out[..., others].max(-1)
        ch = out[..., k]
        out[..., k] = np.where(band & (ch > cap), cap, ch)
    return Image.fromarray(out, "RGBA")


# ====================================================================== 2. pixel art
def crop_figure(rgba, max_h=1024):
    """Cut to the figure's box and make it at most max_h tall (the working picture for the later steps).
    Loose crumbs (under 1% of the figure, e.g. the tip of a neighbour's finger) are dropped first."""
    a = np.asarray(rgba).copy()
    solid = a[..., 3] >= 128
    k = max(1, solid.shape[0] // 256)
    small = solid[::k, ::k]
    lab, sizes = labels(small)
    if len(sizes) > 1:
        keep = np.asarray([False] + [sz >= 0.01 * sum(sizes) for sz in sizes])
        big = np.repeat(np.repeat(keep[lab], k, 0), k, 1)[:solid.shape[0], :solid.shape[1]]
        pad = np.zeros_like(solid)
        pad[:big.shape[0], :big.shape[1]] = big
        # a pixel stays if its small cell (or a neighbouring one, for thin edges) is part of a kept piece
        near = pad | np.roll(pad, k, 0) | np.roll(pad, -k, 0) | np.roll(pad, k, 1) | np.roll(pad, -k, 1)
        a[..., 3] = np.where(near, a[..., 3], 0)
        rgba = Image.fromarray(a, "RGBA")
    ys, xs = np.nonzero(a[..., 3] >= 128)
    if len(ys) == 0:
        raise ValueError("no figure found (the whole picture looks like background)")
    crop = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    if crop.height > max_h:
        crop = crop.resize((max(1, round(crop.width * max_h / crop.height)), max_h), Image.BOX)
    return denoise(crop)


def denoise(rgba, size=5):
    """A median filter on the colours (JPEG noise and paper grain would otherwise become speckles of extra colours).
    Lines thinner than the filter are kept by the shrink anyway; the transparency is not touched."""
    from PIL import ImageFilter
    if size < 3:
        return rgba
    rgb = rgba.convert("RGB").filter(ImageFilter.MedianFilter(size))
    out = rgb.convert("RGBA")
    out.putalpha(rgba.getchannel("A"))
    return out


def shrink(crop, height):
    """Premultiplied shrink (the background colour does not bleed into the edge): (rgb uint8, solid mask)."""
    w = max(1, round(crop.width * height / crop.height))
    c = np.asarray(crop).astype(np.float32)
    pm = Image.fromarray(np.uint8(np.clip(c[..., :3] * (c[..., 3:] / 255.0), 0, 255)))
    small_rgb = np.asarray(pm.resize((w, height), Image.BOX)).astype(np.float32)
    small_a = np.asarray(crop.getchannel("A").resize((w, height), Image.BOX)).astype(np.float32)
    rgb = np.uint8(np.clip(small_rgb / np.maximum(small_a[..., None], 1) * 255.0, 0, 255))
    return rgb, keep_big(small_a >= 128)


def palette_of(rgb, solid, colors):
    seen = rgb[solid]
    side = int(np.ceil(np.sqrt(len(seen))))
    flat = np.zeros((side * side, 3), np.uint8)
    flat[:len(seen)] = seen
    flat[len(seen):] = seen[0]
    return Image.fromarray(flat.reshape(side, side, 3)).quantize(colors=max(2, min(256, colors)),
                                                                 method=Image.Quantize.MEDIANCUT)


def to_index(rgb, solid, pal_img):
    """Map to the palette: (index array with -1 for empty, list of the colours used, renumbered)."""
    q = np.asarray(Image.fromarray(rgb).quantize(palette=pal_img, dither=Image.Dither.NONE)).astype(np.int32)
    pal = np.asarray(pal_img.getpalette()[:768], np.uint8).reshape(-1, 3)
    used = sorted(set(q[solid].tolist()))
    remap = -np.ones(256, np.int32)
    remap[used] = np.arange(len(used))
    return np.where(solid, remap[q], -1), [tuple(int(v) for v in pal[u]) for u in used]


def pixelate(crop, height=64, colors=32):
    """The pixel art of the figure as it is (before the T-pose): (index array, palette)."""
    rgb, solid = shrink(crop, height)
    return to_index(rgb, solid, palette_of(rgb, solid, colors))


def keep_main(m):
    """Only the biggest connected part."""
    return keep_big(m, min_px=10 ** 9)


def labels(m):
    """Connected parts (4-neighbours): (label array, 0 = empty; list of sizes, label k has sizes[k - 1])."""
    lab = np.zeros(m.shape, np.int32)
    sizes = []
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0]:
            continue
        n = len(sizes) + 1
        stack, cnt = [(y0, x0)], 0
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            cnt += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        sizes.append(cnt)
    return lab, sizes


def keep_big(m, min_px=6):
    """Drop specks: keep connected parts with at least min_px pixels (and always the biggest)."""
    lab, sizes = labels(m)
    if not sizes:
        return m
    big = max(sizes)
    keep = [0] + [1 if (s >= min_px or s == big) else 0 for s in sizes]
    return np.asarray(keep, bool)[lab] & m


# ====================================================================== 3. joints
def runs(row):
    """[(start, end_exclusive), ...] of filled pixels in a row."""
    r, x, n = [], 0, len(row)
    while x < n:
        if row[x]:
            s = x
            while x < n and row[x]:
                x += 1
            r.append((s, x))
        else:
            x += 1
    return r


def guess_joints(idx):
    """Joints in pixel coordinates (x right, y down) from the silhouette of a standing figure.
    Only a first guess: a tool should let the user move them."""
    m = idx >= 0
    H, W = m.shape
    cx = float(np.nonzero(m)[1].mean())
    ci = int(round(cx))
    # neck: the narrowest middle run between 12% and 40% of the height (fallback 20%)
    def mid_width(y):
        for s, e in runs(m[y]):
            if s <= ci < e:
                return e - s
        return 0
    ws = [(mid_width(y), y) for y in range(int(H * 0.12), int(H * 0.40))]
    head_w = max([mid_width(y) for y in range(0, int(H * 0.2))] or [1])
    wmin, ny = min(ws) if ws else (0, int(H * 0.2))
    neck_y = ny if 0 < wmin < 0.75 * head_w else int(H * 0.2)
    # crotch: the top of the longest stretch of empty centre in the lower half (the gap between the legs;
    # the shoes may touch below it).  A long dress has no gap: then 55% of the height.
    best, run_top, y = (0, int(H * 0.55)), None, H - 1
    for y in range(int(H * 0.45), H):
        if not m[y, ci]:
            run_top = y if run_top is None else run_top
            if y - run_top + 1 > best[0]:
                best = (y - run_top + 1, run_top)
        else:
            run_top = None
    crotch_y = best[1] if best[0] >= H * 0.08 else int(H * 0.55)
    crotch_y = int(min(max(crotch_y, H * 0.42), H * 0.8))
    ankle_y = int(H - 1 - max(1, H * 0.035))
    # legs: centre of the pixels on each side below the crotch
    leg_x = []
    for side in SIDES:
        xs = [x for yy in range(crotch_y, ankle_y + 1) for x in np.nonzero(m[yy])[0]
              if (x < cx if side == 0 else x >= cx)]
        leg_x.append(float(np.mean(xs)) if xs else cx + (-1 if side == 0 else 1) * W * 0.12)
    # arms: rows between shoulders and crotch with at least three runs -> the outer runs are arms
    shoulder_y = int(neck_y + max(1, H * 0.06))
    arm_pts = {0: [], 1: []}
    for yy in range(shoulder_y + int(H * 0.05), min(H, crotch_y + int(H * 0.10))):
        r = runs(m[yy])
        if len(r) >= 3 and max(r[0][1] - r[0][0], r[-1][1] - r[-1][0]) <= max(3, W * 0.2):
            arm_pts[0].append(((r[0][0] + r[0][1] - 1) / 2, yy, r[0][1] - r[0][0]))
            arm_pts[1].append(((r[-1][0] + r[-1][1] - 1) / 2, yy, r[-1][1] - r[-1][0]))
    J = {"neck": [cx, neck_y], "crotch": [cx, crotch_y]}
    body_half = None
    r = runs(m[min(H - 1, shoulder_y + int(H * 0.04))])
    for s, e in r:
        if s <= ci < e:
            body_half = (e - s) / 2
    body_half = body_half or W * 0.2
    arm_w = []
    for side in SIDES:
        sgn = -1 if side == 0 else 1
        pts = arm_pts[side]
        if len(pts) >= 3:
            hand_y = max(p[1] for p in pts)
            top = pts[0]
            arm_w.append(float(np.median([p[2] for p in pts])))
            sx = cx + sgn * max(abs(top[0] - cx) - arm_w[-1] * 0.5, body_half * 0.55)
            wx = [p[0] for p in pts if p[1] >= hand_y - H * 0.08]
            wrist = [float(np.mean(wx)), hand_y - H * 0.06]
        else:                                                          # arms against the body: assume the outer edge
            aw = max(2.0, H * 0.06)
            arm_w.append(aw)
            sx = cx + sgn * (body_half - aw * 0.5)
            wrist = [cx + sgn * (body_half - aw * 0.3), min(crotch_y - H * 0.02, shoulder_y + H * 0.33)]
        sh = [sx, shoulder_y]
        J["shoulder%d" % side] = sh
        J["elbow%d" % side] = [sh[0] + (wrist[0] - sh[0]) * 0.47, sh[1] + (wrist[1] - sh[1]) * 0.47]
        J["wrist%d" % side] = wrist
        J["hip%d" % side] = [leg_x[side], crotch_y - H * 0.03]
        J["knee%d" % side] = [leg_x[side], (crotch_y + ankle_y) / 2]
        J["ankle%d" % side] = [leg_x[side], ankle_y]
    J["arm_width"] = [float(np.mean(arm_w)), 0]
    tpose_arms(m, J)
    return J


def tpose_arms(m, J):
    """If the arms already stretch out to the sides (a T-pose picture), put the arm joints along them."""
    H, W = m.shape
    xs_all = np.nonzero(m.any(0))[0]
    col = m.sum(0)
    core = np.nonzero(col > 0.5 * col.max())[0]
    if not len(core) or not len(xs_all):
        return
    t0, t1 = core.min(), core.max() + 1                                # the body (many filled pixels per column)
    if min(t0 - xs_all.min(), xs_all.max() + 1 - t1) < H * 0.12:      # nothing reaches far out: arms hang down
        return
    for side in SIDES:
        tip_x = xs_all.min() if side == 0 else xs_all.max()
        inner = tip_x + (2 if side == 0 else -2)
        def band(x):
            ys = np.nonzero(m[:, int(min(max(x, 0), W - 1))])[0]
            ys = ys[ys < H * 0.6]
            if not len(ys):
                return None
            # the run nearest to the shoulders' height
            r = runs(m[:, int(min(max(x, 0), W - 1))])
            r = [q for q in r if q[0] < H * 0.6]
            q = min(r, key=lambda q: abs((q[0] + q[1]) / 2 - J["shoulder%d" % side][1]))
            return (q[0] + q[1]) / 2.0, q[1] - q[0]
        b = band(inner)
        if b is None:
            continue
        sx = t0 if side == 0 else t1
        mid_x = (sx + tip_x) / 2.0
        bm = band(mid_x) or b
        y_arm = bm[0]
        reach = abs(tip_x - sx)
        J["shoulder%d" % side] = [float(sx + (1 if side == 0 else -1) * 0.5), float(y_arm)]
        J["elbow%d" % side] = [float(sx + (tip_x - sx) * 0.5), float(y_arm)]
        J["wrist%d" % side] = [float(sx + (tip_x - sx) * 0.82), float(b[0])]
        J["arm_width"] = [float(max(1.5, bm[1])), 0]
        J["tpose"] = [1, 0]


# ====================================================================== 4. T-pose
def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1e-9
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def arm_tip(J, side):
    E, Wr = np.asarray(J["elbow%d" % side], np.float64), np.asarray(J["wrist%d" % side], np.float64)
    d = Wr - E
    n = np.linalg.norm(d) or 1.0
    return Wr + d / n * max(2.0, n * 0.55)


def arm_masks(solid, J, f=1.0):
    """Which pixels of a picture (f times the pixel-art size) belong to each arm:
    near the line shoulder -> elbow -> wrist -> hand tip, on its own side of the neck, not above the shoulder."""
    H, W = solid.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    r = max(1.0, J["arm_width"][0] * 0.75 + 1.0) * f
    out = []
    for side in SIDES:
        S, E, Wr = (np.asarray(J["%s%d" % (k, side)], np.float32) * f for k in ("shoulder", "elbow", "wrist"))
        tip = arm_tip(J, side).astype(np.float32) * f
        dist = np.minimum.reduce([_seg_dist(xx, yy, S, E), _seg_dist(xx, yy, E, Wr), _seg_dist(xx, yy, Wr, tip) - 0.6 * f])
        m = (dist < r) & solid
        m &= (xx < J["neck"][0] * f) if side == 0 else (xx >= J["neck"][0] * f)
        m &= yy >= S[1] - r
        out.append(m)
    return out


def t_pose(crop, J, height, colors):
    """t_pose_rgb + its own palette: (index array, palette, joints, part array)."""
    rgb, solid, J2, p2 = t_pose_rgb(crop, J, height)
    idx, pal = to_index(rgb, solid, palette_of(rgb, solid, colors))
    p2[idx < 0] = -1
    return idx, pal, J2, p2


def t_pose_rgb(crop, J, height):
    """Turn each arm (as one straight piece) around its shoulder until it points straight out, on the big picture
    (so the arms stay clean), then make the pixel art.  Returns (index array, palette, joints, part array:
    -1 body, 0 / 1 the arm of that side) - all in pixel-art coordinates."""
    a = np.asarray(crop).copy()
    Hh, Wh = a.shape[:2]
    f = Hh / float(height)
    solid = a[..., 3] >= 128
    masks = arm_masks(solid, J, f)
    removed = masks[0] | masks[1]
    body = solid & ~removed
    body &= keep_main(body)                                        # crumbs of the arms that were not caught
    # what the arm covered in front of the body: fill from the nearest body pixel of the same row
    xs = np.arange(Wh)
    left = np.maximum.accumulate(np.where(body, xs, -1), axis=1)
    right = np.minimum.accumulate(np.where(body, xs, Wh)[:, ::-1], axis=1)[:, ::-1]
    hole = removed & (left >= 0) & (right < Wh)
    pick = np.where(xs - left <= right - xs, left, right)
    rows = np.nonzero(hole)
    rgba = np.zeros_like(a)
    rgba[body] = a[body]
    rgba[rows] = a[rows[0], pick[rows]]
    rgba[rows[0], rows[1], 3] = 255
    # the arms, turned, on a wider canvas
    turn = []
    for side in SIDES:
        S = np.asarray(J["shoulder%d" % side], np.float64) * f
        v = arm_tip(J, side) * f - S
        L = float(np.linalg.norm(v)) or 1.0
        ang = (0.0 if side == 1 else np.pi) - np.arctan2(v[1], v[0])
        turn.append((S, ang, L))
    pad = int(np.ceil(max(t[2] for t in turn) + 2))
    W2 = Wh + 2 * pad
    out = np.zeros((Hh, W2, 4), np.uint8)
    part = -np.ones((Hh, W2), np.int32)
    out[:, pad:pad + Wh] = rgba
    part[:, pad:pad + Wh][rgba[..., 3] >= 128] = -1
    yy, xx = np.mgrid[0:Hh, 0:W2].astype(np.float64) + 0.5
    Jh = {k: [v[0] * f + pad, v[1] * f] for k, v in J.items() if k not in ("arm_width", "tpose")}
    for side, m, (S, ang, L) in zip(SIDES, masks, turn):
        Sx, Sy = S[0] + pad, S[1]
        c, s = np.cos(-ang), np.sin(-ang)                          # from the new place back to the old one
        qx, qy = xx - Sx, yy - Sy
        ix = np.floor(c * qx - s * qy + Sx - pad).astype(int)
        iy = np.floor(s * qx + c * qy + Sy).astype(int)
        ok = (ix >= 0) & (ix < Wh) & (iy >= 0) & (iy < Hh)
        hit = np.zeros_like(ok)
        hit[ok] = m[iy[ok], ix[ok]]
        out[hit] = a[iy[hit], ix[hit]]
        part[hit] = side
        c2, s2 = np.cos(ang), np.sin(ang)
        for k in ("elbow", "wrist"):
            x0, y0 = Jh["%s%d" % (k, side)][0] - Sx, Jh["%s%d" % (k, side)][1] - Sy
            Jh["%s%d" % (k, side)] = [c2 * x0 - s2 * y0 + Sx, s2 * x0 + c2 * y0 + Sy]
        Jh["tip%d" % side] = [Sx + (L if side == 1 else -L), Sy]
    cols = np.nonzero((out[..., 3] >= 128).any(0))[0]
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    out, part = out[:, x0:x1], part[:, x0:x1]
    big = Image.fromarray(out, "RGBA")
    rgb, solid2 = shrink(big, height)
    # which arm each pixel belongs to: the majority of its area
    w = rgb.shape[1]
    frac = [np.asarray(Image.fromarray(np.uint8((part == sd) * 255)).resize((w, height), Image.BOX)) for sd in SIDES]
    p2 = -np.ones(solid2.shape, np.int32)
    p2[(frac[0] >= 128) & solid2] = 0
    p2[(frac[1] >= 128) & solid2] = 1
    fx = w / float(x1 - x0)
    J2 = {k: [(v[0] - x0) * fx, v[1] / f] for k, v in Jh.items()}
    J2["arm_width"] = list(J["arm_width"])
    if "tpose" in J:
        J2["tpose"] = list(J["tpose"])
    return rgb, solid2, J2, p2


# ====================================================================== 5. bones for each pixel
BONES = ["hips", "spine", "chest", "neck", "head",
         "rightUpperArm", "rightLowerArm", "rightHand", "leftUpperArm", "leftLowerArm", "leftHand",
         "rightUpperLeg", "rightLowerLeg", "rightFoot", "leftUpperLeg", "leftLowerLeg", "leftFoot"]
PARENT = {"hips": None, "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
          "rightUpperArm": "chest", "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm",
          "leftUpperArm": "chest", "leftLowerArm": "leftUpperArm", "leftHand": "leftLowerArm",
          "rightUpperLeg": "hips", "rightLowerLeg": "rightUpperLeg", "rightFoot": "rightLowerLeg",
          "leftUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "leftFoot": "leftLowerLeg"}
SIDE_NAME = {0: "right", 1: "left"}          # picture-left is the character's right


def bone_points(J):
    """Bone heads in pixel coordinates (after the T-pose)."""
    neck, crotch = J["neck"], J["crotch"]
    hip_y = (J["hip0"][1] + J["hip1"][1]) / 2
    cx = neck[0]
    P = {"hips": [cx, hip_y]}
    P["spine"] = [cx, hip_y + (neck[1] - hip_y) * 0.3]
    P["chest"] = [cx, hip_y + (neck[1] - hip_y) * 0.65]
    P["neck"] = [cx, neck[1] + 1.0]
    P["head"] = [cx, neck[1] - 0.5]
    for side in SIDES:
        n = SIDE_NAME[side]
        P[n + "UpperArm"] = J["shoulder%d" % side]
        P[n + "LowerArm"] = J["elbow%d" % side]
        P[n + "Hand"] = J["wrist%d" % side]
        P[n + "UpperLeg"] = J["hip%d" % side]
        P[n + "LowerLeg"] = J["knee%d" % side]
        P[n + "Foot"] = J["ankle%d" % side]
    return P


def pixel_bones(idx, J, part):
    """Bone number (index into BONES) for every pixel."""
    H, W = idx.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    P = bone_points(J)
    b = np.full((H, W), BONES.index("chest"), np.int32)
    neck_y, crotch_y = J["neck"][1], J["crotch"][1]
    hip_y = P["hips"][1]
    b[yy < neck_y] = BONES.index("head")
    b[(yy >= P["spine"][1]) & (yy < hip_y)] = BONES.index("spine")
    b[yy >= hip_y] = BONES.index("hips")
    legs = yy >= crotch_y
    for side in SIDES:
        n = SIDE_NAME[side]
        mine = legs & ((xx < J["neck"][0]) if side == 0 else (xx >= J["neck"][0]))
        b[mine] = BONES.index(n + "UpperLeg")
        b[mine & (yy >= J["knee%d" % side][1])] = BONES.index(n + "LowerLeg")
        b[mine & (yy >= J["ankle%d" % side][1])] = BONES.index(n + "Foot")
        arm = part == side
        sx = J["shoulder%d" % side][0]
        dist = np.abs(xx - sx)
        b[arm] = BONES.index(n + "UpperArm")
        b[arm & (dist >= abs(J["elbow%d" % side][0] - sx))] = BONES.index(n + "LowerArm")
        b[arm & (dist >= abs(J["wrist%d" % side][0] - sx))] = BONES.index(n + "Hand")
    return b


# ====================================================================== 6. voxels -> mesh
def thickness(idx, part, depth=0.7, max_frac=0.12):
    """Half thickness (in voxels) of each column: distance to the edge of its own piece, times `depth`."""
    H = idx.shape[0]
    hmax = max(1, int(round(H * max_frac)))
    d = edge_distance(idx, part, hmax * 3)
    h = np.clip(np.round(d * depth), 1, hmax).astype(np.int32)
    return np.where(idx >= 0, h, 0)


def edge_distance(idx, part, limit=10 ** 6):
    """Steps to the edge of its own piece (body, right arm, left arm) for every filled pixel (1 = on the edge)."""
    d = np.zeros(idx.shape, np.int32)
    for piece in (part == -1, part == 0, part == 1):
        m = (idx >= 0) & piece
        k = 0
        while m.any() and k < limit:
            k += 1
            d[m] = k
            e = m.copy()
            e[1:] &= m[:-1]
            e[:-1] &= m[1:]
            e[:, 1:] &= m[:, :-1]
            e[:, :-1] &= m[:, 1:]
            e[0] = e[-1] = False
            e[:, 0] = e[:, -1] = False
            m = e
    return d


def build_mesh(idx, h, bones, scale, cx):
    """Faces of the voxel columns. Picture x -> +X (the character's left on picture-right, facing +Z),
    picture y -> -Y with the feet at Y = 0.  One colour per column, so the side faces of a column are merged."""
    H, W = idx.shape
    P, N, C, B, I = [], [], [], [], []

    def quad(corners, normal, col, bone):
        base = len(P)
        P.extend(corners)
        N.extend([normal] * 4)
        C.extend([col] * 4)
        B.extend([bone] * 4)
        I.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    def X(x):
        return (x - cx) * scale

    def Y(y):
        return (H - y) * scale

    for y in range(H):
        for x in range(W):
            k = idx[y, x]
            if k < 0:
                continue
            t = h[y, x]
            bn = bones[y, x]
            z0, z1 = -t * scale, t * scale
            x0, x1, y0, y1 = X(x), X(x + 1), Y(y + 1), Y(y)        # y0 is the lower edge
            quad([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1), k, bn)      # front
            quad([(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)], (0, 0, -1), k, bn)     # back
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xn, yn = x + dx, y + dy
                tn = h[yn, xn] if 0 <= xn < W and 0 <= yn < H else 0
                if tn >= t:
                    continue
                spans = [(-t, t)] if tn == 0 else [(tn, t), (-t, -tn)]
                for a, b_ in spans:
                    za, zb = a * scale, b_ * scale
                    if dx == -1:
                        quad([(x0, y0, za), (x0, y0, zb), (x0, y1, zb), (x0, y1, za)], (-1, 0, 0), k, bn)
                    elif dx == 1:
                        quad([(x1, y0, zb), (x1, y0, za), (x1, y1, za), (x1, y1, zb)], (1, 0, 0), k, bn)
                    elif dy == -1:                                  # the row above: the top face
                        quad([(x0, y1, zb), (x1, y1, zb), (x1, y1, za), (x0, y1, za)], (0, 1, 0), k, bn)
                    else:                                           # the row below: the bottom face
                        quad([(x0, y0, za), (x1, y0, za), (x1, y0, zb), (x0, y0, zb)], (0, -1, 0), k, bn)
    return (np.asarray(P, np.float32), np.asarray(N, np.float32), np.asarray(C, np.int32),
            np.asarray(B, np.int32), np.asarray(I, np.uint32))


# ====================================================================== 6b. several views (a turnaround sheet)
VIEWS = ("front", "right", "back", "left")      # "right" = seen from the character's right side (she faces right)


def split_sheet(cut, gap_frac=0.008):
    """A sheet with the figures side by side -> [(x0, x1), ...] column ranges, one per figure.
    Figures may touch (a fingertip on the next figure's hair): they are split at the thinnest column
    between their bodies (the columns with many filled pixels)."""
    a = np.asarray(cut)[..., 3] >= 128
    col = a.sum(0).astype(np.float64)
    k = max(3, int(a.shape[1] * 0.006))
    sm = np.convolve(col, np.ones(k) / k, "same")
    core = np.nonzero(sm > 0.35 * sm.max())[0]
    if not len(core):
        return [(0, a.shape[1])]
    gap = max(4, int(a.shape[1] * gap_frac))
    cores, s0, p = [], core[0], core[0]
    for x in core[1:]:
        if x > p + gap:
            cores.append((s0, p))
            s0 = x
        p = x
    cores.append((s0, p))
    cuts = [0] + [int(c0[1] + np.argmin(sm[c0[1]:c1[0]])) for c0, c1 in zip(cores, cores[1:])] + [a.shape[1]]
    parts = list(zip(cuts, cuts[1:]))

    def height(p):
        ys = np.nonzero(a[:, p[0]:p[1]].any(1))[0]
        return (ys.max() - ys.min() + 1) if len(ys) else 0
    # a real figure is (nearly) full height; a hanging arm or a leg is not: join it to the narrower neighbour
    while len(parts) > 1:
        hs = [height(p) for p in parts]
        k = int(np.argmin(hs))
        if hs[k] >= 0.75 * max(hs):
            break
        if k == 0:
            j = 1
        elif k == len(parts) - 1:
            j = k - 1
        else:
            j = k - 1 if parts[k - 1][1] - parts[k - 1][0] <= parts[k + 1][1] - parts[k + 1][0] else k + 1
        lo, hi = min(k, j), max(k, j)
        parts[lo:hi + 1] = [(parts[lo][0], parts[hi][1])]
    return parts


def sheet_figures(cut, work=640, shave=3):
    """The figures of a sheet as separate RGBA pictures (same size as `cut`, the other figures see-through).
    Parts that do not touch go whole to the figure whose body they overlap (a hand reaching over the next
    figure's column stays with its own body); only parts that touch two bodies are cut at split_sheet's columns."""
    cols = split_sheet(cut)
    a = np.asarray(cut)
    solid = a[..., 3] >= 128
    if len(cols) == 1:
        return [cut]
    s = min(1.0, work / float(solid.shape[1]))
    h2, w2 = max(1, int(solid.shape[0] * s)), max(1, int(solid.shape[1] * s))
    small = np.asarray(Image.fromarray(np.uint8(solid * 255)).resize((w2, h2), Image.BOX)) > 0
    # a fingertip touching the next figure's hair is a thin contact: shave the outline
    # so such contacts break (`shave` times), label the parts, then let the labels grow back over what was shaved
    core_m = small.copy()
    for _ in range(shave):
        e = core_m.copy()
        e[1:] &= core_m[:-1]
        e[:-1] &= core_m[1:]
        e[:, 1:] &= core_m[:, :-1]
        e[:, :-1] &= core_m[:, 1:]
        core_m = e
    lab, sizes = labels(core_m)
    for _ in range(shave * 3):
        grow = lab.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(lab, dy, 0), dx, 1)
            grow = np.where((grow == 0) & small & (nb > 0), nb, grow)
        lab = grow
    col = solid.sum(0).astype(np.float64)
    owner = np.zeros(len(sizes) + 1, np.int32) - 1                 # component -> figure (-1: cut by columns)
    xs_of = [(int(x0 * s), max(int(x0 * s) + 1, int(x1 * s))) for x0, x1 in cols]
    for k in range(1, len(sizes) + 1):
        ys, xs = np.nonzero(lab == k)
        # which figures' bodies (their thickest columns) does this part cover?
        hits = []
        for f, (x0, x1) in enumerate(cols):
            c = col[x0:x1]
            core = np.nonzero(c > 0.35 * c.max())[0] + x0 if c.max() > 0 else np.array([], int)
            cs = set((core * s).astype(int).tolist())
            if cs.intersection(xs.tolist()):
                hits.append(f)
        if len(hits) == 1:
            owner[k] = hits[0]
        elif not hits:                                              # a loose bit: the figure whose columns hold it
            mx = xs.mean()
            owner[k] = next((f for f, (x0, x1) in enumerate(xs_of) if x0 <= mx < x1), 0)
    big_lab = np.asarray(Image.fromarray(lab.astype(np.int32)).resize((solid.shape[1], solid.shape[0]), Image.NEAREST))
    out = []
    xx = np.arange(solid.shape[1])
    for f, (x0, x1) in enumerate(cols):
        mine = owner[big_lab] == f
        mine |= (owner[big_lab] == -1) & ((xx >= x0) & (xx < x1))[None, :]
        mine &= solid
        b = a.copy()
        b[..., 3] = np.where(mine, a[..., 3], 0)
        out.append(Image.fromarray(b, "RGBA"))
    return out


def faces_right(crop):
    """Side view: does the figure face right?  The feet point forward: compare the feet with the shins above them."""
    a = np.asarray(crop)[..., 3] >= 128
    H = a.shape[0]
    def cx(y0, y1):
        ys, xs = np.nonzero(a[int(H * y0):int(H * y1)])
        return xs.mean() if len(xs) else 0.0
    return cx(0.95, 1.0) >= cx(0.78, 0.88)


def guess_roles(crops):
    """Front / back for the wide figures (arms out), right / left for the narrow ones, from the order and the feet."""
    widths = [c.width / float(c.height) for c in crops]
    wide = [i for i, w in enumerate(widths) if w > 0.6 * max(widths)]
    roles = [None] * len(crops)
    for n, i in enumerate(wide[:2]):
        roles[i] = ("front", "back")[n]
    for i, c in enumerate(crops):
        if roles[i] is None:
            r = "right" if faces_right(c) else "left"
            roles[i] = r if r not in roles else None
    if "front" not in roles:
        roles[0] = "front"
    return roles


def core_center(solid):
    """Middle of the body columns (where many pixels are filled), ignoring outstretched arms and hair tips."""
    col = solid.sum(0).astype(np.float64)
    c = np.nonzero(col > 0.5 * col.max())[0]
    return (c.min() + c.max() + 1) / 2.0 if len(c) else solid.shape[1] / 2.0


def leg_center(solid):
    """Side view: middle of the legs between knee and ankle height (the body's middle front to back)."""
    H = solid.shape[0]
    xs = np.nonzero(solid[int(H * 0.78):int(H * 0.9)])[1]
    return (xs.min() + xs.max() + 1) / 2.0 if len(xs) else solid.shape[1] / 2.0


def voxel_mesh(V, color, bones, scale, cx, z0):
    """Faces of a voxel grid V[y, x, z] (y down, z = the index - z0 is the depth, + is the front).
    color(direction, ys, xs, zs) gives the colour index of each face.  Vectorised: one pass per direction."""
    H, W, D = V.shape
    pad = np.pad(V, 1)
    dirs = {  # name: (neighbour offset in (y, x, z), outward normal in world axes, corners as (dx, dy_up, dz))
        "front": ((0, 0, 1), (0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        "back": ((0, 0, -1), (0, 0, -1), [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]),
        "left": ((0, 1, 0), (1, 0, 0), [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]),      # +X: the character's left
        "right": ((0, -1, 0), (-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        "top": ((-1, 0, 0), (0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]),
        "bottom": ((1, 0, 0), (0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    }
    Ps, Ns, Cs, Bs = [], [], [], []
    bpad = np.pad(bones, 1, constant_values=-1)
    for name, (off, nrm, corners) in dirs.items():
        nb = pad[1 + off[0]:1 + off[0] + H, 1 + off[1]:1 + off[1] + W, 1 + off[2]:1 + off[2] + D]
        # a face also where the neighbour moves with another bone: bent joints show a closed side, not a hole
        other = (bpad[1 + off[0]:1 + off[0] + H, 1 + off[1]:1 + off[1] + W] != bones)[:, :, None]
        ys, xs, zs = np.nonzero(V & (~nb | other))
        if not len(ys):
            continue
        base = np.stack([xs, H - 1 - ys, zs - z0], 1).astype(np.float32)
        quad = base[:, None, :] + np.asarray(corners, np.float32)[None]        # (n, 4, 3) in voxel units
        quad[..., 0] -= cx
        Ps.append((quad * scale).reshape(-1, 3))
        Ns.append(np.repeat(np.asarray([nrm], np.float32), len(ys) * 4, 0))
        Cs.append(np.repeat(color(name, ys, xs, zs), 4))
        Bs.append(np.repeat(bones[ys, xs], 4))
    P = np.concatenate(Ps)
    n = len(P) // 4
    I = (np.arange(n, dtype=np.uint32)[:, None] * 4 + np.asarray([0, 1, 2, 0, 2, 3], np.uint32)).reshape(-1)
    return P, np.concatenate(Ns), np.concatenate(Cs).astype(np.int32), np.concatenate(Bs).astype(np.int32), I


def coarse_groups(pal, k):
    """Colour groups ("parts": hair, skin, clothes ...): the palette reduced to k colours; palette index -> group."""
    if k <= 1 or len(pal) <= k:
        return np.arange(len(pal))
    img = Image.fromarray(np.asarray(pal, np.uint8)[None])
    q = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return np.asarray(q)[0].astype(np.int32)


def shape_by_labels(idx, pal, part, occ, W, D, zmax, round_p=2.4, groups=6):
    """The rounded cross-sections, one per colour group.  Each row of the side view is cut into stretches of
    one group (hair behind, tunic, face in front ...).  The stretch across the middle of the body takes its
    width from the front view's body; any other stretch takes the width of the same group in the picture
    that looks at it (the back for things behind, the front for things in front), so a ponytail gets the
    ponytail's width, not the body's.  Not found there: the rows nearby, then a round strand."""
    H = idx["front"].shape[0]
    grp = coarse_groups(pal, groups)
    def glab(a):
        return np.where(a >= 0, grp[np.maximum(a, 0)], -1)
    body = (idx["front"] >= 0) & (part < 0)
    labF = np.where(body, glab(idx["front"]), -1)
    labB = None
    if "back" in idx:
        sb = idx["back"] >= 0
        cb, cf = core_center(sb), core_center(idx["front"] >= 0)
        fb = np.floor(cb - (np.arange(W) + 0.5 - cf)).astype(int)
        ok = (fb >= 0) & (fb < idx["back"].shape[1])
        labB = -np.ones((H, W), np.int32)
        labB[:, ok] = glab(idx["back"])[:, fb[ok]]
        labB[~body] = -1
    side = sides_first = None
    for r in ("right", "left"):
        if r in idx:
            side = r
            break
    # the side view's groups along depth: same sampling as `occ`
    sv = idx[side]
    c = leg_center(sv >= 0)
    zc = np.arange(D) - zmax + 0.5
    xs = np.floor(c + zc if side == "right" else c - zc).astype(int)
    okz = (xs >= 0) & (xs < sv.shape[1])
    labS = -np.ones((H, D), np.int32)
    labS[:, okz] = glab(sv)[:, xs[okz]]
    labS[~occ] = -1
    cx_body = core_center(body | (idx["front"] >= 0))
    xg, zg = np.arange(W) + 0.5, np.arange(D) + 0.5
    R = np.zeros((H, W, D), bool)

    def ell(y, x0, x1, z0, z1):
        ax, cxe = (x1 - x0) / 2.0, (x0 + x1) / 2.0
        az, cz = (z1 - z0) / 2.0, (z0 + z1) / 2.0
        ex = np.abs((xg[x0:x1] - cxe) / max(ax, 0.5)) ** round_p
        ez = np.abs((zg[z0:z1] - cz) / max(az, 0.5)) ** round_p
        R[y, x0:x1, z0:z1] |= (ex[:, None] + ez[None, :]) <= 1.0

    def group_runs(lab, y, L):
        if lab is None or not (0 <= y < H):
            return []
        return runs(lab[y] == L)

    for y in range(H):
        for z0, z1 in runs(occ[y]):
            # stretches of one group, the short ones (outlines, 1 px) joined to the one before
            segs = []
            z = z0
            while z < z1:
                L = labS[y, z]
                e = z
                while e < z1 and labS[y, e] == L:
                    e += 1
                if segs and (e - z < 2 or segs[-1][1] - segs[-1][0] < 2):
                    segs[-1] = (segs[-1][0], e, segs[-1][2] if segs[-1][1] - segs[-1][0] >= e - z else L)
                else:
                    segs.append((z, e, L))
                z = e
            mid = [k for k, (a_, b_, _) in enumerate(segs) if a_ <= zmax < b_]
            main = mid[0] if mid else max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
            front_runs = runs(body[y])
            for k, (a_, b_, L) in enumerate(segs):
                if k == main:                                          # the body itself: every piece seen in front
                    xr = front_runs
                else:
                    look = labB if (a_ + b_) / 2.0 < zmax else labF
                    if look is None:
                        look = labF
                    xr = []
                    for dy in (0, 1, -1, 2, -2, 3, -3):
                        xr = group_runs(look, y + dy, L)
                        if xr:
                            break
                    if not xr:                                          # a round strand in the middle
                        w = max(1, b_ - a_)
                        x0 = int(round(cx_body - w / 2.0))
                        xr = [(max(0, x0), min(W, x0 + w))]
                for x0, x1 in xr:
                    ell(y, x0, x1, a_, b_)
    return R


def build_views(crops, roles, J, height, colors, meters=1.5, depth=0.7, meta=None, version="1.0", reach=2.5,
                round_p=2.4, groups=6):
    """Several views of one figure in a T-pose (crops of the same figure: front, and any of back / right / left)
    -> (VRM bytes, info, extra).  The front gives x and y and the joints; a side view gives the depth of every row
    (the shape is what both views agree on); the back and the sides give the colours of those faces."""
    by = {r: c for c, r in zip(crops, roles) if r}
    rgb_f, solid_f, J2, part = t_pose_rgb(by["front"], J, height)
    H, W = solid_f.shape
    rgbs = {"front": (rgb_f, solid_f)}
    for r in ("back", "right", "left"):
        if r in by:
            rgbs[r] = shrink(by[r], height)
    pal_img = palette_of(np.concatenate([v[0][v[1]] for v in rgbs.values()])[None], np.ones((1, sum(int(v[1].sum()) for v in rgbs.values())), bool), colors)
    idx = {}
    pal = None
    allpix = np.concatenate([v[0][v[1]] for v in rgbs.values()])
    q, pal = to_index(allpix[None], np.ones((1, len(allpix)), bool), pal_img)
    n0 = 0
    for r, (rgb, solid) in rgbs.items():
        k = int(solid.sum())
        a = -np.ones(solid.shape, np.int32)
        a[solid] = q[0, n0:n0 + k]
        idx[r] = a
        n0 += k
    part[idx["front"] < 0] = -1
    bones = pixel_bones(idx["front"], J2, part)
    cx = J2["neck"][0]
    d = edge_distance(idx["front"], part)
    # depth range: wide enough for the side views
    sides = [r for r in ("right", "left") if r in idx]
    zmax = int(np.ceil(max([idx[r].shape[1] for r in sides] + [H * 0.3]))) + 2
    D = 2 * zmax
    zc = np.arange(D) - zmax + 0.5                                     # depth of each layer, + = front
    arm = part >= 0
    lim_body = np.where(arm, np.maximum(1, np.round(d * depth)), d * reach + 1.0)
    V = (idx["front"] >= 0)[:, :, None] & (np.abs(zc)[None, None, :] <= lim_body[:, :, None])
    side_x, occ_of = {}, {}
    for r in sides:
        sv = idx[r] >= 0
        c = leg_center(sv)
        xs = np.floor(c + zc if r == "right" else c - zc).astype(int)  # "right" faces right: + depth is to the right
        side_x[r] = xs
        ok = (xs >= 0) & (xs < sv.shape[1])
        occ = np.zeros((H, D), bool)
        occ[:, ok] = sv[:, xs[ok]]
        occ_of[r] = occ
    if sides:
        V &= shape_by_labels(idx, pal, part, occ_of[sides[0]], W, D, zmax, round_p, groups) | arm[:, :, None]
    else:                                                              # no side view: the puff thickness
        h = thickness(idx["front"], part, depth)
        V = (idx["front"] >= 0)[:, :, None] & (np.abs(zc)[None, None, :] <= h[:, :, None])
    fb = None
    if "back" in idx:
        sb = idx["back"] >= 0
        cb, cf = core_center(sb), core_center(idx["front"] >= 0)
        fb = np.floor(cb - (np.arange(W) + 0.5 - cf)).astype(int)      # front column -> back column (mirrored)

    def look(r, ys, cols):
        a = idx[r]
        ok = (cols >= 0) & (cols < a.shape[1])
        out = -np.ones(len(ys), np.int32)
        out[ok] = a[ys[ok], cols[ok]]
        return out

    # colours by what each picture really sees: the front picture colours the cube nearest to it in its column,
    # the back picture the farthest one, a side picture the outermost one in its row; the rest takes a neighbour's
    ys_all, xs_all, zs_all = np.nonzero(V)
    seen = {}                                                          # view -> colour grid (-1 = not seen)

    def first_hit(axis, last):
        """Index along `axis` of the first (or last) filled cube, for every line of V."""
        n = V.shape[axis]
        ar = np.arange(n).reshape([-1 if a == axis else 1 for a in range(3)])
        if last:
            return np.where(V, ar, -1).max(axis)
        return np.where(V, ar, n).min(axis)

    def grid(fill_lines, cols):
        g = -np.ones(V.shape, np.int32)
        ys, xs, zs = fill_lines
        g[ys, xs, zs] = cols
        return g

    zf, zb = first_hit(2, True), first_hit(2, False)
    yy, xx = np.nonzero(zf >= 0)
    seen["front"] = grid((yy, xx, zf[yy, xx]), idx["front"][yy, xx])
    if fb is not None:
        cb_ = look("back", yy, fb[xx])
        cb_ = np.where(cb_ >= 0, cb_, idx["front"][yy, xx])
        seen["back"] = grid((yy, xx, zb[yy, xx]), cb_)
    for r, (axis_last) in (("right", False), ("left", True)):
        src = r if r in side_x else ("left" if r == "right" else "right")
        if src not in side_x:
            continue
        xe = first_hit(1, axis_last)
        yy2, zz2 = np.nonzero((xe >= 0) & (xe < W))
        cs = look(src, yy2, side_x[src][zz2])
        keep = cs >= 0
        seen[r] = grid((yy2[keep], xe[yy2[keep], zz2[keep]], zz2[keep]), cs[keep])
    base = -np.ones(V.shape, np.int32)
    for r in ("back", "right", "left", "front"):                      # the front wins where several see a cube
        if r in seen:
            base = np.where(seen[r] >= 0, seen[r], base)
    for _ in range(D + W):                                             # cubes no picture sees: from a neighbour
        miss = V & (base < 0)
        if not miss.any():
            break
        for ax in range(3):
            for sh in (1, -1):
                nb = np.roll(base, sh, ax)
                take = miss & (nb >= 0)
                base[take] = nb[take]
                miss &= ~take
    base[V & (base < 0)] = np.broadcast_to(idx["front"][:, :, None], V.shape)[V & (base < 0)]

    def color(name, ys, xs, zs):
        c = base[ys, xs, zs].copy()
        if name in seen:                                               # this face looks at that picture
            v = seen[name][ys, xs, zs]
            c = np.where(v >= 0, v, c)
        return c

    scale = meters / H
    mesh = voxel_mesh(V, color, bones, scale, cx, zmax)
    P = bone_points(J2)
    bone_pos = {bn: ((p[0] - cx) * scale, (H - p[1]) * scale, 0.0) for bn, p in P.items()}
    m = dict(DEFAULT_META)
    m.update(meta or {})
    data = write_vrm(mesh, bone_pos, pal, thumbnail(idx["front"], pal), m, version)
    info = {"pixels": [int(W), int(H)], "colors": len(pal), "tris": int(len(mesh[4]) // 3), "verts": int(len(mesh[0])),
            "bytes": len(data), "version": version, "views": sorted(idx), "depth_px": int(V.any((0, 1)).sum())}
    return data, info, (idx, pal, J2, part, bones, V)


# ====================================================================== 7. VRM
DEFAULT_META = {
    "name": "Retro avatar", "version": "1.0", "authors": ["unknown"],
    "avatarPermission": "onlyAuthor", "commercialUsage": "personalNonProfit", "creditNotation": "required",
    "allowRedistribution": False, "modification": "prohibited",
    "allowExcessivelyViolentUsage": False, "allowExcessivelySexualUsage": False,
    "allowPoliticalOrReligiousUsage": False, "allowAntisocialOrHateUsage": False,
}


def meta_1(m, thumb_image):
    out = {"name": m["name"], "version": m.get("version", "1.0"), "authors": list(m.get("authors") or ["unknown"]),
           "licenseUrl": "https://vrm.dev/licenses/1.0/", "thumbnailImage": thumb_image}
    for k in ("avatarPermission", "commercialUsage", "creditNotation", "allowRedistribution", "modification",
              "allowExcessivelyViolentUsage", "allowExcessivelySexualUsage", "allowPoliticalOrReligiousUsage",
              "allowAntisocialOrHateUsage"):
        out[k] = m.get(k, DEFAULT_META[k])
    for k in ("copyrightInformation", "contactInformation", "otherLicenseUrl"):
        if m.get(k):
            out[k] = m[k]
    return out


def meta_0(m, thumb_texture):
    """The same choices in VRM 0.x words (0.x has fewer of them; anything not allowed there stays strict)."""
    user = {"onlyAuthor": "OnlyAuthor", "onlySeparatelyLicensedPerson": "ExplicitlyLicensedPerson",
            "everyone": "Everyone"}[m.get("avatarPermission", "onlyAuthor")]
    if not m.get("allowRedistribution", False):
        lic = "Redistribution_Prohibited"
    elif m.get("modification", "prohibited") == "prohibited":
        lic = "CC_BY_ND" if m.get("creditNotation") == "required" else "Other"
    else:
        lic = "CC_BY" if m.get("creditNotation") == "required" else "CC0"
    return {"title": m["name"], "version": m.get("version", "1.0"), "author": ", ".join(m.get("authors") or ["unknown"]),
            "contactInformation": m.get("contactInformation", ""), "reference": "",
            "texture": thumb_texture, "allowedUserName": user,
            "violentUssageName": "Allow" if m.get("allowExcessivelyViolentUsage") else "Disallow",
            "sexualUssageName": "Allow" if m.get("allowExcessivelySexualUsage") else "Disallow",
            "commercialUssageName": "Disallow" if m.get("commercialUsage", "personalNonProfit") == "personalNonProfit" else "Allow",
            "otherPermissionUrl": "", "licenseName": lic,
            "otherLicenseUrl": "https://vrm.dev/licenses/1.0/" if lic == "Other" else ""}


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    return b.getvalue()


def thumbnail(idx, palette, size=256):
    H, W = idx.shape
    pal = np.asarray(palette + [(0, 0, 0)], np.uint8)
    rgb = pal[np.where(idx >= 0, idx, len(palette))]
    a = np.where(idx >= 0, 255, 0).astype(np.uint8)
    im = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    s = max(1, (size - 16) // max(H, W))
    im = im.resize((W * s, H * s), Image.NEAREST)
    out = Image.new("RGBA", (size, size), (43, 51, 95, 255))
    out.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return out


def write_vrm(mesh, bone_pos, palette, thumb, meta, version="1.0"):
    """mesh = (positions, normals, colour index, bone index, indices) in VRM 1.0 axes; bone_pos = {bone: (x, y, z)}."""
    P, Nn, C, B, I = mesh
    if version == "0.x":                                       # 0.x faces -Z: turn everything half a turn
        flip = np.array([-1, 1, -1], np.float32)
        P, Nn = P * flip, Nn * flip
        bone_pos = {k: tuple(np.asarray(v) * flip) for k, v in bone_pos.items()}
        # the winding stays right: a half turn is a rotation, not a mirror
    side = 16
    pal_im = Image.new("RGB", (side, side), (0, 0, 0))
    pal_im.putdata([tuple(c) for c in palette] + [(0, 0, 0)] * (side * side - len(palette)))
    uv = np.stack([((C % side) + 0.5) / side, ((C // side) + 0.5) / side], 1).astype(np.float32)
    joints = np.zeros((len(B), 4), np.uint16)
    joints[:, 0] = B
    weights = np.zeros((len(B), 4), np.float32)
    weights[:, 0] = 1.0

    binb = bytearray()
    views, accs = [], []

    def add_view(data, target=None):
        while len(binb) % 4:
            binb.append(0)
        v = {"buffer": 0, "byteOffset": len(binb), "byteLength": len(data)}
        if target:
            v["target"] = target
        binb.extend(data)
        views.append(v)
        return len(views) - 1

    def add_acc(arr, ctype, typ, target=None, minmax=False):
        vi = add_view(arr.tobytes(), target)
        a = {"bufferView": vi, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = arr.min(0).tolist()
            a["max"] = arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    a_pos = add_acc(P, 5126, "VEC3", 34962, True)
    a_nrm = add_acc(Nn, 5126, "VEC3", 34962)
    a_uv = add_acc(uv, 5126, "VEC2", 34962)
    a_j = add_acc(joints, 5123, "VEC4", 34962)
    a_w = add_acc(weights, 5126, "VEC4", 34962)
    big = int(I.max()) >= 65535
    a_i = add_acc(I.astype(np.uint32 if big else np.uint16).reshape(-1, 1), 5125 if big else 5123, "SCALAR", 34963)
    accs[a_i]["count"] = int(len(I))
    # nodes: bones (no rotation: the rest pose is the T-pose), then the mesh
    nodes = []
    for bn in BONES:
        p = np.asarray(bone_pos[bn], np.float64)
        par = PARENT[bn]
        t = p - (np.asarray(bone_pos[par], np.float64) if par else 0)
        nodes.append({"name": bn, "translation": [float(v) for v in t]})
    for bn in BONES:
        kids = [BONES.index(c) for c in BONES if PARENT[c] == bn]
        if kids:
            nodes[BONES.index(bn)]["children"] = kids
    ibm = np.zeros((len(BONES), 16), np.float32)
    for k, bn in enumerate(BONES):
        M = np.eye(4, dtype=np.float32)
        M[:3, 3] = -np.asarray(bone_pos[bn], np.float32)
        ibm[k] = M.T.reshape(-1)                                # column-major
    a_ibm = add_acc(ibm, 5126, "MAT4")
    mesh_node = len(nodes)
    nodes.append({"name": "body", "mesh": 0, "skin": 0})
    v_pal = add_view(png_bytes(pal_im))
    v_th = add_view(png_bytes(thumb))
    while len(binb) % 4:
        binb.append(0)
    j = {
        "asset": {"version": "2.0", "generator": "VRM RETRO MAKER (haruka_apps)"},
        "scene": 0, "scenes": [{"nodes": [BONES.index("hips"), mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": "body", "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w},
            "indices": a_i, "material": 0, "mode": 4}]}],
        "skins": [{"joints": list(range(len(BONES))), "inverseBindMatrices": a_ibm, "skeleton": BONES.index("hips")}],
        "materials": [{"name": "voxel", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
            "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 1}],
        "samplers": [{"magFilter": NEAREST, "minFilter": NEAREST, "wrapS": 33071, "wrapT": 33071},
                     {"magFilter": 9729, "minFilter": 9729}],
        "images": [{"name": "palette", "bufferView": v_pal, "mimeType": "image/png"},
                   {"name": "thumbnail", "bufferView": v_th, "mimeType": "image/png"}],
        "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binb)}],
        "extensionsUsed": ["KHR_materials_unlit"],
    }
    head = BONES.index("head")
    if version == "0.x":
        j["extensionsUsed"].append("VRM")
        j["extensions"] = {"VRM": {
            "exporterVersion": "VRM RETRO MAKER", "specVersion": "0.0",
            "meta": meta_0(meta, 1),
            "humanoid": {"humanBones": [{"bone": bn, "node": k, "useDefaultValues": True} for k, bn in enumerate(BONES)],
                         "armStretch": 0.05, "legStretch": 0.05, "upperArmTwist": 0.5, "lowerArmTwist": 0.5,
                         "upperLegTwist": 0.5, "lowerLegTwist": 0.5, "feetSpacing": 0, "hasTranslationDoF": False},
            "firstPerson": {"firstPersonBone": head, "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
                            "meshAnnotations": [{"mesh": 0, "firstPersonFlag": "Auto"}], "lookAtTypeName": "Bone",
                            **{k: {"curve": [0, 0, 0, 1, 1, 1, 1, 0], "xRange": 90, "yRange": 10}
                               for k in ("lookAtHorizontalInner", "lookAtHorizontalOuter",
                                         "lookAtVerticalDown", "lookAtVerticalUp")}},
            "blendShapeMaster": {"blendShapeGroups": []},
            "secondaryAnimation": {"boneGroups": [], "colliderGroups": []},
            "materialProperties": [{"name": "voxel", "shader": "VRM/UnlitTexture", "renderQueue": 2000,
                                    "floatProperties": {"_BlendMode": 0, "_Cutoff": 0.5},
                                    "vectorProperties": {"_Color": [1, 1, 1, 1], "_MainTex": [0, 0, 1, 1]},
                                    "textureProperties": {"_MainTex": 0}, "keywordMap": {},
                                    "tagMap": {"RenderType": "Opaque"}}]}}
    else:
        j["extensionsUsed"].append("VRMC_vrm")
        j["extensions"] = {"VRMC_vrm": {
            "specVersion": "1.0", "meta": meta_1(meta, 1),
            "humanoid": {"humanBones": {bn: {"node": k} for k, bn in enumerate(BONES)}},
            "firstPerson": {"meshAnnotations": [{"node": mesh_node, "type": "auto"}]},
            "lookAt": {"type": "bone", "offsetFromHeadBone": [0, 0.06, 0]}}}
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binb), 0x004E4942) + bytes(binb))


# ====================================================================== all steps
def make(img, height=64, colors=32, meters=1.5, depth=0.7, tol=40, joints=None, meta=None, version="1.0"):
    """Picture (PIL image) -> (VRM bytes, info).  Pass `joints` (from guess_joints, moved by the user) to skip the guess."""
    crop = crop_figure(cut_out(img, tol))
    idx, pal = pixelate(crop, height, colors)
    J = joints or guess_joints(idx)
    return build(crop, idx, pal, J, height, colors, meters, depth, meta, version)


def build(crop, idx, pal, J, height, colors, meters=1.5, depth=0.7, meta=None, version="1.0"):
    """Working picture + its pixel art + joints -> (VRM bytes, info)."""
    idx2, pal2, J2, part = t_pose(crop, J, height, colors)
    bones = pixel_bones(idx2, J2, part)
    h = thickness(idx2, part, depth)
    H = idx2.shape[0]
    scale = meters / H
    cx = J2["neck"][0]
    mesh = build_mesh(idx2, h, bones, scale, cx)
    P = bone_points(J2)
    bone_pos = {bn: ((p[0] - cx) * scale, (H - p[1]) * scale, 0.0) for bn, p in P.items()}
    m = dict(DEFAULT_META)
    m.update(meta or {})
    data = write_vrm(mesh, bone_pos, pal2, thumbnail(idx, pal), m, version)
    info = {"pixels": [int(idx.shape[1]), int(idx.shape[0])], "colors": len(pal2), "tris": int(len(mesh[4]) // 3),
            "verts": int(len(mesh[0])), "bytes": len(data), "version": version}
    return data, info, (idx2, pal2, J2, part, bones)


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    ver = "0.x" if "--0x" in sys.argv else "1.0"
    nums = [a for a in sys.argv[3:] if not a.startswith("--")]
    height = int(nums[0]) if nums else 64
    colors = int(nums[1]) if len(nums) > 1 else 32
    data, info, _ = make(Image.open(src), height, colors, version=ver)
    open(dst, "wb").write(data)
    print(json.dumps(info))
