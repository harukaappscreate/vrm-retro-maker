"""VRM RETRO MAKER - drop a VRM (0.x or 1.0) and get a retro VRM back: small textures, 16-256 shared colours.
Or drop a picture of a character and get a voxel VRM (img2vrm.py); you place the joints by dragging them.
The result is an ordinary VRM of the same version (bones, expressions, spring bones and license unchanged)
that works in any VRM app.  Built on VRM RETRO VIEWER, so you can check it with 231 motions before saving.

    python main.py                 then drop a .vrm file on the window
    python main.py avatar.vrm      open a file directly (a .png / .jpg picture of a character too)

Mouse: drag to turn, wheel to zoom.  Keys: LEFT / RIGHT turn, UP / DOWN zoom, SPACE next motion, T T-pose,
R make retro (64 colours, 128 px), TAB before / after, S save the retro VRM, P screenshot, G GIF, L license.
In the browser, index.html adds a tool bar and an info panel and talks to this app (see Bridge).

Nothing is uploaded anywhere: the file is read inside your own computer / browser.
Motions: Quaternius Universal Animation Library 1 & 2 and KayKit Character Animations (all CC0), via open-vrma.
Made with Pyxel 3 by Takashi Kitao (MIT License), NumPy (BSD-3-Clause), Pillow (MIT-CMU); in the browser, Pyodide (MPL-2.0).
"""
# SPDX-License-Identifier: MIT
import base64
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile

import numpy as np
import pyxel
from pyxel.cube import Camera, Mat4, Node, Primitive, Shading, Vec3

import fbx_io
import vrma_io
import vrm_io as V
import vrm_retro as R
import img2vrm as I

W, H = 320, 240
UI = [0x000000, 0x2B335F, 0x7E2072, 0x19959C, 0x8B4852, 0x395C98, 0xA9C1FF, 0xEEEEEE,
      0xD4186C, 0xD38441, 0xE9C35B, 0x70C6A9, 0x7696DE, 0xA3A3A3, 0xFF9798, 0xEDC7B0]
BG = 1
INBOX = "/tmp/inbox"
PIC_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
JOINT_KEYS = ["neck", "crotch", "shoulder0", "elbow0", "wrist0", "shoulder1", "elbow1", "wrist1",
              "hip0", "knee0", "ankle0", "hip1", "knee1", "ankle1"]
JOINT_LINES = [("neck", "crotch"), ("shoulder0", "elbow0"), ("elbow0", "wrist0"), ("shoulder1", "elbow1"),
               ("elbow1", "wrist1"), ("hip0", "knee0"), ("knee0", "ankle0"), ("hip1", "knee1"), ("knee1", "ankle1"),
               ("shoulder0", "shoulder1"), ("hip0", "hip1")]
TMP = tempfile.mkdtemp(prefix="vrmv_")
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS8 = ["down", "down_left", "left", "up_left", "up", "up_right", "right", "down_right"]
DIRS4 = ["down", "left", "up", "right"]
try:
    import js                                               # in the browser (Pyodide) only
except ImportError:
    js = None


def quat_of(M):
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    return V.mat_to_quat(R)


def qrot(q, v):
    u = q[:3]
    return v + 2 * np.cross(u, np.cross(u, v) + q[3] * v)


# ====================================================================== motions
class Motions:
    """The bundled CC0 motions: per humanoid bone, the world rotation delta from the T-pose (VRM 1.0 frame)."""

    def __init__(self, path):
        d = np.load(path)
        self.bones = [str(b) for b in d["bones"]]
        self.meta = json.loads(bytes(d["meta"]).decode())
        self.by_id = {m["id"]: m for m in self.meta}
        self.D = {m["id"]: d["d_" + m["id"]].astype(np.float32) for m in self.meta}
        self.Hp = {m["id"]: d["h_" + m["id"]].astype(np.float32) for m in self.meta}

    def add(self, label, D, Hp, group, loop=True):
        """Add a motion the user loaded (FBX / VRMA).  Returns its id ("my_<name>", made unique)."""
        base = "my_" + "".join(ch for ch in label.lower().replace(" ", "_") if ch.isalnum() or ch == "_")[:40]
        mid, k = base, 2
        while mid in self.by_id:
            mid, k = "%s_%d" % (base, k), k + 1
        m = dict(id=mid, label=label, group=group, source="user", loop=loop, frames=len(D),
                 seconds=round((len(D) - 1) / 30.0, 3))
        self.meta.append(m)
        self.by_id[mid] = m
        self.D[mid], self.Hp[mid] = D.astype(np.float32), Hp.astype(np.float32)
        return mid

    def frame(self, mid, t):
        """(deltas (B, 4), hips offset (3,)) at time t (seconds); loops wrap, one-shots hold the last frame."""
        D, Hp, m = self.D[mid], self.Hp[mid], self.by_id[mid]
        n = len(D)
        f = t * 30.0
        f = f % (n - 1) if m["loop"] else min(f, n - 1.001)
        i = int(f)
        u = f - i
        a, b = D[i], D[min(i + 1, n - 1)]
        b = np.where((np.sum(a * b, 1) < 0)[:, None], -b, b)
        q = a + (b - a) * u
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q, Hp[i] + (Hp[min(i + 1, n - 1)] - Hp[i]) * u


# ====================================================================== the figure
class Figure(Node):
    """The converted VRM: one textured Primitive per material, skinned with numpy every shown frame."""

    def __init__(self, model, motions):
        super().__init__()
        self.m = model
        self.motions = motions
        self.prims, self.p1 = [], []
        for p in model.parts:
            pr = Primitive(Primitive.MODE_TRIANGLES, p["pos"].reshape(-1).tolist(), p["idx"].tolist(), [],
                           p["uv"].reshape(-1).tolist(), cull=Primitive.CULL_NONE if p["double"] else Primitive.CULL_BACK)
            pr.compute_normals()
            self.prims.append(pr)
            self.p1.append(np.hstack([p["pos"], np.ones((len(p["pos"]), 1), np.float32)]))
        self.textures = []
        for k, idx in enumerate(model.tex_index):
            path = os.path.join(TMP, "t%d.png" % k)
            png_from_index(idx, model.palette, path)
            self.textures.append(pyxel.Image.from_image(path))
        # draw order: opaque, cut-out, then the see-through bits (eyes, brows)
        self.draw_order = sorted(range(len(model.parts)), key=lambda i: {"OPAQUE": 0, "MASK": 1}.get(model.parts[i]["alpha"], 2))
        self.rest_wq = np.array([quat_of(M) for M in model.rest_world])
        # humanoid bone -> node, in the order of the motion data
        self.bone_nodes = [model.human.get(b, -1) for b in motions.bones]
        self.node_bone = {n: k for k, n in enumerate(self.bone_nodes) if n >= 0}
        hips = model.human.get("hips")
        self.hips = hips
        self.hip_h = float(model.rest_world[hips][1, 3]) if hips is not None else 1.0
        self.flip = model.version == "VRM 0.x"                # 0.x faces -Z: turn the deltas by 180 degrees
        self.motion = "idle"
        self.t = 0.0
        self.q = model.rest_q.astype(np.float64).copy()
        self.tr = model.rest_t.copy()
        self.skin_ms = 0.0

    def play(self, mid):
        if mid == "tpose" or mid in self.motions.by_id:
            self.motion = mid
            self.t = 0.0

    def pose(self, dt):
        self.t += dt
        m = self.m
        q = m.rest_q.astype(np.float64).copy()
        tr = m.rest_t.copy()
        if self.motion != "tpose" and self.hips is not None:
            D, hp = self.motions.frame(self.motion, self.t)
            if self.flip:
                D = D * np.array([-1.0, 1.0, -1.0, 1.0], np.float32)
                hp = hp * np.array([-1.0, 1.0, -1.0], np.float32)
            Wq = {}
            for i in m.order:
                p = m.parent[i]
                pw = Wq.get(p) if p >= 0 else None
                if pw is None:
                    pw = self.rest_wq[p] if p >= 0 else np.array([0.0, 0.0, 0.0, 1.0])
                k = self.node_bone.get(i)
                if k is not None:
                    w = V.qmul(D[k], self.rest_wq[i])
                    q[i] = V.qmul(V.qconj(pw), w)
                    Wq[i] = w
                else:
                    Wq[i] = V.qmul(pw, q[i])
            p = m.parent[self.hips]
            pw = Wq.get(p, np.array([0.0, 0.0, 0.0, 1.0])) if p >= 0 else np.array([0.0, 0.0, 0.0, 1.0])
            tr[self.hips] = m.rest_t[self.hips] + qrot(V.qconj(pw), hp * self.hip_h)
        self.q, self.tr = q, tr

    def skin(self):
        t0 = time.perf_counter()
        local = self.m.local_mats(self.q, self.tr)
        world = self.m.world(local).astype(np.float32)
        for p, pr, p1 in zip(self.m.parts, self.prims, self.p1):
            mats = (world[p["skin_nodes"]] @ p["ibm"])[:, :3, :]           # (joints, 3, 4)
            pos = np.zeros((len(p1), 3), np.float32)
            for k in range(4):
                w = p["ww"][:, k]
                if not w.any():
                    continue
                pos += w[:, None] * np.einsum("nij,nj->ni", mats[p["slot"][:, k]], p1)
            pr.positions[:] = pos.reshape(-1).tolist()
            pr.compute_normals()
        self.skin_ms = self.skin_ms * 0.8 + (time.perf_counter() - t0) * 200

    def on_draw(self):
        m = Mat4.from_euler(Vec3(0.0, float(self.m.face_yaw), 0.0))
        for i in self.draw_order:
            p = self.m.parts[i]
            if p["keyed"]:
                self.prim(m, self.prims[i], self.textures[p["tex"]], colkey=V.KEY)
            else:
                self.prim(m, self.prims[i], self.textures[p["tex"]])


def png_from_index(idx, palette, path):
    """Indices -> PNG in the exact palette colours (pyxel.Image.from_image maps them back)."""
    from PIL import Image
    cols = np.zeros((256, 3), np.uint8)
    cols[:16] = [((c >> 16) & 255, (c >> 8) & 255, c & 255) for c in UI]
    cols[16:16 + len(palette)] = palette
    cols[255] = (255, 0, 255)
    Image.fromarray(cols[idx]).save(path)


# ====================================================================== the page (browser) <-> this app
class Bridge:
    """index.html appends commands to pyxel._vrmv (strings / JSON); this app reports back through js.vrmv*()."""

    @staticmethod
    def commands():
        cmds = list(getattr(pyxel, "_vrmv", None) or [])
        if cmds:
            pyxel._vrmv = []
        return cmds

    @staticmethod
    def send(fn, data):
        if js is None:
            return
        try:
            getattr(js, fn)(json.dumps(data))
        except Exception:
            pass


# ====================================================================== app
class App:
    def __init__(self, path=None):
        pyxel.init(W, H, title="VRM RETRO MAKER", fps=30, capture_sec=6)
        pyxel.mouse(True)
        pyxel.colors[:] = UI + [0x808080] * (255 - 16) + [0xFF00FF]
        self.stage = Node()
        self.stage.camera = Camera()
        self.stage.camera.fov = 30.0
        self.stage.camera.near = 0.05
        self.stage.camera.far = 50.0
        self.stage.camera.clear_color = BG
        self.motions = Motions(os.path.join(HERE, "motions.npz"))
        self.fig = None
        self.state = "drop"
        self.err = ""
        self.pending = path
        self.yaw, self.pitch, self.dist = 0.0, 8.0, 3.6
        self.drag = None
        self.show_lic = js is None                            # the browser page shows the license in its own panel
        self.show_perf = False
        self.times = {}
        self.draw_ms = 0.0
        self.note, self.note_t = "", 0.0
        self.job, self.job_kind, self.progress = None, "", (0.0, "")
        self.preview, self.preview_t, self.preview_k = None, 0.0, None
        self.src, self.src_name, self.out, self.report = None, "", None, None
        self.views = {}                                        # "before" / "after" -> (figure, colours, shading)
        self.side = "before"
        self.retro_req = None
        self.src_allowed = False
        self.mode = "retro"                                      # "retro": VRM -> retro VRM, "pic": picture -> voxel VRM
        self.pic = None
        self.pic_req = None
        self.drag_j = None
        os.makedirs(INBOX, exist_ok=True)
        self.seen = set(os.listdir(INBOX))
        self.used = set()
        self.send_motions()
        pyxel.run(self.update, self.draw)

    def send_motions(self, select=None):
        Bridge.send("vrmvReady", {"motions": [dict(id=m["id"], label=m["label"], loop=m["loop"], group=m.get("group", ""),
                                                   seconds=m.get("seconds", 1.0), user=m.get("source") == "user")
                                             for m in self.motions.meta],
                                 "select": select if isinstance(select, list) or select is None else [select]})

    # ------------------------------------------------------------------ input files
    def new_file(self):
        # dropped on the window (desktop: pyxel.dropped_files, browser: pyxel.js also fills pyxel._dropped_files)
        for f in list(getattr(pyxel, "dropped_files", None) or []) + list(getattr(pyxel, "_dropped_files", None) or []):
            if f not in self.used:
                self.used.add(f)
                return f
        try:
            for f in sorted(os.listdir(INBOX)):
                if f not in self.seen:
                    self.seen.add(f)
                    return os.path.join(INBOX, f)
        except OSError:
            pass
        return None

    def load(self, path):
        self.state = "load"
        self.msg = "CONVERTING " + os.path.basename(path).split("_", 1)[-1][:28].upper()
        self.pending_load = path
        Bridge.send("vrmvStatus", {"text": "Converting " + os.path.basename(path).split("_", 1)[-1] + " ..."})

    def do_load(self, path):
        t0 = time.perf_counter()
        steps = {}
        try:
            data = open(path, "rb").read()
            fig, cols, sh, t1 = self.build(data, lambda s: steps.setdefault(s, time.perf_counter() - t0))
            model = fig.m
            t2 = time.perf_counter()
            j = R.read_glb(data)[0]
            allowed, lic = R.modification_allowed(j)
        except Exception as e:                               # a broken / unsupported file: say why and wait for the next one
            self.state = "drop" if self.fig is None else "view"
            self.err = ("CANNOT READ: %s" % e)[:76].upper()
            Bridge.send("vrmvStatus", {"text": "Cannot read this file: %s" % e, "error": True})
            return
        finally:
            try:
                if path.startswith(INBOX):
                    os.remove(path)                          # free the browser memory
            except OSError:
                pass
        name = os.path.basename(path)
        self.src = data
        self.src_name = name.split("_", 1)[-1] if path.startswith(INBOX) else name
        self.src_allowed = allowed
        self.out, self.report = None, None
        self.views = {"before": (fig, cols, sh)}
        self.mode = "retro"
        Bridge.send("vrmvMode", {"mode": "retro"})
        self.side = None
        self.show("before")
        self.times = dict(read=steps.get("TEXTURES", 0), textures=t1 - t0 - steps.get("TEXTURES", 0), build=t2 - t1)
        self.yaw, self.pitch, self.dist = 0.0, 8.0, max(2.2, model.height * 2.3)
        self.state = "view"
        self.err = ""
        missing = [b for b in ("hips", "spine", "head", "leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg")
                   if b not in model.human]
        Bridge.send("vrmvInfo", dict(title=model.title, author=model.author, version=model.version, license=model.license,
                                     warnings=model.warnings, tris=model.tris, verts=model.verts, parts=len(model.parts),
                                     seconds=round(t2 - t0, 2), missing=missing, retro_allowed=allowed,
                                     retro_license=lic, bytes=len(data)))
        Bridge.send("vrmvStatus", {"text": "Ready. Choose colours and size, then Make retro." if allowed else
                                   "This avatar's license does not allow modification, so it cannot be made retro.",
                                   "error": not allowed})

    def build(self, data, progress=None):
        """VRM bytes -> (figure, Pyxel colours, shading, time after the textures)."""
        model = V.VrmModel(data, progress=progress)
        cols = UI + [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in model.palette]
        cols += [0x808080] * (256 - len(cols))
        cols[255] = 0xFF00FF
        cols = cols[:256]
        pyxel.colors[:] = cols                            # the textures below are made with these colours
        sh = Shading(list(cols))
        sh.direction = Vec3(-0.4, -0.8, -0.5)
        t1 = time.perf_counter()
        return Figure(model, self.motions), cols, sh, t1

    def show(self, side):
        """Put the "before" or "after" figure on the stage; the motion keeps playing where it was."""
        if side not in self.views or side == self.side:
            return
        fig, cols, sh = self.views[side]
        old = self.fig
        if old is not None:
            self.stage.remove_child(old)
            fig.motion, fig.t = old.motion, old.t
        pyxel.colors[:] = cols
        self.stage.shading = sh
        self.fig = fig
        self.stage.add_child(fig)
        self.side = side
        Bridge.send("vrmvSide", {"side": side})

    def do_retro(self, req):
        """Make the retro VRM from the original bytes, then show it."""
        try:
            colors = max(2, min(256, int(req.get("colors", 64))))
            size = max(16, min(1024, int(req.get("size", 128))))
            out, rep = R.retro(self.src, colors, size, bool(req.get("dither")))
            self.views.pop("after", None)
            fig, cols, sh, _ = self.build(out)
            pyxel.colors[:] = self.views[self.side][1]       # build() changed the colours: put back what is shown
        except Exception as e:
            self.state = "view"
            Bridge.send("vrmvStatus", {"text": "Could not make the retro VRM: %s" % e, "error": True})
            return
        self.out, self.report = out, rep
        self.views["after"] = (fig, cols, sh)
        self.side = None if self.side == "after" else self.side
        self.state = "view"
        self.show("after")
        Bridge.send("vrmvRetro", rep)
        Bridge.send("vrmvStatus", {"text": "Done: %d colours, textures up to %d px, %.1f MB (was %.1f MB). "
                                   "Compare with Before / After, then Download." % (
                                       rep["colors_used"], size, rep["bytes_after"] / 1e6, rep["bytes_before"] / 1e6)})

    def route_file(self):
        f = self.new_file()
        if not f:
            return False
        if f.lower().endswith((".fbx", ".vrma")):
            self.job, self.job_kind, self.progress = self.read_motion(f), "fbx", (0.0, "")
        elif f.lower().endswith(PIC_EXT):
            self.state, self.msg, self.pending_pic = "picload", "READING THE PICTURE", f
            Bridge.send("vrmvStatus", {"text": "Reading the picture ..."})
        else:
            self.load(f)
        return True

    # ------------------------------------------------------------------ picture -> voxel VRM
    PIC_DEFAULT = {"height": 64, "colors": 32, "tol": 40, "depth": 0.7, "meters": 1.5, "mirror": True, "holes": True}

    def open_picture(self, path):
        from PIL import Image
        name = os.path.basename(path)
        name = name.split("_", 1)[-1] if path.startswith(INBOX) else name
        try:
            img = Image.open(path)
            img.draft("RGB", (1600, 1600))                    # big JPEGs: decode smaller (much faster)
            img = img.convert("RGBA")
            img.thumbnail((1600, 1600), Image.LANCZOS)
            self.pic = {"src": img, "name": name, "set": dict(self.PIC_DEFAULT), "cut_tol": None}
            self.pic_redo(guess=True)
        except Exception as e:
            self.pic = None
            self.state = "drop" if self.fig is None else "view"
            Bridge.send("vrmvStatus", {"text": "Cannot read this picture: %s" % e, "error": True})
            return
        finally:
            try:
                if path.startswith(INBOX):
                    os.remove(path)
            except OSError:
                pass
        self.mode = "pic"
        self.state = "pic"
        Bridge.send("vrmvMode", {"mode": "pic"})
        self.send_pic("Check the red points (joints): drag them onto the shoulders, elbows, wrists, hips, knees "
                      "and ankles. Then Make VRM.")

    def send_pic(self, text):
        p = self.pic
        h, w = p["idx"].shape
        thumbs = []
        if len(p["crops"]) > 1:                              # a sheet with several views: small pictures for the page
            from PIL import Image
            for c in p["crops"]:
                t = c.copy()
                t.thumbnail((96, 96), Image.BOX)
                b = io.BytesIO()
                t.save(b, format="PNG")
                thumbs.append("data:image/png;base64," + base64.b64encode(b.getvalue()).decode())
        Bridge.send("vrmvPic", {"name": p["name"], "pixels": [w, h], "colors": len(p["pal"]), "set": p["set"],
                                "arm": round(p["J"]["arm_width"][0], 1), "views": thumbs, "roles": p["roles"]})
        Bridge.send("vrmvStatus", {"text": text})

    def pic_redo(self, guess=False):
        """Cut out, pixel art and (when asked, or the first time) the joint guess, from the current settings."""
        p, st = self.pic, self.pic["set"]
        if p["cut_tol"] != (st["tol"], st.get("holes", True)):
            cut = I.cut_out(p["src"], st["tol"], holes=st.get("holes", True))
            p["crops"] = [I.crop_figure(f) for f in I.sheet_figures(cut)]
            p["roles"] = I.guess_roles(p["crops"]) if len(p["crops"]) > 1 else ["front"]
            p["cut_tol"] = (st["tol"], st.get("holes", True))
            guess = True
        p["crop"] = p["crops"][p["roles"].index("front")]
        old_h = p["idx"].shape[0] if "idx" in p else None
        p["idx"], p["pal"] = I.pixelate(p["crop"], int(st["height"]), int(st["colors"]))
        new_h = p["idx"].shape[0]
        if guess or "J" not in p:
            p["J"] = I.guess_joints(p["idx"])
        elif old_h and old_h != new_h:
            k = new_h / float(old_h)
            p["J"] = {key: ([v[0] * k, v[1] * k] if key != "arm_width" else [v[0] * k, 0]) for key, v in p["J"].items()}
        self.pic_image()

    def pic_image(self):
        """The pixel art as a Pyxel image (UI colours + up to 239 picture colours, 255 = see-through)."""
        from PIL import Image
        p = self.pic
        pal = p["pal"]
        disp = pal[:239]
        lut = np.arange(len(pal))
        if len(pal) > 239:                                   # more colours than the screen has: the nearest shown one
            a = np.asarray(disp, np.int32)
            for k in range(239, len(pal)):
                lut[k] = int(((a - np.asarray(pal[k])) ** 2).sum(1).argmin())
        cols = UI + [r << 16 | g << 8 | b for r, g, b in disp]
        cols += [0x808080] * (256 - len(cols))
        cols[255] = 0xFF00FF
        p["cols"] = cols[:256]
        pyxel.colors[:] = p["cols"]
        idx = p["idx"]
        ind = np.where(idx >= 0, lut[np.maximum(idx, 0)] + 16, 255).astype(np.uint8)
        full = np.zeros((256, 3), np.uint8)
        full[:] = [(c >> 16 & 255, c >> 8 & 255, c & 255) for c in p["cols"]]
        path = os.path.join(TMP, "pic.png")
        Image.fromarray(full[ind]).save(path)
        p["img"] = pyxel.Image.from_image(path)
        h, w = idx.shape
        p["sc"] = max(1, min((H - 28) // h, (W - 8) // w))
        p["ox"] = (W - w * p["sc"]) // 2
        p["oy"] = 12 + (H - 28 - h * p["sc"]) // 2

    def pic_xy(self, v):
        p = self.pic
        return p["ox"] + v[0] * p["sc"], p["oy"] + v[1] * p["sc"]

    def pic_mouse(self):
        p = self.pic
        mx, my = pyxel.mouse_x, pyxel.mouse_y
        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            best = None
            for k in JOINT_KEYS:
                x, y = self.pic_xy(p["J"][k])
                d = (x - mx) ** 2 + (y - my) ** 2
                if d <= 49 and (best is None or d < best[0]):
                    best = (d, k)
            self.drag_j = best[1] if best else None
        if self.drag_j and pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
            k = self.drag_j
            h, w = p["idx"].shape
            x = min(max((mx - p["ox"]) / p["sc"], 0.0), float(w))
            y = min(max((my - p["oy"]) / p["sc"], 0.0), float(h))
            p["J"][k] = [x, y]
            if p["set"].get("mirror") and k[-1] in "01":         # the same joint on the other side, mirrored
                other = k[:-1] + ("1" if k[-1] == "0" else "0")
                p["J"][other] = [2 * p["J"]["neck"][0] - x, y]
        elif not pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
            self.drag_j = None
        if pyxel.btnp(pyxel.KEY_M):
            self.pic_command("picmake:{}")
        if pyxel.btnp(pyxel.KEY_J):
            self.pic_command("pic:" + json.dumps({"guess": True}))
        if pyxel.btnp(pyxel.KEY_X):
            p["set"]["mirror"] = not p["set"].get("mirror")

    def pic_command(self, c):
        p = self.pic
        if p is None:
            return
        if c.startswith("pic:"):
            req = json.loads(c[4:])
            st = p["set"]
            for k in ("height", "colors", "tol", "depth", "meters", "mirror", "holes"):
                if k in req:
                    st[k] = req[k]
            if "arm" in req:
                p["J"]["arm_width"] = [max(1.0, float(req["arm"])), 0]
            if "roles" in req:
                roles = [r if r in I.VIEWS else None for r in req["roles"]][:len(p["crops"])]
                if "front" not in roles:
                    Bridge.send("vrmvStatus", {"text": "One view must be the front.", "error": True})
                    return
                front_changed = roles.index("front") != p["roles"].index("front")
                p["roles"] = roles
                if front_changed:
                    req["guess"] = True
            if any(k in req for k in ("height", "colors", "tol", "guess", "holes")):
                self.pic_redo(guess=bool(req.get("guess")))
                self.send_pic("Joints guessed again." if req.get("guess") else "Updated.")
        elif c == "edit":
            self.state = "pic"
            self.mode = "pic"
            pyxel.colors[:] = p["cols"]
            Bridge.send("vrmvMode", {"mode": "pic", "edit": True})
        elif c.startswith("picmake:"):
            self.pic_req = json.loads(c[8:])
            self.state, self.msg = "picmake", "MAKING THE VOXEL VRM"
            Bridge.send("vrmvStatus", {"text": "Making the voxel VRM ..."})
        elif c.startswith("picsave:"):
            req = json.loads(c[8:])
            if not req.get("rights"):
                Bridge.send("vrmvStatus", {"text": "Please confirm that you may use this picture.", "error": True})
                return
            ver = "0.x" if req.get("version") == "0.x" else "1.0"
            try:
                data = self.voxel_bytes(req.get("meta"), ver)
            except Exception as e:
                Bridge.send("vrmvStatus", {"text": "Could not write the VRM: %s" % e, "error": True})
                return
            stem = os.path.splitext(p["name"])[0] or "picture"
            path = os.path.join(TMP, "%s_voxel_%s.vrm" % (stem, "1_0" if ver == "1.0" else "0_x"))
            with open(path, "wb") as f:
                f.write(data)
            self.deliver(path)
            self.note, self.note_t = ("SAVED " + os.path.basename(path)).upper()[:60], 3.0
            Bridge.send("vrmvStatus", {"text": "Saved %s (VRM %s, %.2f MB)." % (os.path.basename(path), ver, len(data) / 1e6)})

    def voxel_bytes(self, meta, version):
        p, st = self.pic, self.pic["set"]
        if len([r for r in p["roles"] if r]) > 1:              # several views: front + back / sides
            data, info, _ = I.build_views(p["crops"], p["roles"], p["J"], int(st["height"]), int(st["colors"]),
                                          float(st["meters"]), float(st["depth"]), meta or p.get("meta"), version)
        else:
            data, info, _ = I.build(p["crop"], p["idx"], p["pal"], p["J"], int(st["height"]), int(st["colors"]),
                                    float(st["meters"]), float(st["depth"]), meta or p.get("meta"), version)
        p["info"] = info
        return data

    def make_voxel(self, req):
        p = self.pic
        if req.get("meta"):
            p["meta"] = req["meta"]
        try:
            data = self.voxel_bytes(p.get("meta"), "1.0")
            fig, cols, sh, _ = self.build(data)
        except Exception as e:
            self.state = "pic"
            pyxel.colors[:] = p["cols"]
            Bridge.send("vrmvStatus", {"text": "Could not make the VRM: %s" % e, "error": True})
            return
        self.views = {"after": (fig, cols, sh)}
        self.side = None
        self.src, self.out, self.report = None, None, None
        self.src_allowed = False
        self.mode = "pic"
        self.state = "view"
        self.show("after")
        self.dist = max(2.2, fig.m.height * 2.3)
        self.fig.play("walk" if "walk" in self.motions.by_id else "idle")
        Bridge.send("vrmvMotion", {"id": self.fig.motion})
        info = p["info"]
        Bridge.send("vrmvVoxel", info)
        Bridge.send("vrmvStatus", {"text": "Done: %d x %d pixels, %d colours, %s triangles. Try the motions, "
                                   "then download. Edit joints to fix anything." % (
                                       info["pixels"][0], info["pixels"][1], info["colors"], format(info["tris"], ","))})

    def draw_pic(self):
        p = self.pic
        h, w = p["idx"].shape
        sc = p["sc"]
        pyxel.blt(p["ox"] + (w * sc - w) / 2, p["oy"] + (h * sc - h) / 2, p["img"], 0, 0, w, h, 255, 0, sc)
        for a, b in JOINT_LINES:
            x0, y0 = self.pic_xy(p["J"][a])
            x1, y1 = self.pic_xy(p["J"][b])
            pyxel.line(x0, y0, x1, y1, 10)
        # arm thickness: a circle at each elbow
        r = max(1.0, p["J"]["arm_width"][0] * 0.75 + 1.0) * sc
        for sd in "01":
            x, y = self.pic_xy(p["J"]["elbow" + sd])
            pyxel.circb(x, y, r, 12)
        for k in JOINT_KEYS:
            x, y = self.pic_xy(p["J"][k])
            col = 7 if k == self.drag_j else 8
            pyxel.circ(x, y, 2, col)
            pyxel.circb(x, y, 3, 0)
        views = [r for r in p["roles"] if r]
        tag = ("  VIEWS: " + " ".join(views)) if len(views) > 1 else ""
        pyxel.text(4, 3, ("%s  %dX%d PX%s" % (p["name"], w, h, tag))[:78].upper(), 7)
        pyxel.text(4, H - 10, "DRAG THE RED POINTS ONTO THE JOINTS" + ("   M MAKE VRM  J GUESS  X MIRROR %s" % (
            "ON" if p["set"].get("mirror") else "OFF") if js is None else "  (MIRROR %s)" % (
            "ON" if p["set"].get("mirror") else "OFF")), 13)

    def out_name(self):
        stem = os.path.splitext(self.src_name)[0] or "avatar"
        return "%s_retro%d.vrm" % (stem, self.report["colors_asked"])

    def read_motion(self, path):
        """A motion file (.fbx or .vrma): bake every animation in it like the bundled motions and add them to the list.
        A generator (one animation per step), run like an export so the page stays alive with big files."""
        name = os.path.basename(path)
        if path.startswith(INBOX):
            name = name.split("_", 1)[-1]                    # the page adds "<time>_" in front
        stem = os.path.splitext(name)[0].replace("_", " ").strip() or "motion"
        added = []
        try:
            yield 0.02, "Reading " + name
            t0 = time.perf_counter()
            data = open(path, "rb").read()
            if name.lower().endswith(".vrma"):
                D, Hp, _sec, _name = vrma_io.read_vrma(data, self.motions.bones)
                added.append((self.motions.add(stem, D, Hp, "Your motions", loop=True), stem))
                rig = "VRM"
            else:
                ff = fbx_io.FbxFile(data)
                human, rig = fbx_io.humanoid(ff.bones)
                takes = [k for k, t in enumerate(ff.takes) if t[2] > 0 and not is_tpose(t[0])]
                if not takes:
                    raise ValueError("no animation in this file")
                droop = fbx_io.arm_droop(ff.motion(takes[0]), human) if "leftUpperArm" in human else 0.0
                named = len(takes) > 1                       # several animations: use their own names
                loops = any("loop" in ff.takes[k][0].lower() for k in takes)
                for n, k in enumerate(takes):
                    take = ff.takes[k][0]
                    D, Hp = fbx_io.to_viewer(ff.motion(k), self.motions.bones, human)
                    label = take.replace("_", " ").strip() if named else stem
                    loop = is_loop_name(take) if loops else True
                    added.append((self.motions.add(label, D, Hp, "Your motions", loop=loop), label))
                    yield (n + 1) / len(takes), "%s (%d/%d)" % (label, n + 1, len(takes))
            sec = time.perf_counter() - t0
        except Exception as e:
            self.err = ("CANNOT READ MOTION: %s" % e)[:76].upper()
            self.note, self.note_t = self.err, 4.0
            Bridge.send("vrmvStatus", {"text": "Cannot read %s: %s" % (name, e), "error": True})
            if added:                                        # some animations of the file were added before the error
                self.send_motions(select=[a[0] for a in added])
            return
        finally:
            try:
                if path.startswith(INBOX):
                    os.remove(path)
            except OSError:
                pass
        self.err = ""
        mid = added[0][0]
        self.send_motions(select=[a[0] for a in added])
        if self.fig is not None:
            self.fig.play(mid)
            Bridge.send("vrmvMotion", {"id": mid})
        what = added[0][1] if len(added) == 1 else "%d motions from %s" % (len(added), name)
        self.note = ("ADDED: " + what).upper()[:60]
        self.note_t = 3.0
        text = "Added %s (%s bone names, read in %.1f s)" % (what, rig, sec)
        if name.lower().endswith(".fbx") and droop > 20:            # the rig rests in an A-pose, not a T-pose
            text += (". Note: this skeleton rests in an A-pose (arms %d degrees down), so the arms may look wrong"
                     % round(droop))
            self.note = "A-POSE SKELETON: ARMS MAY LOOK WRONG"
            self.note_t = 5.0
        Bridge.send("vrmvStatus", {"text": text, "error": name.lower().endswith(".fbx") and droop > 20})

    # ------------------------------------------------------------------ commands from the page or keys
    def command(self, c):
        if self.state != "view":
            return
        if self.mode == "pic" and (c == "edit" or c.startswith("picsave:") or c.startswith("pic:")
                                   or c.startswith("picmake:") or c.startswith("joints:")):
            self.pic_command(c)
            return
        if c.startswith("motion:"):
            self.fig.play(c[7:])
            Bridge.send("vrmvMotion", {"id": self.fig.motion})
        elif c.startswith("preview:"):
            arg = c[8:]
            self.set_preview(None if arg == "off" else json.loads(arg))
        elif c.startswith("retro:"):
            if not self.src_allowed:
                self.note, self.note_t = "THE LICENSE FORBIDS MODIFICATION: NOT CONVERTED", 4.0
                Bridge.send("vrmvStatus", {"text": "This avatar's license does not allow modification, "
                                           "so this tool will not make a retro version of it.", "error": True})
                return
            self.retro_req = json.loads(c[6:])
            self.state = "retro"
            self.msg = "MAKING THE RETRO VRM"
            Bridge.send("vrmvStatus", {"text": "Making the retro VRM ..."})
        elif c.startswith("show:"):
            self.show(c[5:])
        elif c == "download":
            if self.out is None:
                return
            path = os.path.join(TMP, self.out_name())
            with open(path, "wb") as f:
                f.write(self.out)
            self.deliver(path)
            self.note, self.note_t = ("SAVED " + self.out_name()).upper()[:60], 3.0
        elif c == "view:reset":
            self.yaw, self.pitch = 0.0, 8.0
            self.dist = max(2.2, self.fig.m.height * 2.3)
        elif c == "shot":
            pyxel.screenshot()
        elif c == "gif":
            pyxel.screencast()
        elif c.startswith("export:") or c == "export3d":
            if self.job is None:
                self.job = self.export(json.loads(c[7:])) if c.startswith("export:") else self.export3d()
                self.job_kind = "sprites" if c.startswith("export:") else "3d"
                self.progress = (0.0, "Starting")
                Bridge.send("vrmvProgress", {"busy": True, "kind": self.job_kind, "pct": 0, "text": "Starting"})

    def next_motion(self):
        ids = [m["id"] for m in self.motions.meta] + ["tpose"]
        i = ids.index(self.fig.motion) if self.fig.motion in ids else 0
        self.command("motion:" + ids[(i + 1) % len(ids)])

    # ------------------------------------------------------------------ export
    def export(self, req):
        """One PNG per motion (rows = directions, columns = frames), sheet.json, a Pyxel sample and a README, zipped.
        A generator: it yields (fraction done, text) after each frame so the page and the screen can show progress."""
        fig = self.fig
        c = int(req.get("cell", 64))
        dirs = DIRS8 if int(req.get("dirs", 8)) == 8 else DIRS4
        mids = [m for m in req.get("motions", ["idle", "walk"]) if m in self.motions.by_id] or ["idle"]
        frames_of = {mid: sheet_frames(self.motions.by_id[mid], req, c) for mid in mids}
        # by speed: frames exactly 1 / fps apart (6, 10, 15 fps divide Pyxel's 30), unless a long motion hit the cap
        step = {mid: (1.0 / float(req["fps"]) if req.get("fps") and not req.get("frames")
                      and frames_of[mid] < 8192 // c else None) for mid in mids}
        cam = self.stage.camera
        old = (cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t)
        cam.fov = 12.0
        cam.clear_color = V.KEY
        name = "".join(ch for ch in fig.m.title if ch.isalnum() or ch in "_-")[:24] or "vrm"
        from PIL import Image
        files, sheet_meta = {}, {}
        nd = len(dirs)
        work = sum(frames_of[mid] * (nd + 0.3) for mid in mids)   # draws (+ saving the PNG and the GIF)
        total, done = work * 1.05, 0.0                            # + the combined image and the zip

        def view(yaw, target, x, y):
            self.sprite_camera(yaw)
            self.stage.draw(x, y, c, c, target=target)

        for mid in mids:
            m = self.motions.by_id[mid]
            dur = m["seconds"]
            nfr = frames_of[mid]
            sheet = pyxel.Image(c * nfr, c * len(dirs))
            for k in range(nfr):
                fig.motion = mid
                fig.t = frame_time(m, k, nfr, step[mid])
                fig.pose(0.0)
                fig.skin()
                for row, dname in enumerate(dirs):
                    view(DIRS8.index(dname) * 45.0, sheet, k * c, row * c)
                done += nd
                cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t = old   # the screen keeps its own view
                yield done / total, "%s %d/%d" % (m["label"], k + 1, nfr)
                old = (cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t)
                cam.fov = 12.0
                cam.clear_color = V.KEY
            base = os.path.join(TMP, "%s_%s" % (name, mid))
            sheet.save(base, 1)                                     # pyxel adds .png
            a = np.array(Image.open(base + ".png").convert("RGBA"))
            key = (a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 255)
            a[key] = (255, 0, 255, 0)                               # transparent; magenta underneath = Pyxel colour key
            b = io.BytesIO()
            Image.fromarray(a).save(b, "PNG", optimize=True)
            fn = "%s_%s.png" % (name, mid)
            files[fn] = b.getvalue()
            fps = 1.0 / step[mid] if step[mid] else (nfr / dur if m["loop"] else (nfr - 1) / dur)
            sheet_meta[mid] = dict(file=fn, frames=nfr, fps=round(fps, 2), loop=m["loop"], gif="%s_%s.gif" % (name, mid))
            files[sheet_meta[mid]["gif"]] = gif_of_row(a, 0, c, nfr, fps, m["loop"])   # the front row: a preview
            done += nfr * 0.3
        cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t = old
        # every motion stacked in one image too, while that stays a sensible size (a long selection skips it)
        yield done / total, "Combining"
        all_name = None
        if len(mids) * len(dirs) * c <= 8192:
            sheets = [np.array(Image.open(io.BytesIO(files[sheet_meta[mid]["file"]]))) for mid in mids]
            allimg = np.zeros((sum(x.shape[0] for x in sheets), max(x.shape[1] for x in sheets), 4), np.uint8)
            allimg[...] = (255, 0, 255, 0)
            y = 0
            for mid, x in zip(mids, sheets):
                allimg[y:y + x.shape[0], :x.shape[1]] = x
                sheet_meta[mid]["y"] = y                            # where this motion starts in the combined image
                y += x.shape[0]
            b = io.BytesIO()
            Image.fromarray(allimg).save(b, "PNG", optimize=True)
            all_name = "%s_all.png" % name
            files[all_name] = b.getvalue()
        timing = ("%d frames per motion" % int(req["frames"])) if req.get("frames") else \
            ("%g fps (each motion has as many frames as its length needs)" % float(req.get("fps", 10)))
        mine = [self.motions.by_id[mid]["label"] for mid in mids if self.motions.by_id[mid].get("source") == "user"]
        info = dict(cell=c, directions=dirs, motions=sheet_meta, all_in_one=all_name, key_color="#FF00FF",
                    timing=timing, user_motions=mine,
                    palette=["%06X" % (int(v) & 0xFFFFFF) for v in list(pyxel.colors)[:256]],
                    source=dict(title=fig.m.title, author=fig.m.author, version=fig.m.version, license=fig.m.license,
                                warnings=fig.m.warnings))
        files["sheet.json"] = json.dumps(info, indent=2).encode()
        files["README.txt"] = readme_text(info).encode()
        try:
            files["pyxel_sample/sample_sprites.py"] = open(os.path.join(HERE, "sample_sprites.py"), "rb").read()
            files["pyxel_sample/HOW_TO_RUN.txt"] = howto_text(
                "sample_sprites.py", "", "arrows walk, hold X to run, Z attack (punch / sword / spell, if exported).",
                "矢印で歩く、X を押しながらで走る、Z で攻撃（punch / sword / spell を書き出したとき）。").encode()
        except OSError:
            pass
        yield max(done / total, 0.97), "Zipping"
        zpath = os.path.join(TMP, name + "_sprites.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fn, data in files.items():
                z.writestr(name + "_sprites/" + fn, data)
        self.deliver(zpath)
        self.note = "SAVED %s_SPRITES.ZIP (%d MOTIONS)" % (name.upper(), len(mids))
        self.note_t = 4.0
        Bridge.send("vrmvStatus", {"text": "Saved %s_sprites.zip (%d motions, %d directions, %d px)" % (name, len(mids), len(dirs), c)})

    def export3d(self):
        """The converted avatar for Pyxel 3 games: model.npz + tex_*.png + motions.npz + vrm_figure.py + sample_3d.py."""
        from PIL import Image
        fig = self.fig
        m = fig.m
        name = "".join(ch for ch in m.title if ch.isalnum() or ch in "_-")[:24] or "vrm"
        files = {}
        arrays = {}
        parts = []
        for k, p in enumerate(m.parts):
            nv = len(p["pos"])
            arrays["p%d_pos" % k] = p["pos"].astype(np.float32)
            arrays["p%d_uv" % k] = p["uv"].astype(np.float32)
            arrays["p%d_idx" % k] = p["idx"].astype(np.uint16 if nv < 65536 else np.uint32)
            arrays["p%d_slot" % k] = p["slot"].astype(np.uint8 if len(p["skin_nodes"]) < 256 else np.uint16)
            arrays["p%d_ww" % k] = p["ww"].astype(np.float16)
            arrays["p%d_skin_nodes" % k] = np.asarray(p["skin_nodes"], np.int32)
            arrays["p%d_ibm" % k] = np.asarray(p["ibm"], np.float32)
            parts.append(dict(name=str(p.get("name", "part%d" % k)), tex=int(p["tex"]), alpha=p["alpha"],
                              double=bool(p["double"]), keyed=bool(p["keyed"])))
            yield 0.6 * (k + 1) / len(m.parts), "Mesh %d/%d" % (k + 1, len(m.parts))
        hips = m.human.get("hips", -1)
        info = dict(title=m.title, author=m.author, version=m.version, license=m.license, warnings=m.warnings,
                    height=float(m.height), face_yaw=float(m.face_yaw), flip=bool(fig.flip), hips=int(hips),
                    hip_h=float(fig.hip_h), textures=len(m.tex_index), parts=parts, tris=int(m.tris),
                    motions=len(self.motions.meta),
                    user_motions=[x["label"] for x in self.motions.meta if x.get("source") == "user"])
        arrays.update(parent=np.asarray(m.parent, np.int32), order=np.asarray(m.order, np.int32),
                      rest_t=m.rest_t.astype(np.float32), rest_q=m.rest_q.astype(np.float32),
                      rest_s=m.rest_s.astype(np.float32), rest_wq=fig.rest_wq.astype(np.float32),
                      bone_nodes=np.asarray(fig.bone_nodes, np.int32),
                      palette=np.array([int(r) << 16 | int(g) << 8 | int(b) for r, g, b in m.palette], np.int64),
                      info=np.frombuffer(json.dumps(info).encode(), np.uint8))
        b = io.BytesIO()
        np.savez_compressed(b, **arrays)
        files["model.tmx"] = tmx_wrap(b.getvalue(), "vrm_model")    # a Tiled map: Code & Magic accepts it
        cols = np.zeros((256, 3), np.uint8)
        cols[16:16 + len(m.palette)] = m.palette
        cols[255] = (255, 0, 255)
        for k, idx in enumerate(m.tex_index):
            b = io.BytesIO()
            Image.fromarray(cols[idx]).save(b, "PNG", optimize=True)
            files["tex_%d.png" % k] = b.getvalue()
            yield 0.6 + 0.35 * (k + 1) / len(m.tex_index), "Texture %d/%d" % (k + 1, len(m.tex_index))
        files["motions.tmx"] = tmx_wrap(open(os.path.join(HERE, "motions.npz"), "rb").read(), "vrm_motions")
        mine = self.user_motions_npz()
        if mine:
            files["my_motions.tmx"] = tmx_wrap(mine, "vrm_motions")   # kept apart: motions.tmx stays under 3 MB
            info["my_motions_mb"] = round(len(files["my_motions.tmx"]) / 1e6, 2)
        for fn in ("vrm_figure.py", "sample_3d.py"):
            try:
                files[fn] = open(os.path.join(HERE, fn), "rb").read()
            except OSError:
                pass
        files["README.txt"] = readme3d_text(info).encode()
        files["HOW_TO_RUN.txt"] = howto_text(
            "sample_3d.py", " numpy", "arrows walk, hold X to run, Z punch, SPACE every motion, T T-pose.",
            "矢印で歩く、X を押しながらで走る、Z でパンチ、SPACE で全モーション、T で T ポーズ。").encode()
        yield 1.0, "Zipping"
        zpath = os.path.join(TMP, name + "_pyxel3d.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fn, data in files.items():
                z.writestr(name + "_pyxel3d/" + fn, data)
        self.deliver(zpath)
        size = os.path.getsize(zpath)
        self.note = "SAVED %s_PYXEL3D.ZIP (%.1f MB)" % (name.upper(), size / 1e6)
        self.note_t = 4.0
        text = "Saved %s_pyxel3d.zip (%.1f MB)" % (name, size / 1e6)
        if info.get("my_motions_mb", 0) > 3.0:
            text += (". my_motions.tmx is %.1f MB: too big for Code & Magic (3 MB per file), fine in Pyxel"
                     % info["my_motions_mb"])
        Bridge.send("vrmvStatus", {"text": text})

    def sprite_camera(self, yaw):
        """The camera of the sprite sheets: from above at 25 degrees, the whole figure with room for jumps and arms."""
        h = self.fig.m.height
        dist = h * 1.3 / (2 * math.tan(math.radians(6.0)))
        el = math.radians(25.0)
        eye = Vec3(math.sin(math.radians(yaw)) * dist * math.cos(el), h * 0.55 + dist * math.sin(el),
                   math.cos(math.radians(yaw)) * dist * math.cos(el))
        self.stage.camera.transform = Mat4.look_at(eye, Vec3(0.0, h * 0.55, 0.0), Vec3.UP)

    # ------------------------------------------------------------------ live preview of the sprite export
    PREVIEW_POS = {"down": (1, 2), "down_left": (0, 2), "left": (0, 1), "up_left": (0, 0), "up": (1, 0),
                   "up_right": (2, 0), "right": (2, 1), "down_right": (2, 2)}      # where each direction sits (3 x 3)

    def set_preview(self, req):
        """req: the export settings ({"cell", "dirs", "fps" or "frames"}) or None to go back to the 3D view."""
        self.preview = req
        self.preview_t = 0.0
        self.preview_k = None
        if req:
            c = int(req.get("cell", 64))
            self.preview_img = pyxel.Image(c * 3, c * 3)
        Bridge.send("vrmvPreview", {"on": bool(req)})

    def draw_preview(self):
        fig, req = self.fig, self.preview
        c = int(req.get("cell", 64))
        dirs = DIRS8 if int(req.get("dirs", 8)) == 8 else DIRS4
        m = self.motions.by_id.get(fig.motion) or self.motions.by_id["idle"]
        n = sheet_frames(m, req, c)
        step = 1.0 / float(req["fps"]) if req.get("fps") and not req.get("frames") and n < 8192 // c else None
        speed = 1.0 / step if step else (n / m["seconds"] if m["loop"] else (n - 1) / m["seconds"])
        k = int(self.preview_t * speed)
        k = k % n if m["loop"] else min(k, n - 1)
        if k != self.preview_k or getattr(self, "preview_mid", None) != m["id"]:   # draw again only on a new frame
            self.preview_k, self.preview_mid = k, m["id"]
            cam = self.stage.camera
            old = (cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t)
            cam.fov, cam.clear_color = 12.0, V.KEY
            fig.motion, fig.t = m["id"], frame_time(m, k, n, step)
            fig.pose(0.0)
            fig.skin()
            self.preview_img.cls(V.KEY)
            for d in dirs:
                gx, gy = self.PREVIEW_POS[d]
                self.sprite_camera(DIRS8.index(d) * 45.0)
                self.stage.draw(gx * c, gy * c, c, c, target=self.preview_img)
            cam.fov, cam.clear_color, cam.transform, fig.motion, fig.t = old
        # show it as big as fits, in whole pixels when it can (the dots stay sharp)
        size = c * 3
        s = min((W - 8) / size, (H - 30) / size)
        s = math.floor(s) if s >= 1 else s
        x0, y0 = (W - size * s) / 2, 12 + (H - 30 - size * s) / 2
        pyxel.blt(x0 + (size * s - size) / 2, y0 + (size * s - size) / 2, self.preview_img, 0, 0, size, size, V.KEY,
                  scale=s)
        fps_txt = "%d FPS" % round(speed) if step else "%.1f FPS" % speed
        pyxel.text(4, 4, ("SPRITE PREVIEW  %d PX  %s  %d DIRECTIONS" % (c, fps_txt, len(dirs))), 7)
        pyxel.text(4, H - 10, ("%s  %d FRAMES" % (m["label"], n)).upper()[:40], 13)
        if s < 1:
            pyxel.text(W - 4 - 4 * 13, H - 10, "SHOWN SMALLER", 5)
        self.preview_t += 1 / 30

    def user_motions_npz(self):
        """The motions the user added (FBX / VRMA), in the format of motions.npz; None when there are none."""
        user = [x for x in self.motions.meta if x.get("source") == "user"]
        if not user:
            return None
        d = {"bones": np.array(self.motions.bones)}
        meta = list(user)
        for x in user:
            d["d_" + x["id"]] = self.motions.D[x["id"]].astype(np.float16)
            d["h_" + x["id"]] = self.motions.Hp[x["id"]].astype(np.float16)
        d["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        b = io.BytesIO()
        np.savez_compressed(b, **d)
        return b.getvalue()

    def deliver(self, path):
        """Browser: start a download.  Desktop: copy next to main.py."""
        if js is not None:
            js._savePyxelFile(path)
        else:
            import shutil
            shutil.copy(path, os.path.join(HERE, os.path.basename(path)))

    # ------------------------------------------------------------------ update / draw
    def update(self):
        if self.pending:
            if self.pending.lower().endswith(PIC_EXT):
                self.state, self.msg, self.pending_pic = "picload", "READING THE PICTURE", self.pending
            else:
                self.load(self.pending)
            self.pending = None
            return
        if self.state == "load":
            self.do_load(self.pending_load)
            return
        if self.state == "retro":
            self.do_retro(self.retro_req)
            return
        if self.state == "picload":
            self.open_picture(self.pending_pic)
            return
        if self.state == "picmake":
            self.make_voxel(self.pic_req)
            return
        if self.state == "pic":
            if self.route_file():
                return
            for c in Bridge.commands():
                self.pic_command(c)
            if self.state == "pic":
                self.pic_mouse()
            return
        if self.job is not None:                          # an export runs a little every frame (the page stays alive)
            try:
                self.progress = next(self.job)
                Bridge.send("vrmvProgress", {"busy": True, "kind": self.job_kind, "pct": int(self.progress[0] * 100),
                                             "text": self.progress[1]})
            except StopIteration:
                self.job = None
                Bridge.send("vrmvProgress", {"busy": False, "kind": self.job_kind})
            except Exception as e:
                self.job = None
                Bridge.send("vrmvProgress", {"busy": False, "kind": self.job_kind})
                Bridge.send("vrmvStatus", {"text": "%s failed: %s" % ("Reading the motion" if self.job_kind == "fbx"
                                                                      else "Export", e), "error": True})
            return
        if self.route_file():
            return
        for c in Bridge.commands():
            self.command(c)
        if pyxel.btnp(pyxel.KEY_F1):
            self.show_perf = not self.show_perf
        if self.state != "view":
            return
        # mouse: drag to turn (and tilt), wheel to zoom
        if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
            if self.drag:
                self.yaw -= (pyxel.mouse_x - self.drag[0]) * 1.2
                self.pitch = max(-20.0, min(60.0, self.pitch + (pyxel.mouse_y - self.drag[1]) * 0.6))
            self.drag = (pyxel.mouse_x, pyxel.mouse_y)
        else:
            self.drag = None
        if pyxel.mouse_wheel:
            self.dist = max(0.8, min(12.0, self.dist * (0.9 ** pyxel.mouse_wheel)))
        if pyxel.btn(pyxel.KEY_LEFT):
            self.yaw -= 3.0
        if pyxel.btn(pyxel.KEY_RIGHT):
            self.yaw += 3.0
        if pyxel.btn(pyxel.KEY_UP):
            self.dist = max(0.8, self.dist * 0.97)
        if pyxel.btn(pyxel.KEY_DOWN):
            self.dist = min(12.0, self.dist * 1.03)
        if pyxel.btnp(pyxel.KEY_SPACE):
            self.next_motion()
        if pyxel.btnp(pyxel.KEY_T):
            self.command("motion:tpose")
        if pyxel.btnp(pyxel.KEY_R):
            self.command('retro:{"colors": 64, "size": 128}')
        if pyxel.btnp(pyxel.KEY_TAB):
            self.command("show:" + ("before" if self.side == "after" else "after"))
        if pyxel.btnp(pyxel.KEY_S):
            if self.mode == "pic":
                ver = "0.x" if pyxel.btn(pyxel.KEY_SHIFT) else "1.0"
                self.command("picsave:" + json.dumps({"version": ver, "rights": True}))
            else:
                self.command("download")
        if pyxel.btnp(pyxel.KEY_E) and self.mode == "pic":
            self.command("edit")
        if pyxel.btnp(pyxel.KEY_P):
            self.command("shot")
        if pyxel.btnp(pyxel.KEY_G):
            self.command("gif")
        if pyxel.btnp(pyxel.KEY_L):
            self.show_lic = not self.show_lic
        if not self.preview:
            self.fig.pose(1 / 30)

    def draw(self):
        pyxel.cls(BG)
        if self.state == "drop":
            self.center(88, "VRM RETRO MAKER", 10)
            self.center(100, "DROP A VRM FILE OR A PICTURE HERE", 7)
            self.center(112, "VRM 0.X / 1.0 -> RETRO VRM", 13)
            self.center(122, "PICTURE OF A CHARACTER (PNG / JPG) -> VOXEL VRM", 13)
            self.center(140, "NOTHING IS UPLOADED. THE FILE STAYS ON YOUR DEVICE.", 6)
            if self.err:
                self.center(170, self.err, 8)
            if self.note_t > 0:
                self.note_t -= 1 / 30
                self.center(186, self.note, 10)
            if self.job is not None:
                self.draw_overlays()
            return
        if self.state == "pic":
            self.draw_pic()
            self.draw_overlays()
            return
        if self.state in ("load", "retro", "picload", "picmake"):
            self.center(110, self.msg, 7)
            self.center(122, "PLEASE WAIT...", 13)
            return
        fig = self.fig
        if self.preview:                                     # the sprite preview replaces the free 3D view
            self.draw_preview()
            self.draw_overlays()
            return
        fig.skin()
        t0 = time.perf_counter()
        h = fig.m.height
        a, b = math.radians(self.yaw), math.radians(self.pitch)
        tgt = Vec3(0.0, h * 0.55, 0.0)
        eye = Vec3(math.sin(a) * math.cos(b) * self.dist, h * 0.55 + math.sin(b) * self.dist, math.cos(a) * math.cos(b) * self.dist)
        self.stage.camera.transform = Mat4.look_at(eye, tgt, Vec3.UP)
        self.stage.draw(0, 0, W, H)
        self.draw_ms = self.draw_ms * 0.8 + (time.perf_counter() - t0) * 200
        m = fig.m
        label = "T-POSE" if fig.motion == "tpose" else self.motions.by_id[fig.motion]["label"].upper()
        pyxel.text(4, H - 10, label, 13)
        if self.mode == "pic":
            s = "VOXEL VRM  (E: EDIT JOINTS)" if js is None else "VOXEL VRM"
            pyxel.rect(W - len(s) * 4 - 22, H - 12, len(s) * 4 + 6, 9, 0)
            pyxel.text(W - len(s) * 4 - 19, H - 10, s, 10)
        elif self.out is not None:
            s = "AFTER (%d COLOURS)" % self.report["colors_used"] if self.side == "after" else "BEFORE"
            pyxel.rect(W - len(s) * 4 - 22, H - 12, len(s) * 4 + 6, 9, 0)
            pyxel.text(W - len(s) * 4 - 19, H - 10, s, 10 if self.side == "after" else 7)
        if js is None:                                       # desktop: the help and license are drawn here
            pyxel.text(4, 4, ("%s  %s" % (m.title, m.version))[:60].upper(), 7)
            pyxel.text(4, 12, ("BY " + m.author)[:60].upper(), 13)
            pyxel.text(4, H - 20, ("E EDIT  S SAVE 1.0  SHIFT+S SAVE 0.X  SPACE MOTION  L INFO" if self.mode == "pic" else
                                   "R RETRO  TAB BEFORE/AFTER  S SAVE  SPACE MOTION  L INFO"), 5)
        elif m.warnings:
            pyxel.rect(W - 13, 3, 10, 9, 8)
            pyxel.text(W - 9, 5, "!", 7)
        if self.show_lic:
            y = 26
            for s in m.license:
                pyxel.text(4, y, s[:76].upper(), 6)
                y += 8
            for s in m.warnings:
                pyxel.rect(3, y - 1, len(s) * 4 + 3, 8, 8)
                pyxel.text(5, y, s, 7)
                y += 9
        self.draw_overlays()
        if self.show_perf:
            tt = self.times
            pyxel.text(W - 132, 4, "TRIS %d VERTS %d" % (m.tris, m.verts), 10)
            pyxel.text(W - 132, 12, "READ %.2fS TEX %.2fS BUILD %.2fS" % (tt["read"], tt["textures"], tt["build"]), 10)
            pyxel.text(W - 132, 20, "SKIN %.1fMS DRAW %.1fMS" % (fig.skin_ms, self.draw_ms), 10 if fig.skin_ms + self.draw_ms < 33 else 8)

    def draw_overlays(self):
        """Export progress and short notes, over whatever the screen shows."""
        if self.job is not None:
            f, text = self.progress
            pyxel.rect(40, H // 2 - 14, W - 80, 28, 0)
            pyxel.rectb(40, H // 2 - 14, W - 80, 28, 7)
            what = {"3d": "EXPORTING 3D", "fbx": "READING MOTIONS"}.get(self.job_kind, "EXPORTING SPRITES")
            self.center(H // 2 - 9, (what + "  " + text).upper()[:56], 7)
            pyxel.rect(48, H // 2 + 2, int((W - 96) * f), 5, 11)
            pyxel.rectb(48, H // 2 + 2, W - 96, 5, 13)
        if self.note_t > 0:
            self.note_t -= 1 / 30
            pyxel.rect(0, H - 22, W, 10, 0)
            self.center(H - 20, self.note, 10)

    @staticmethod
    def center(y, s, col):
        pyxel.text(W // 2 - len(s) * 2, y, s, col)


def short_list(names, n=8):
    """"a, b, c and 12 more" (a README line stays readable)."""
    return ", ".join(names[:n]) + (" and %d more" % (len(names) - n) if len(names) > n else "")


def is_tpose(name):
    """A take that only holds the T-pose (the viewer has its own T-pose)."""
    return "tpose" in name.lower().replace("-", "").replace("_", "").replace(" ", "")


def is_loop_name(name):
    """Repeat it?  "Loop" in the name (Quaternius), idles and poses; the rest play once (Death01, Interact)."""
    t = name.lower()
    return "loop" in t or "idle" in t or "pose" in t


def frame_time(m, k, n, step):
    """Time (s) of frame k of n: every `step` seconds (by speed), or spread over the motion (fixed count / capped)."""
    if step:
        return min(k * step, m["seconds"]) if not m["loop"] else k * step
    return m["seconds"] * k / n if m["loop"] else m["seconds"] * k / max(1, n - 1)


def sheet_frames(m, req, cell):
    """Frames of one motion in its sheet: a fixed count ("frames"), or by speed ("fps", the length decides), and then
    no more than fits in 8192 px (long motions)."""
    if req.get("frames"):
        return max(2, int(req["frames"]))
    fps = float(req.get("fps", 10))
    n = max(2, int(round(m["seconds"] * fps)) + (0 if m["loop"] else 1))
    return min(n, 8192 // cell)


def gif_of_row(a, row, c, n, fps, loop):
    """Animated GIF (2x, transparent) of one row of a sheet: the preview of a motion facing the viewer."""
    from PIL import Image
    cells = [a[row * c:(row + 1) * c, k * c:(k + 1) * c] for k in range(n)]
    strip = np.concatenate(cells, 1)                          # one shared palette for every frame
    alpha = strip[..., 3] >= 128
    pal = Image.fromarray(strip[..., :3]).quantize(255, method=Image.Quantize.MEDIANCUT)
    idx = np.array(pal)
    idx[~alpha] = 255
    palette = pal.getpalette()[:255 * 3] + [255, 0, 255]
    frames = []
    for k in range(n):
        f = Image.fromarray(np.ascontiguousarray(idx[:, k * c:(k + 1) * c]), "P").resize((c * 2, c * 2), Image.NEAREST)
        f.putpalette(palette)
        frames.append(f)
    ends = [int(round((k + 1) * 100 / max(fps, 1))) * 10 for k in range(n)]   # GIF delays are in 10 ms steps
    durs = [max(20, e - b) for b, e in zip([0] + ends[:-1], ends)]
    if not loop:
        durs[-1] = 700                                        # one-shot motions: hold the last frame a moment
    b = io.BytesIO()
    frames[0].save(b, "GIF", save_all=True, append_images=frames[1:], duration=durs, loop=0, transparency=255,
                   disposal=2, optimize=False)
    return b.getvalue()


def howto_text(script, extra_pip, keys_en, keys_ja):
    """Step-by-step run instructions (English + Japanese) for people new to Python."""
    pip = "pip install pyxel" + extra_pip
    return "\n".join([
        "HOW TO RUN %s" % script,
        "=" * 40,
        "",
        "1. Install Python 3 (once)",
        "   Windows: https://www.python.org/downloads/  -> run the installer.",
        "            IMPORTANT: tick \"Add python.exe to PATH\" on the first screen.",
        "   Mac:     https://www.python.org/downloads/  -> run the installer.",
        "",
        "2. Open a terminal in THIS folder",
        "   Windows: open this folder in Explorer, click the address bar, type  cmd  and press Enter.",
        "   Mac:     right-click this folder in Finder -> \"New Terminal at Folder\".",
        "",
        "3. Install Pyxel (once)",
        "   " + pip,
        "   (Mac: if \"pip\" is not found, use  pip3  instead.)",
        "",
        "4. Run",
        "   pyxel run " + script,
        "   (the first word is  pyxel , not python.  This also works:  python " + script + " )",
        "",
        "Keys: " + keys_en,
        "Close the window (or press Esc) to quit.",
        "",
        "",
        "%s の動かし方" % script,
        "=" * 40,
        "",
        "1. Python 3 を入れる（最初の 1 回だけ）",
        "   Windows: https://www.python.org/downloads/ からインストーラーを実行します。",
        "            最初の画面で「Add python.exe to PATH」に必ずチェックを入れてください。",
        "   Mac:     https://www.python.org/downloads/ からインストーラーを実行します。",
        "",
        "2. このフォルダでターミナルを開く",
        "   Windows: エクスプローラーでこのフォルダを開き、アドレスバーに  cmd  と入力して Enter。",
        "   Mac:     Finder でこのフォルダを右クリック →「フォルダに新規ターミナル」。",
        "",
        "3. Pyxel を入れる（最初の 1 回だけ）",
        "   " + pip,
        "   （Mac で pip が見つからないときは  pip3  を使います）",
        "",
        "4. 実行する",
        "   pyxel run " + script,
        "   （最初の単語は python ではなく  pyxel  です。 python " + script + "  でも動きます）",
        "",
        "操作: " + keys_ja,
        "終わるときはウィンドウを閉じるか、Esc キーを押します。",
    ]) + "\n"


def readme_text(info):
    src = info["source"]
    lines = [
        "Sprite sheets made with VRM RETRO VIEWER (haruka_apps)",
        "",
        "== No programming needed ==",
        "The PNG and GIF files are ordinary images: open them in any image viewer or game tool",
        "(Unity, Godot, GameMaker, RPG Maker, Aseprite ...).",
        "  *_<motion>.png : one motion per image. Rows = directions, columns = frames.",
        "  *_<motion>.gif : the front row of the sheet as an animation (same frames, same speed).",
        "Cell: %d x %d px.  Rows (top to bottom): %s" % (info["cell"], info["cell"], ", ".join(info["directions"])),
        "Frames: %s." % info.get("timing", ""),
        "The background is transparent.",
        "",
        "== For Pyxel / programmers (optional) ==",
        "sheet.json gives each motion's file, frames, fps and loop, and \"palette\": the 256 colours every PNG uses.",
        "In Pyxel: set pyxel.colors to that palette, then load any PNG with pyxel.Image.from_image (no colour shift).",
        "*_all.png (small exports only) has every motion stacked; \"y\" in sheet.json is where each one starts.",
        "Transparent pixels are magenta (#FF00FF) underneath, so Pyxel can use magenta as the colour key.",
        "pyxel_sample/sample_sprites.py walks your avatar around (needs Python 3 and Pyxel).",
        "Step-by-step guide (Windows / Mac): pyxel_sample/HOW_TO_RUN.txt",
        "  pip install pyxel",
        "  pyxel run pyxel_sample/sample_sprites.py",
        "  Arrows: walk, X: run, Z: attack (if punch / sword / spell was exported).",
        "",
        "== The avatar (from the VRM's own license data) ==",
        "%s by %s (%s)" % (src["title"], src["author"], src["version"]),
    ] + ["  " + s for s in src["license"]] + ["  WARNING: " + s for s in src["warnings"]] + [
        "These sheets are made from that avatar: follow its license when you use or share them.",
        "",
        "== Motions ==",
        "Quaternius Universal Animation Library 1 and 2 (CC0) and KayKit Character Animations (CC0, Kay Lousberg),",
        "converted to VRM Animation by the open-vrma project.",
    ] + ([
        "Your own motions (FBX / VRMA): " + short_list(info["user_motions"]) + ".",
        "  They keep the terms of where you got them.",
    ] if info.get("user_motions") else []) + [
        "",
        "== Made with ==",
        "Pyxel 3 by Takashi Kitao (MIT License) - https://github.com/kitao/pyxel",
        "NumPy (BSD-3-Clause) and Pillow (MIT-CMU); in the browser, Pyodide (MPL-2.0).",
        "",
        "== Disclaimer ==",
        "This tool and its output are provided \"AS IS\", without warranty of any kind.",
        "The author (haruka_apps) accepts no liability for any damage or trouble arising from their use.",
        "You are responsible for following the license of the VRM you use.",
    ]
    return "\n".join(lines) + "\n"


def tmx_wrap(data, kind):
    """Bytes -> a valid 1x1 Tiled map (.tmx) with the bytes as base64 in its "data" property (for Code & Magic)."""
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<map version="1.10" tiledversion="1.10.2" orientation="orthogonal" renderorder="right-down" '
            'width="1" height="1" tilewidth="8" tileheight="8" infinite="0">\n'
            ' <properties>\n'
            '  <property name="data" value="%s"/>\n'
            ' </properties>\n'
            ' <layer id="1" name="data" width="1" height="1">\n'
            '  <data encoding="csv">\n0\n</data>\n'
            ' </layer>\n'
            '</map>\n' % base64.b64encode(data).decode("ascii")).encode()


def readme3d_text(info):
    lines = [
        "3D avatar for Pyxel 3, made with VRM RETRO VIEWER (haruka_apps)",
        "",
        "== Try it ==",
        "Step-by-step guide (Windows / Mac): HOW_TO_RUN.txt",
        "Needs Python 3, Pyxel 3 and numpy:",
        "  pip install pyxel numpy",
        "  pyxel run sample_3d.py",
        "  Arrows: walk, X: run, Z: punch, SPACE: every motion, T: T-pose.",
        "",
        "== Use it in your own game ==",
        "Copy this folder next to your game, then:",
        "  from vrm_figure import Figure",
        "  fig = Figure(\"folder/\")       # model.tmx, motions.tmx, tex_*.png",
        "  stage.add_child(fig)            # a pyxel.cube Node",
        "  fig.play(\"walk\")",
        "  fig.animate(1 / 30)             # every frame",
        "Motions: %d of them (fig.motion_ids() lists the ids; plus \"tpose\")." % info["motions"],
        "  Short ids: idle, walk, run, sprint, jump, punch, sword, spell, hit, death, roll, crouch, talk, dance.",
        "  The rest keep the source names: e.g. sword_regular_a (Quaternius), kk_walking_a (KayKit), kkl_idle_a (KayKit Large).",
        "The avatar uses palette colours 16-255 (255 = magenta colour key). Colours 0-15 stay yours.",
        "vrm_figure.py uses only pyxel, numpy, json, base64 and io (no os / sys).",
        "",
        "== Code & Magic ==",
        "Upload model.tmx, motions.tmx (and my_motions.tmx if there is one), every tex_*.png, vrm_figure.py,",
        "and sample_3d.py renamed to main.py.",
    ] + ([
        "NOTE: my_motions.tmx is %.1f MB, over Code & Magic's 3 MB per file. Add fewer motions of your own" % info["my_motions_mb"],
        "  for Code & Magic (the files work in Pyxel itself).",
    ] if info.get("my_motions_mb", 0) > 3.0 else []) + [
        "(The .tmx files are 1x1 Tiled maps carrying the numpy data as base64, because Code & Magic accepts",
        " .py / .png / .tmx / .glb / .ogg / .pyxres only.)",
        "",
        "== The avatar (from the VRM's own license data) ==",
        "%s by %s (%s), %d triangles" % (info["title"], info["author"], info["version"], info["tris"]),
    ] + ["  " + s for s in info["license"]] + ["  WARNING: " + s for s in info["warnings"]] + [
        "This data is converted from that avatar: follow its license when you use or share it.",
        "",
        "== Motions ==",
        "Quaternius Universal Animation Library 1 and 2 (CC0) and KayKit Character Animations (CC0, Kay Lousberg),",
        "converted to VRM Animation by the open-vrma project.",
    ] + ([
        "Your own motions (FBX / VRMA) are in my_motions.tmx (ids start with my_): " + short_list(info["user_motions"]) + ".",
        "  They keep the terms of where you got them. Mixamo, for example: fine inside your own games,",
        "  but do not share the files or this data on their own (e.g. as an asset pack).",
    ] if info.get("user_motions") else []) + [
        "",
        "== Code ==",
        "vrm_figure.py and sample_3d.py: MIT License (haruka_apps).",
        "They run on Pyxel 3 by Takashi Kitao (MIT License, https://github.com/kitao/pyxel) and NumPy (BSD-3-Clause).",
        "",
        "== Disclaimer ==",
        "This tool and its output are provided \"AS IS\", without warranty of any kind.",
        "The author (haruka_apps) accepts no liability for any damage or trouble arising from their use.",
        "You are responsible for following the license of the VRM you use.",
    ]
    return "\n".join(lines) + "\n"


ARG = next((a for a in sys.argv[1:] if a.lower().endswith((".vrm", ".glb") + PIC_EXT)), None)
App(ARG)
