# SPDX-License-Identifier: MIT
"""Make a VRM look retro: shrink its colour textures and reduce them to one shared palette.

Bones, weights, expressions, spring bones, materials and the licence are left as they are.
VRM 0.x and 1.0 are both written back in their own version.
"""
import io
import json
import struct

from PIL import Image

NEAREST = 9728


def read_glb(data):
    if data[:4] != b"glTF":
        raise ValueError("not a GLB / VRM file")
    n = struct.unpack_from("<I", data, 12)[0]
    j = json.loads(data[20:20 + n])
    off = 20 + n
    binc = b""
    if off + 8 <= len(data):
        m = struct.unpack_from("<I", data, off)[0]
        binc = data[off + 8:off + 8 + m]
    return j, binc


def write_glb(j, binc):
    js = json.dumps(j, ensure_ascii=False, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binc += b"\0" * (-len(binc) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binc), 0x004E4942) + binc
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def vrm_version(j):
    ext = j.get("extensions", {})
    if "VRMC_vrm" in ext:
        return "1.0"
    if "VRM" in ext:
        return "0.x"
    return None


def modification_allowed(j):
    """(allowed, text). Unknown or missing is treated as allowed with a note."""
    ext = j.get("extensions", {})
    if "VRMC_vrm" in ext:
        mod = ext["VRMC_vrm"].get("meta", {}).get("modification", "prohibited")
        return mod != "prohibited", f"modification: {mod}"
    if "VRM" in ext:
        lic = ext["VRM"].get("meta", {}).get("licenseName", "")
        nd = lic in ("Redistribution_Prohibited",) or lic.endswith("_ND")
        return not nd, f"licenseName: {lic or '(none)'}"
    return True, "no VRM licence found"


def color_textures(j):
    """Texture indices that carry colour (base colour, MToon shade colour). Normal maps etc. are left alone."""
    tex = set()
    for m in j.get("materials", []):
        bc = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bc:
            tex.add(bc["index"])
        mt = m.get("extensions", {}).get("VRMC_materials_mtoon", {})
        if "shadeMultiplyTexture" in mt:
            tex.add(mt["shadeMultiplyTexture"]["index"])
    for mp in j.get("extensions", {}).get("VRM", {}).get("materialProperties", []):  # VRM 0.x
        for key in ("_MainTex", "_ShadeTexture"):
            if key in mp.get("textureProperties", {}):
                tex.add(mp["textureProperties"][key])
    return sorted(tex)


def image_bytes(j, binc, img_index):
    bv = j["bufferViews"][j["images"][img_index]["bufferView"]]
    o = bv.get("byteOffset", 0)
    return binc[o:o + bv["byteLength"]]


def retro(data, colors=64, max_size=128, dither=False):
    """Return (new_vrm_bytes, report dict)."""
    colors = max(2, min(256, int(colors)))
    j, binc = read_glb(data)
    version = vrm_version(j)
    if version is None:
        raise ValueError("this GLB has no VRM extension")
    ok, lic = modification_allowed(j)
    if not ok:
        raise ValueError("the license of this avatar does not allow modification (%s)" % lic)
    thumb = j.get("extensions", {}).get("VRMC_vrm", {}).get("meta", {}).get("thumbnailImage")
    if thumb is None:
        t = j.get("extensions", {}).get("VRM", {}).get("meta", {}).get("texture")
        thumb = j["textures"][t]["source"] if isinstance(t, int) and 0 <= t < len(j.get("textures", [])) else None

    tex_ids = color_textures(j)
    img_ids = sorted({j["textures"][t]["source"] for t in tex_ids if "source" in j["textures"][t]} - {thumb})

    # 1) shrink every colour image (box filter keeps the average colour of each new pixel)
    small = {}
    for i in img_ids:
        im = Image.open(io.BytesIO(image_bytes(j, binc, i))).convert("RGBA")
        w, h = im.size
        s = min(1.0, max_size / max(w, h))
        small[i] = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX)

    # 2) one palette shared by all images, made from the visible pixels
    import numpy as np
    seen = [np.asarray(im)[..., :3][np.asarray(im)[..., 3] >= 128] for im in small.values()]
    seen = np.concatenate(seen) if seen else np.zeros((0, 3), np.uint8)
    if len(seen) == 0:
        seen = np.zeros((1, 3), np.uint8)
    side = int(np.ceil(np.sqrt(len(seen))))
    flat = np.zeros((side * side, 3), np.uint8)
    flat[:len(seen)] = seen
    flat[len(seen):] = seen[0]                                # fill the square with a colour already there
    strip = Image.fromarray(flat.reshape(side, side, 3))
    pal_img = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    used = sorted({c for _, c in pal_img.getcolors(maxcolors=256 * 256)})

    # 3) map each image to the palette, keep its alpha, write PNG
    new_png = {}
    for i, im in small.items():
        q = im.convert("RGB").quantize(palette=pal_img,
                                       dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE)
        out = q.convert("RGB").convert("RGBA")
        out.putalpha(im.getchannel("A"))
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        new_png[i] = buf.getvalue()

    # 4) rebuild the binary chunk with the new images (other data copied as is)
    img_view = {j["images"][i]["bufferView"]: i for i in new_png}
    new_bin = bytearray()
    for vi, bv in enumerate(j["bufferViews"]):
        new_bin += b"\0" * (-len(new_bin) % 4)
        if vi in img_view:
            chunk = new_png[img_view[vi]]
        else:
            o = bv.get("byteOffset", 0)
            chunk = binc[o:o + bv["byteLength"]]
        bv["byteOffset"] = len(new_bin)
        bv["byteLength"] = len(chunk)
        new_bin += chunk
    j["buffers"][0]["byteLength"] = len(new_bin) + (-len(new_bin) % 4)
    for i in new_png:
        j["images"][i]["mimeType"] = "image/png"

    # 5) crisp pixels in other apps too: a nearest-neighbour sampler for the changed textures
    j.setdefault("samplers", []).append({"magFilter": NEAREST, "minFilter": NEAREST, "wrapS": 10497, "wrapT": 10497})
    near = len(j["samplers"]) - 1
    for t in tex_ids:
        if j["textures"][t].get("source") in new_png:
            j["textures"][t]["sampler"] = near

    out = write_glb(j, bytes(new_bin))
    report = {
        "version": version,
        "modification_allowed": ok,
        "licence": lic,
        "images_changed": len(new_png),
        "images_kept": len(j.get("images", [])) - len(new_png),
        "colors_asked": colors,
        "colors_used": len(used),
        "max_size": max_size,
        "bytes_before": len(data),
        "bytes_after": len(out),
    }
    return out, report


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    data, rep = retro(open(src, "rb").read(), cols, size)
    open(dst, "wb").write(data)
    print(json.dumps(rep, ensure_ascii=False))
