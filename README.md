# VRM RETRO MAKER

[日本語](README.ja.md)

Two tools in one page:

1. **VRM → retro VRM.** Open a VRM avatar (0.x or 1.0) and get a **retro VRM** back: small textures that share one palette of 16, 32, 64, 128 or 256 colours, with sharp (nearest) pixels.
The result is an ordinary VRM of the same version, so it works in any VRM app (VSeeFace, VRoid Hub, three-vrm, Unity ...).
Check it before saving with 231 CC0 motions, and switch between Before and After.

2. **Picture → voxel VRM.** Open a picture of a character (PNG / JPG, for example one you made with an AI image tool) and get a **voxel VRM**:
the picture becomes pixel art (32–128 px tall, 16–256 colours), each pixel a column of voxels (thicker towards the middle of the body),
with a skeleton you place yourself by dragging the joints. Download it as VRM 1.0 or VRM 0.x.

A sibling of [VRM RETRO VIEWER](https://github.com/harukaappscreate/vrm-retro-viewer) and [VRM Motion Studio](https://github.com/harukaappscreate/vrm-motion-studio).

Your file is never uploaded. Reading and converting happen inside your browser (or your own computer).

## What changes, what stays

| Changed | Kept as is |
|---|---|
| Colour textures (base colour, MToon shade; 0.x `_MainTex` / `_ShadeTexture`): shrunk to the size you choose (64 / 128 / 256 px max, never enlarged), reduced to one shared palette, saved as PNG | Bones, skin weights, meshes, expressions (blend shapes), spring bones, colliders, look-at, first person |
| A nearest-neighbour sampler for those textures | Materials and their settings, normal maps and other non-colour textures, the thumbnail |
| | The license (meta) and the VRM version |

Avatars whose license forbids modification are refused (VRM 1.0 `modification: prohibited`; VRM 0.x `Redistribution_Prohibited` and the `*_ND` licenses).
For the others, follow the original license when you use or share the result.

## Picture → voxel VRM: how it works

1. **Cut out.** A transparent PNG keeps its transparency. Otherwise the background is what touches the border and is close to its usual colour ("Background cut" sets how close).
2. **Pixel art.** The figure is shrunk to the pixel height you choose and reduced to one palette.
3. **Joints.** The tool guesses the neck, crotch, shoulders, elbows, wrists, hips, knees and ankles from the outline. Check them and drag them into place (Mirror moves both sides together). "Arm thickness" sets how wide the arms are cut out (blue circles).
4. **T-pose.** VRM needs the arms straight out. Each arm is cut out of the full-size picture, turned around its shoulder, and what it covered on the body is filled from the body's colours beside it. Then the pixel art is made again.
5. **Voxels.** Every pixel becomes a column of cubes, as thick as its distance to the edge of its part ("Body depth"). The back shows the same colours as the front. Each column moves with one bone (no blending, like block figures).
6. **VRM.** One mesh, one 16 × 16 palette texture (sharp pixels, unlit), 17 humanoid bones, the pixel art as the thumbnail. No facial expressions or spring bones.

**Several views (a turnaround sheet).** A picture with the figure two to four times side by side (front, side, back, other side, all in a T-pose) is split into its views automatically, and each view gets a role (front / back / side facing right / side facing left; the feet tell which way a side view faces). You can change the roles on the page. Then:
- the **side view gives the depth**: a cube stays only where the front and the side view both show the figure (the arms keep the "puff" thickness, because from the side they hide in the body),
- the **back view colours the back**, the side views colour the sides.

- each **colour group** (hair, skin, clothes ...) is shaped on its own: in every row of the side view, a stretch of one colour behind or in front of the body takes its width from the same colour in the back or front view, so a ponytail becomes a ponytail, not a slab as wide as the body,
- at the borders between bones, the cubes also get their inside faces, so bent joints show no holes.

A **bright green background** (#00FF00) is cut out everywhere, also between a ponytail and the head. On a white background, closed-in gaps of the background colour are cut out too ("Cut out closed-in background"; turn it off for white clothes). Figures that touch on the sheet (a fingertip on the next figure's hair) are separated.

Every view is scaled to the same height and lined up on the middle of the body, so small differences between the views (common in AI sheets) are evened out. Tried with one AI sheet (four views, 2576 × 1438): split and roles right, the long hair and cape show from the side and the back.

**License of the new avatar.** The fields start with the strictest choices (only the author may use it, personal non-profit, no redistribution, no modification, credit required). Change them before you download. VRM 0.x has fewer fields: commercial use is allow / disallow, and "no redistribution" becomes `Redistribution_Prohibited`.

**Your picture.** Use only pictures you made or may use. For AI pictures, check the terms of the service you used. No real people without their consent, and no characters that belong to someone else. The download stays off until you tick the box that says so.

Works best with: one character, standing, from the front, arms a little away from the body, on a plain background. Hands on hips, crossed arms, side views and busy backgrounds need more fixing or will not work.

## Run it yourself

You need Python 3 and Pyxel 3 (`pip install pyxel numpy pillow`).

```bash
python main.py                 # then drop a .vrm file on the window
python main.py avatar.vrm      # or open a file directly
```

Keys: **R** make retro (64 colours, 128 px), **TAB** before / after, **S** save the retro VRM next to main.py, SPACE next motion, T T-pose, P screenshot, G GIF, L license.
With a picture: drag the joints, **M** make VRM, **J** guess the joints again, **X** mirror on / off; then **S** save VRM 1.0, **SHIFT+S** save VRM 0.x, **E** back to the joints. (On the desktop the license fields keep their strict defaults; use the browser page to change them.)

Command line only (no window):

```bash
python vrm_retro.py avatar.vrm avatar_retro.vrm 64 128     # colours, max texture size
python img2vrm.py picture.png avatar.vrm 64 32              # pixel height, colours (add --0x for VRM 0.x)
```

**Browser:** `web/index.html` with `web/vrm_retro_maker.pyxapp`. Serve the `web` folder (for example `python -m http.server` inside it) and open it.
To rebuild the `.pyxapp` and the itch.io zip: `python tools/build_web.py`.

## Checked

- Sendagaya Shibu (VRM 1.0, CC0) and three-vrm-girl (VRM 0.x, pixiv): 16 / 64 / 256 colours. The glTF Validator reports the same result as the original file (0 errors).
- Both load and play in VRM RETRO VIEWER.
- Picture → VRM, with test figures drawn by a script (pants, skirt with long hair, arms out): VRM 1.0 and 0.x pass the glTF Validator (0 errors) and load in three-vrm 3.5.5 (all 15 required bones, meta read correctly) and in VRM RETRO VIEWER (walk, punch). Not yet tried with real AI pictures.

## Files

| File | What it is |
|---|---|
| `main.py` | the app (based on VRM RETRO VIEWER) |
| `vrm_retro.py` | the converter: VRM bytes in, retro VRM bytes out (Pillow only) |
| `img2vrm.py` | picture → voxel VRM: cut out, pixel art, joints, T-pose, voxels, VRM 1.0 / 0.x writer (Pillow and NumPy) |
| `vrm_io.py`, `vrm_figure.py`, `fbx_io.py`, `vrma_io.py`, `motions.npz` | from VRM RETRO VIEWER: reading VRM, the preview figure, your own motions, the bundled motions |
| `web/` | `index.html`, `vrm_retro_maker.pyxapp`, `sample.vrm` |
| `tools/build_web.py` | builds the browser version |

## Credits

- Motions: Quaternius Universal Animation Library 1 & 2 and KayKit Character Animations (Kay Lousberg), all CC0, via open-vrma
- `web/sample.vrm`: Sendagaya Shibu, a CC0 sample model of VRoid (VRM data by coati), made smaller for VRM RETRO VIEWER
- [Pyxel](https://github.com/kitao/pyxel) by Takashi Kitao (MIT License); Pyodide (MPL-2.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU)

## License

MIT License (haruka_apps). See LICENSE.

This tool and everything it makes are provided "AS IS", without warranty of any kind.
You are responsible for following the license of the VRM you load.
