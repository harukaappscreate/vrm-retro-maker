"""Build the itch.io HTML5 upload: web/index.html + web/vrm_retro_maker.pyxapp, zipped as vrm_retro_maker_itch.zip.

    python tools/build_web.py
"""
# SPDX-License-Identifier: MIT
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FILES = ("main.py", "vrm_io.py", "vrm_retro.py", "img2vrm.py", "fbx_io.py", "vrma_io.py", "vrm_figure.py", "motions.npz")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "vrm_retro_maker"
        app.mkdir()
        for f in FILES:
            shutil.copy(ROOT / f, app / f)
        subprocess.run(["pyxel", "package", str(app), str(app / "main.py")], cwd=tmp, check=True)
        shutil.copy(Path(tmp) / "vrm_retro_maker.pyxapp", WEB / "vrm_retro_maker.pyxapp")
    out = ROOT / "vrm_retro_maker_itch.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("index.html", "vrm_retro_maker.pyxapp", "sample.vrm"):
            z.write(WEB / f, f)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
