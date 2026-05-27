"""
Copies modified robosuite asset files from robosuite_patches/ into the active
Python environment's robosuite installation.

Run once after installing requirements.txt:
    python apply_patches.py
"""
import shutil
import sys
from pathlib import Path

PATCHES = [
    "models/assets/grippers/panda_gripper.xml",
    "models/assets/grippers/meshes/panda_gripper/finger_longer.stl",
    "models/assets/objects/round-nut.xml",
]

def main():
    try:
        import robosuite
    except ImportError:
        sys.exit("robosuite is not installed in this environment. Run: pip install -r requirements.txt")

    robo_root = Path(robosuite.__file__).parent
    patch_root = Path(__file__).parent / "robosuite_patches"

    for rel in PATCHES:
        src = patch_root / rel
        dst = robo_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  patched: {rel}")

    print("All patches applied.")

if __name__ == "__main__":
    main()
