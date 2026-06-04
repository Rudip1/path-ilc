import os, sys
os.environ.setdefault("MUJOCO_GL", "glx")
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import kuka_plant

plant = kuka_plant.FlexArmPlant()        # builds the arm with a large offscreen buffer
plant.reset(kuka_plant.HOME)
m, d = plant.model, plant.data

W, H = 700, 950
r = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.55]; cam.distance = 2.15
cam.azimuth = 132.0; cam.elevation = -10.0

r.update_scene(d, camera=cam)
rgb = r.render().copy()
r.enable_segmentation_rendering()
r.update_scene(d, camera=cam)
seg = r.render()[:, :, 0].copy()
r.disable_segmentation_rendering()
r.close()

plane_ids = {i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
mask = (seg >= 0)
for pid in plane_ids:
    mask &= (seg != pid)

gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
gray = 255 - (255 - gray) * 0.82
out = np.full((H, W), 255, dtype=np.uint8)
out[mask] = gray[mask].astype(np.uint8)
ys, xs = np.where(mask)
pad = 18
y0, y1 = max(ys.min()-pad,0), min(ys.max()+pad,H)
x0, x1 = max(xs.min()-pad,0), min(xs.max()+pad,W)
Image.fromarray(out[y0:y1, x0:x1]).save("kuka_arm.png")
print(f"wrote kuka_arm.png  crop={x1-x0}x{y1-y0}  arm_px={mask.sum()}")
