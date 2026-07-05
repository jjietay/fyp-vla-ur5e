import mujoco
import numpy as np
from PIL import Image

model = mujoco.MjModel.from_xml_path('assets/mujoco/ur5e/scene.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
save_path = 'media/test_depth.png'

r = mujoco.Renderer(model, height=480, width=640)

# --- RGB pass ---
r.update_scene(data, camera='fixed_cam')
rgb = r.render()                                  # (480, 640, 3), uint8
Image.fromarray(rgb).save('test_rgb.png')

# --- Depth pass ---
r.enable_depth_rendering()
r.update_scene(data, camera='fixed_cam')          # must re-update after mode switch
depth = r.render()                                # (480, 640), float32, metres

# Normalise depth to 0-255 for visualisation
depth_vis = depth.copy()
mask = depth_vis < 5.0                        # in-scene pixels
d_min = depth_vis[mask].min()
d_max = depth_vis[mask].max()

depth_vis = np.clip((depth_vis - d_min) / (d_max - d_min), 0, 1)
depth_vis = (depth_vis * 255).astype(np.uint8)
Image.fromarray(depth_vis).save(save_path)

print('saved rgb + depth')