
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np

# Paths to the images
periodicity_path = "visuals/periodicity_angle_overlay.png"
visual_map_path = "visuals/visuals_map.png"
output_path = "visuals/overlay_visual_periodicity.png"

# Load images as arrays
periodicity_img = iio.imread(periodicity_path)
visual_map_img = iio.imread(visual_map_path)


fig, ax = plt.subplots(figsize=(12, 12))

# Cover any previous titles
height = periodicity_img.shape[0]
width = periodicity_img.shape[1]
white_height = int(0.03 * height)  
white_block = np.ones((white_height, width, 3), dtype=np.uint8) * 255

if periodicity_img.shape[-1] == 4:
    white_block = np.ones((white_height, width, 4), dtype=np.uint8) * 255

periodicity_img_clean = np.vstack([white_block, periodicity_img[white_height:]])
visual_map_img_clean = np.vstack([white_block, visual_map_img[white_height:]])

ax.imshow(periodicity_img_clean)
ax.imshow(visual_map_img_clean, alpha=0.5)  
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
fig.suptitle("Overlay: Periodicity and Stability (Angle Space)", fontsize=18, y=0.985)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"Saved overlay to {output_path}")
