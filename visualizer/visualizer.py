import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from PIL import Image
import sys
import napari
from scipy import ndimage

directory = sys.argv[1]

'''
------------------------------------------------------------
VISUALIZER 1 : 2D Visualization of each layer of holograms
------------------------------------------------------------
'''
intensity_dir = directory + '/intensity/'

files = sorted([f for f in os.listdir(intensity_dir) if f.endswith('.tif')], 
               key= lambda f: int(''.join(filter(str.isdigit, f))))

if not files : 
    print("Erreur : Aucun fichier .tif trouvé")
    exit()

first_image = np.array(Image.open(os.path.join(intensity_dir, files[0])))
height, width = first_image.shape
num_layers = len(files)

data_stack = np.zeros((height, width, num_layers))

for i, filename in enumerate(files) : 
    img_path = os.path.join(intensity_dir, filename)
    data_stack[:, :, i] = np.array(Image.open(img_path))


fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15) 

initial_layer = num_layers // 2
img_display = ax.imshow(data_stack[:, :, initial_layer], cmap='gray')
ax.set_title(f"Couche Z: {initial_layer}")
plt.colorbar(img_display)


ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03]) #
slider = Slider(ax_slider, 'Couche', 0, num_layers - 1, valinit=initial_layer, valfmt='%0.0f')

def update(val):
    layer_idx = int(slider.val)
    img_display.set_data(data_stack[:, :, layer_idx])
    ax.set_title(f"Intensité Z: {layer_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


'''
------------------------------------------------------------
VISUALIZER 2 : 2D Visualization of each layer of the object
------------------------------------------------------------
'''
data_directory = directory + "/obj/volume_3d.npy"
data = np.load(data_directory)

num_layers = data.shape[2]


fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15) 

initial_layer = num_layers // 2
img_display = ax.imshow(data[:, :, initial_layer], cmap='magma')
ax.set_title(f"Couche Z: {initial_layer}")
plt.colorbar(img_display)


ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03]) #
slider = Slider(ax_slider, 'Couche', 0, num_layers - 1, valinit=initial_layer, valfmt='%0.0f')

def update(val):
    layer_idx = int(slider.val)
    img_display.set_data(data[:, :, layer_idx])
    ax.set_title(f"Couche Z: {layer_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


'''
----------------------------------------------
VISUALIZER 3 : 3D Visualization using NAPARI
----------------------------------------------
'''

#threshold pour Iso de napari
threshold = 0.5
mask = data > threshold

viewer=napari.view_image(data, rendering="iso", name="E. Coli 3D")
z_center, y_center, x_center = ndimage.center_of_mass(mask)
viewer.add_points(np.array([[z_center, y_center, x_center]]), size=1, name="Position", face_color="red", symbol="cross")
napari.run()