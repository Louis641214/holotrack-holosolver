import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from PIL import Image
import sys
import napari
from scipy import ndimage
import pandas as pd


directory = sys.argv[1]

data_directory = directory + "/obj/volume_3d.npy"
data = np.load(data_directory)
data = np.flip(data, axis=2)
data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
parent_dir = os.path.dirname(directory.rstrip('/'))
csv = os.path.join(parent_dir, "bacteria_0.csv")

def extract_bacteria_positions(volume_3d, threshold=0.5, 
                               pix_size=5.5, magnification=40.0, step_z=0.5):
    """
    Extrait les positions (en voxels et en mètres) de plusieurs bactéries 
    à partir du volume 3D prédit.

    pix_size : micromètre
    step_z : micromètre

    """
    
    # Calcul des tailles de voxels (pour la conversion en mètres)
    vox_size_xy = pix_size / magnification
    vox_size_z = step_z

    # 2. Seuillage pour créer un masque binaire
    mask = volume_3d > threshold
    
    # 3. Labeling : Isoler les bactéries indépendantes
    # structure=np.ones((3,3,3)) permet de connecter les pixels en diagonale (26-connectivité)
    labeled_volume, num_bacteria = ndimage.label(mask, structure=np.ones((3,3,3)))
    
    print(f"🔬 Nombre de bactéries détectées : {num_bacteria}")
    
    if num_bacteria == 0:
        return pd.DataFrame() # Retourne un dataframe vide si rien n'est trouvé

    # 4. Calcul des barycentres pondérés par la densité prédite
    # Le paramètre 'volume_3d' en premier argument permet de faire un barycentre pondéré !
    labels = np.arange(1, num_bacteria + 1)
    centers_of_mass_voxels = ndimage.center_of_mass(volume_3d, labeled_volume, labels)
    
    results = []
    
    for i, (x_vox, y_vox, z_vox) in enumerate(centers_of_mass_voxels):
        # 5. Conversion des voxels en mètres (Physical Space)
        x_m = x_vox * vox_size_xy
        y_m = y_vox * vox_size_xy
        z_m = z_vox * vox_size_z
        
        # Récupération de la densité maximale de cette bactérie (pour info/tri)
        bact_mask = (labeled_volume == (i + 1))
        max_density = np.max(volume_3d[bact_mask])
        
        results.append({
            'bacterium_id': i + 1,
            'x_voxel': x_vox,
            'y_voxel': y_vox,
            'z_voxel': z_vox,
            'x_position_m': x_m,
            'y_position_m': y_m,
            'z_position_m': z_m,
            'max_density': max_density
        })
        
    return pd.DataFrame(results), labeled_volume


predicted_position, labeled_volume = extract_bacteria_positions(
    data, 
    threshold=0.5  # Ajuste ce seuil selon tes observations
)

def extract_bacteria_true_positions(csv) :
    true_position = pd.read_csv(csv)
    idx = np.arange(1, len(true_position)+1)
    true_position.insert(0, "bacterium_id", idx)
    return true_position

true_position = extract_bacteria_true_positions(csv)
print(true_position)
print("\nPositions réelles :")
print(true_position[['bacterium_id', 'x_voxel', 'y_voxel', 'z_voxel']])
print("\nPositions prédites :")
print(predicted_position[['bacterium_id', 'x_voxel', 'y_voxel', 'z_voxel']])
print("\nDelta :")
error_df = np.abs(true_position[['x_voxel', 'y_voxel', 'z_voxel']] - predicted_position[['x_voxel', 'y_voxel', 'z_voxel']])
error_df.insert(0, "bacterium_id", predicted_position["bacterium_id"])
print(error_df)
'''
------------------------------------------------------------
VISUALIZER 1 : 2D Visualization of each layer of holograms
------------------------------------------------------------
'''
"""
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
"""

'''
------------------------------------------------------------
VISUALIZER 2 : 2D Visualization of each layer of the object
------------------------------------------------------------
'''

"""
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
"""

'''
----------------------------------------------
VISUALIZER 3 : 3D Visualization using NAPARI
----------------------------------------------
'''
"""
longueur_fleche = 10 

# 2. On prépare le tableau vide pour Napari (N, 2, 3)
N_bacteries = len(true_position)
vecteurs_napari = np.zeros((N_bacteries, 2, 3))

for i in range(N_bacteries):
    x_c = true_position.iloc[i]['x_voxel']
    y_c = true_position.iloc[i]['y_voxel']
    z_c = true_position.iloc[i]['z_voxel']
    
    # Simon Becker utilise : 
    # angle1 (theta) pour la rotation dans le plan XY
    # angle2 (phi) pour l'inclinaison par rapport à Z
    # (D'après ses lignes 150-153)
    t_rad = np.radians(true_position.iloc[i]['theta_angle'])
    p_rad = np.radians(true_position.iloc[i]['phi_angle'])
    
    # --- CALCUL DIRECT ISSU DU CODE SOURCE ---
    # On suit strictement ses m2_x, m2_y, m2_z (lignes 155-157)
    # Le vecteur directionnel est (m2 - centre)
    dx = longueur_fleche * np.sin(p_rad) * np.cos(t_rad)
    dy = longueur_fleche * np.sin(p_rad) * np.sin(t_rad)
    dz = longueur_fleche * np.cos(p_rad)
    
    # On centre la flèche sur le point vert (True)
    # Origine = Centre - demi_vecteur, Direction = vecteur complet
    vecteurs_napari[i, 0, :] = [x_c - dx/2, y_c - dy/2, z_c - dz/2]
    vecteurs_napari[i, 1, :] = [dx, dy, dz]
# 3. On ajoute le calque des vecteurs à Napari
"""
viewer=napari.view_image(data, rendering="iso", name="Bacteries", blending="translucent_no_depth")
viewer.dims.ndisplay = 3
viewer.axes.visible = True
viewer.axes.colored = True
viewer.axes.dashed = False
viewer.dims.axis_labels = ['X', 'Y', 'Z']
viewer.layers["Bacteries"].bounding_box.visible = True
viewer.add_labels(labeled_volume, name="Bactéries Isolées (Labeling)", blending="translucent_no_depth")
viewer.add_points(np.array(predicted_position.iloc[:, 1:4]), size=5, name="predicted_position", face_color="red", 
                  symbol="cross", blending="translucent_no_depth", features={'id' : predicted_position["bacterium_id"]}, 
                  text={
                        'string': 'ID: {id}',     # On appelle la feature 'id'
                        'size': 10,               # Taille de la police
                        'color': 'white',         # Couleur du texte
                        'translation': [-10, 0, 0] # On décale le texte un peu plus haut en Z pour ne pas cacher la croix
                    })
viewer.add_points(np.array(true_position[['x_voxel', 'y_voxel', 'z_voxel']]), size=5, name="true_position", face_color="green", symbol="cross", blending="translucent_no_depth")
"""
viewer.add_vectors(
    vecteurs_napari,
    edge_width=2,         
    edge_color='yellow',  
    name='Orientation (Ground Truth)',
    blending='translucent'
)
"""

napari.run()



