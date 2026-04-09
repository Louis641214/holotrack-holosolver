import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from PIL import Image
import sys
import napari
from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import pandas as pd


directory = sys.argv[1]

data_directory = directory + "/obj/volume_3d.npy"
data = np.load(data_directory)
data = np.flip(data, axis=2)
data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
parent_dir = os.path.dirname(directory.rstrip('/'))
csv = os.path.join(parent_dir, "bacteria_0.csv")

def extract_bacteria_positions(volume_3d, threshold=0.50, 
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
        
    return pd.DataFrame(results), labeled_volume, vox_size_z 


predicted_position, labeled_volume, vox_size_z = extract_bacteria_positions(
    data)

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

# =====================================================================
# MATCHING DES BACTÉRIES
# =====================================================================

# 1. On extrait les coordonnées sous forme de matrices Numpy
coords_true = true_position[['x_voxel', 'y_voxel', 'z_voxel']].values
coords_pred = predicted_position[['x_voxel', 'y_voxel', 'z_voxel']].values

# 2. On calcule la matrice des distances (chaque vrai point vers chaque point prédit)
cost_matrix = distance_matrix(coords_true, coords_pred)

# 3. L'algorithme trouve la meilleure association 1-pour-1
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# row_ind contient les indices des vraies bactéries (CSV)
# col_ind contient les indices des bactéries prédites correspondantes (SciPy)

# 4. On réorganise les DataFrames pour qu'ils soient alignés
matched_true = true_position.iloc[row_ind].copy().reset_index(drop=True)
matched_pred = predicted_position.iloc[col_ind].copy().reset_index(drop=True)

print("\n--- RÉSULTAT DU MATCHING ---")
for i in range(len(matched_true)):
    id_vrai = matched_true.loc[i, 'bacterium_id']
    id_pred = matched_pred.loc[i, 'bacterium_id']
    dist = cost_matrix[row_ind[i], col_ind[i]]
    print(f"Vraie Bactérie {id_vrai} <--> Prédite Bactérie {id_pred} (Distance: {dist:.2f} voxels)")

# =====================================================================
# CALCUL DU DELTA
# =====================================================================

print("\nDelta final (Erreur absolue) :")
error_df = np.abs(matched_true[['x_voxel', 'y_voxel', 'z_voxel']] - matched_pred[['x_voxel', 'y_voxel', 'z_voxel']])

# On ajoute les IDs pour que ce soit clair à la lecture
error_df.insert(0, "matched_true_id", matched_true["bacterium_id"])
error_df.insert(1, "matched_pred_id", matched_pred["bacterium_id"])

print(error_df)

# Pour Napari, on garde les dataframes originaux, mais on pourrait très bien
# utiliser les IDs 'matched_pred_id' si tu veux que les labels collent !

# =====================================================================
# SYNCHRONISATION DES LABELS POUR NAPARI
# =====================================================================

# 1. On crée un dictionnaire de traduction : {SciPy_ID : True_ID}
mapping_dict = dict(zip(matched_pred['bacterium_id'], matched_true['bacterium_id']))

# 2. On applique ce dictionnaire à toutes nos prédictions
# Si une prédiction n'a pas de correspondance (Faux Positif), on affiche "FP" ou son ID de base
predicted_position['display_id'] = predicted_position['bacterium_id'].map(mapping_dict)

# On remplace les NaN (les non-matchés) par un texte clair
predicted_position['display_id'] = predicted_position['display_id'].fillna('Erreur/FP')

# Pour avoir des entiers propres sur les matchés (ex: "1" au lieu de "1.0")
predicted_position['display_id'] = predicted_position['display_id'].apply(
    lambda x: str(int(x)) if isinstance(x, float) and not np.isnan(x) else str(x)
)


'''
----------------------------------------------
VISUALIZER : 3D Visualization using NAPARI
----------------------------------------------
'''

viewer=napari.view_image(data, rendering="iso", name="Bacteries", blending="translucent_no_depth", contrast_limits=[0.0, 1.0])
viewer.dims.ndisplay = 3
viewer.axes.visible = True
viewer.axes.colored = True
viewer.axes.dashed = False
viewer.dims.axis_labels = ['X', 'Y', 'Z']
viewer.layers["Bacteries"].bounding_box.visible = True
viewer.add_labels(labeled_volume, name="Bactéries Isolées (Labeling)", blending="translucent_no_depth")
viewer.add_points(
    np.array(predicted_position[['x_voxel', 'y_voxel', 'z_voxel']]), 
    size=5, 
    name="predicted_position", 
    face_color="red", 
    symbol="cross", 
    blending="translucent_no_depth", 
    # --- CHANGEMENT ICI : On utilise la nouvelle colonne ---
    features={'id' : predicted_position["display_id"]}, 
    # -------------------------------------------------------
    text={
          'string': 'ID: {id}',     
          'size': 10,               
          'color': 'white',         
          'translation': [-10, 0, 0] 
      }
)
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




#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#NOTE: Others visualizers we used during the implementation :

'''
-------------------------------------------------------------------------------------
VISUALIZER 1 : 2D Visualization of each layer of holograms (Reconstructed Holograms)
--------------------------------------------------------------------------------------
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
-----------------------------------------------------------------------
VISUALIZER 2 : 2D Visualization of each layer of the object (Heat Map)
-----------------------------------------------------------------------
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