import json
import yaml
import sys
import os
import pandas as pd

def update_config(sim_name):
    """
    Met à jour config.yaml à partir du JSON de simulation dans Simulator/results/<sim_name>/.
    """
    result_dir = os.path.join("Simulator", "results", sim_name)
    result_json_path = os.path.join(result_dir, "config_bacteria_random.json")
    
    # On suppose que le fichier CSV s'appelle bacteria_0.csv (ajuste si besoin)
    csv_path = os.path.join(result_dir, "object_positions/bacteria_0.csv")
    config_yaml_path = "holotrack_model/config/config.yaml"

    if not os.path.exists(result_json_path):
        print(f"Erreur : Le fichier de résultat '{result_json_path}' n'existe pas.")
        return

    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier CSV '{csv_path}' n'existe pas.")
        return

    with open(result_json_path, 'r') as f:
        res = json.load(f)

    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Erreur : '{config_yaml_path}' introuvable.")
        return

    # ==========================================
    # 1. Paramètres physiques (conversions vers µm)
    # ==========================================
    step_z_um = res['step_z'] * 1e6
    physical_length_um = round(res['pix_size'] / res['magnification'] * 1e6, 6)

    config['model']['physical_params']['waveLength'] = res['wavelength'] * 1e6
    config['model']['physical_params']['step_z'] = step_z_um
    config['model']['physical_params']['z_max'] = res['z_size'] * step_z_um
    config['model']['physical_params']['physicalLength'] = physical_length_um

    # Calcul de la phase
    dn = res['index_object'] - res['index_medium']
    wavelength_um = res['wavelength'] * 1e6
    thickness_um = (res['thickness_min'] + res['thickness_max']) / 2 * 1e6 
    phase_rad = (2 * 3.14159265 / wavelength_um) * dn * thickness_um
    config['model']['physical_params']['phase_shift'] = round(phase_rad, 4)

    # ==========================================
    # 2. Paramètres de pré-entrainement (Multi-Cibles)
    # ==========================================
    # Nettoyage des anciennes clés uniques si elles existent encore dans le yaml
    for old_key in ['x0', 'y0', 'z0', 'r']:
        config['model']['pre_training'].pop(old_key, None)

    targets = []
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        # X et Y en pixels (directement les voxels)
        x0 = float(row['x_voxel'])
        y0 = float(row['y_voxel'])
        
        # Z en microns (Voxel * step_z)
        z0 = round(float(row['z_voxel']) * step_z_um, 2)
        
        # Calcul du rayon (r) en pixels
        # 1. Epaisseur en mètres -> conversion en microns
        bact_thickness_um = float(row['thickness']) * 1e6
        # 2. Rayon physique (moitié de l'épaisseur)
        radius_um = bact_thickness_um / 2.0
        # 3. Rayon en pixels (division par la taille d'un pixel)
        r_pixels = radius_um / physical_length_um
        
        targets.append({
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "r": round(r_pixels, 2)
        })

    # Mise à jour de la liste des cibles dans le dictionnaire
    config['model']['pre_training']['targets'] = targets

    # ==========================================
    # 3. Chemins relatifs
    # ==========================================
    current_dir = os.getcwd()
    abs_result_dir = os.path.abspath(result_dir)
    abs_hologram_path = os.path.join(abs_result_dir, "simulated_hologram", "holo_0.tiff")
    rel_hologram_path = "./" + os.path.relpath(abs_hologram_path, current_dir)
    config['data']['root_dir'] = rel_hologram_path

    # Sauvegarde du fichier
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config mise à jour avec succès dans {config_yaml_path}")
    print(f"🔬 {len(targets)} bactéries ajoutées au pré-entraînement.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python holotrack_model/config/config.py <nom_simulation>")
        print("  ex: python holotrack_model/config/config.py 2026_03_11_13_11_29")
    else:
        update_config(sys.argv[1])