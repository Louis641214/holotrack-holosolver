import json
import yaml
import sys
import os

def update_config(sim_name):
    """
    Met à jour config.yaml à partir du JSON de simulation dans Simulator/results/<sim_name>/.
    """
    result_dir = os.path.join("Simulator", "results", sim_name)
    result_json_path = os.path.join(result_dir, "config_bacteria_random.json")
    config_yaml_path = "holotrack_model/config/config.yaml"

    if not os.path.exists(result_json_path):
        print(f"Erreur : Le fichier de résultat '{result_json_path}' n'existe pas.")
        return

    with open(result_json_path, 'r') as f:
        res = json.load(f)

    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Erreur : '{config_yaml_path}' introuvable.")
        return

    # Paramètres physiques (conversions vers µm)
    config['model']['physical_params']['waveLength'] = res['wavelength'] * 1e6
    config['model']['physical_params']['step_z'] = res['step_z'] * 1e6
    config['model']['physical_params']['z_max'] = res['z_size'] * (res['step_z'] * 1e6)
    config['model']['physical_params']['physicalLength'] = round(res['pix_size'] / res['magnification'] * 1e6, 6)

    dn = res['index_object'] - res['index_medium']
    wavelength_um = res['wavelength'] * 1e6
    thickness_um = (res['thickness_min'] + res['thickness_max']) / 2 * 1e6 

    # Calcul de la phase réelle : (2 * pi / lambda) * dn * épaisseur
    phase_rad = (2 * 3.14159265 / wavelength_um) * dn * thickness_um

    config['model']['physical_params']['phase_shift'] = round(phase_rad, 4)

    # Chemins relatifs
    current_dir = os.getcwd()
    abs_result_dir = os.path.abspath(result_dir)
    abs_hologram_path = os.path.join(abs_result_dir, "simulated_hologram", "holo_0.tiff")
    rel_hologram_path = "./" + os.path.relpath(abs_hologram_path, current_dir)
    config['data']['root_dir'] = rel_hologram_path

    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config mise à jour avec succès dans {config_yaml_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python holotrack_model/config/config.py <nom_simulation>")
        print("  ex: python holotrack_model/config/config.py 2026_03_11_13_11_29")
    else:
        update_config(sys.argv[1])