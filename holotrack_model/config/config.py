import json
import yaml
import sys
import os

def update_config(result_json_path):

    config_yaml_path = "holotrack_model/config.yaml"
    
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

    config['model']['physical_params']['waveLength'] = res.get('wavelength') * 1e6
    config['model']['physical_params']['step_z'] = res.get('step_z') * 1e6
    config['model']['physical_params']['z_max'] = res.get('number_of_propagation') * (res.get('step_z') * 1e6)

    delta_n = res.get('index_bacterie') - res.get('index_milieu')
    config['model']['physical_params']['phase_shift'] = round(delta_n, 4)


    current_dir = os.getcwd()
    abs_result_dir = os.path.abspath(os.path.dirname(result_json_path))
    folder_name = os.path.basename(abs_result_dir)
    abs_hologram_path = os.path.join(abs_result_dir, "simulated_hologram", "holo0.tiff")
    rel_hologram_path = "./" + os.path.relpath(abs_hologram_path, current_dir)
    rel_save_dir = f"./holotrack_model/Results/{folder_name}"
    rel_log_dir = f"./holotrack_model/Logs/{folder_name}"
    config['data']['root_dir'] = rel_hologram_path
    config['test']['save_dir'] = rel_save_dir
    config['logging']['logdir'] = rel_log_dir

    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config mise à jour avec succès dans {config_yaml_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python config.py <nom_fic_result.json>")
    else:
        update_config(sys.argv[1])