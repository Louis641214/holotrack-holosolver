# Holotrack

### 1. Installation du projet

Exécutez les commandes suivantes :

```bash
# Cloner et entrer dans le dossier
git clone https://gitlab-research.centralesupelec.fr/cei-2025-2026/2025-2026-holotrack.git
cd 2025-2026-holotrack

# Créer et activer votre environnement virtuel
uv venv
source .venv/bin/activate

# Installer les dépendances
uv pip install -e .
```

### 2. Simulateur

Pour lancer une simulation, il est possible d'exécuter la commande suivante en ayant au préalable remplis les fichiers de configurations json (ex : config_bacteria_random.json)

```bash
python3 Simulator/simu\ holo/main_simu_hologram.py Simulator/simu\ holo/configs/config_bacteria_random.json
```

Il est possible de lancer une simulation depuis une interface graphique avec la commande suivante

```bash
python3 Simulator/simu\ bact\ GUI/simulation_gui.py
```

Pour visualiser une simulation

```bash
python3 Simulator/visualizer/visualizer.py
```

Pour aligner les paramètres du modèle avec ceux de la simulation, exécuter la commande suivante

```bash
python3 holotrack_model/config.py Simulator/results/2026_03_04_12_58_27/parameters_simu_bact.json 
```