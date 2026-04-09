# Holotrack

## 0. Informations

### Présentation

**Holotrack** est un projet étudiant développé par **Erwan Gouriou** et **Louis Bonnecaze**, étudiants à CentraleSupélec, dans le cadre de leur projet de fin d’études (SDI-Metz 2025-2026).

L’objectif du projet est la **reconstruction et la localisation 3D de bactéries à partir d’hologrammes**.


### Architecture du projet

Le projet repose sur une structure modulaire :

- **Holosolver** : modèle 
- **Simulator** : générateur d’hologrammes synthétiques
- **Visualizer** : outil de visualisation des résultats

### 🔧 Technologies utilisées

- Python 3.12+

---


## 1. Installation du projet

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

---


## 2. Simulateur

Pour lancer une simulation, il est possible d'exécuter la commande suivante en ayant au préalable remplis les fichiers de configurations json (ex : config_bacteria_random.json)

```bash
python3 Simulator/simu\ holo/main_simu_hologram.py Simulator/simu\ holo/configs/config_bacteria_random.json
```

Pour visualiser une simulation

```bash
python3 Simulator/visualizer/visualizer.py
```

Pour aligner les paramètres du modèle avec ceux de la simulation, exécuter la commande suivante avec le nom de la simulation

```bash
python3 holotrack_model/config/config.py 2026_03_11_13_11_42
```
