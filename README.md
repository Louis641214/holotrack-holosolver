# 2025-2026-holotrack

Version actuelle du sujet : https://plmlatex.math.cnrs.fr/project/68c2f4e290a695063e0c76d5
## 📋 Vue d'ensemble

Ce projet intègre deux modules principaux :

### 🔬 holotrack_model
Module de suivi et modélisation d'hologrammes avec deep learning.

### 🧬 Simulator (Simu-Bacteria-Holograms)
Simulation d'hologrammes de Gabor pour la détection et localisation 3D de bactéries et sphères. Ce module combine des méthodes classiques de traitement d'images holographiques avec des techniques de deep learning (U-Net 3D) pour la segmentation volumétrique.

---

## 🎯 Fonctionnalités principales du Simulator

- **Simulation d'hologrammes** : Génération d'hologrammes avec bactéries ou sphères (configuration JSON)
- **Interface graphique** : GUI interactive pour paramétrer et générer des datasets
- **Pipeline classique** : Localisation 3D par propagation angulaire, focus et CCL3D
- **Deep Learning** : Segmentation 3D avec U-Net pour l'apprentissage supervisé
- **Reconstruction volumétrique** : Méthode du spectre angulaire avec accélération GPU (CuPy/CUDA)

## 🚀 Démarrage rapide

### 1. Simulation d'hologrammes (Recommandé)

Génération d'hologrammes via fichiers de configuration JSON :

```bash
cd Simulator/simu\ holo
python main_simu_hologram.py configs/config_bacteria_random.json
```

**Options disponibles** :
- `config_bacteria_random.json` : Bactéries aléatoires
- `config_bacteria_list.json` : Bactéries à positions prédéfinies
- `config_sphere_random.json` : Sphères aléatoires
- `config_sphere_list.json` : Sphères à positions prédéfinies

### 2. Interface graphique interactive

Génération de datasets pour l'entraînement de réseaux de neurones :

```bash
cd Simulator/simu\ bact\ GUI
python simulation_gui.py
```

Permet de :
- Configurer les paramètres (taille, nombre d'objets, propriétés optiques)
- Générer des lots d'hologrammes
- Choisir les formats de sortie (BMP, TIFF, NPY, NPZ)
- Visualiser les résultats avec `visualizer_gui.py`

### 3. Pipeline de localisation classique

Pipeline éducatif sans IA pour comprendre les principes de reconstruction holographique :

```bash
cd Simulator/localisation_pipeline
python pipeline_holotracker_locate_simple.py
```

**Étapes du pipeline** :
1. **Propagation** : Méthode du spectre angulaire pour reconstruction 3D
2. **Focus** : Calcul du critère de focus (Tenengrad)
3. **Détection** : Seuillage et composantes connexes 3D
4. **Localisation** : Extraction des coordonnées 3D (barycentres)

### 4. Deep Learning (U-Net 3D)

Segmentation volumétrique par réseau de neurones convolutif 3D :

```bash
cd Simulator/deep_learning_segmentation
python train_UNET3D.py  # Entraînement
python test_UNET3D.py   # Test et évaluation
```

## 📁 Structure du projet

```
2025-2026-holotrack/
├── holotrack_model/                # Module principal holotrack
│   ├── src/
│   │   └── torchholo/              # Implementation torch
│   └── ...
│
├── visualizer/                     # Visualisation des résultats
│
├── Simulator/                      # ⭐ Module de simulation
│   ├── simu holo/                  # Simulation par config JSON
│   │   ├── main_simu_hologram.py
│   │   └── configs/
│   │
│   ├── simu bact GUI/              # Interface graphique
│   │   ├── simulation_gui.py
│   │   ├── visualizer_gui.py
│   │   └── processor_simu_bact.py
│   │
│   ├── localisation_pipeline/      # Pipelines de localisation
│   │   ├── pipeline_holotracker_locate_simple.py
│   │   └── main_reconstruction_volume.py
│   │
│   ├── deep_learning_segmentation/ # Deep learning (U-Net 3D)
│   │   ├── train_UNET3D.py
│   │   ├── test_UNET3D.py
│   │   └── model.py
│   │
│   └── libs/                       # 📦 Modules centralisés
│       ├── simu_hologram.py        # Génération hologrammes
│       ├── propagation.py          # Propagation onde
│       ├── traitement_holo.py      # Post-processing
│       ├── typeHolo.py             # Définitions types
│       ├── CCL3D.py                # Composantes connexes 3D
│       └── focus.py                # Critères de focus
│
└── [Documentation]
    ├── README.md                   # Ce fichier
    ├── pyproject.toml              # Configuration du projet
    └── LICENSE
```

## 🔧 Installation et prérequis

### Matériel
- **GPU NVIDIA** avec support CUDA (obligatoire pour CuPy et le Simulator)

### Installation

```bash
# Cloner et se placer dans le répertoire
cd 2025-2026-holotrack

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # sur Linux/Mac
# ou
.venv\Scripts\activate  # sur Windows

# Installer les dépendances
pip install -e .
```

**Note sur CuPy** : Par défaut, `pyproject.toml` utilise `cupy-cuda11x`. Si vous avez CUDA 12.x, changez-le en `cupy-cuda12x`.

## 📚 Documentation

### Modules principaux
- [holotrack_model/README.md](holotrack_model/README.md) - Module principal
- [Simulator/README.md](Simulator/README.md) - Documentation complète du Simulator
- [visualizer/README.md](visualizer/README.md) - Visualisation

### Guides du Simulator
- [Simulator/QUICK_START.md](Simulator/QUICK_START.md) - Guide de démarrage rapide
- [Simulator/PROJECT_STRUCTURE.md](Simulator/PROJECT_STRUCTURE.md) - Organisation détaillée
- [Simulator/simu holo/README.md](Simulator/simu%20holo/README.md) - Documentation simulation JSON
- [Simulator/libs/README.md](Simulator/libs/README.md) - Documentation des modules libs

## 🛠️ Utilisation

### Génération de données d'entraînement (Simulator)

1. Créer une configuration (ou copier un template)
2. Lancer la simulation :
   ```bash
   cd Simulator
   python "simu holo/main_simu_hologram.py" "simu holo/configs/ma_config.json"
   ```
3. Les résultats sont dans `Simulator/simu_bacteria/` ou `Simulator/simu_sphere/`

### Test du pipeline classique

```bash
cd Simulator/localisation_pipeline
python pipeline_holotracker_locate_simple.py
```

Résultats : `result.csv` avec positions (X, Y, Z) des objets détectés

### Entraînement U-Net 3D

1. Générer des données avec `Simulator/simu holo/` (option `save_npz_data`)
2. Configurer `Simulator/deep_learning_segmentation/config_train.json`
3. Lancer :
   ```bash
   cd Simulator/deep_learning_segmentation
   python train_UNET3D.py
   ```

## 📖 Méthodes implémentées (Simulator)

### Propagation
- **Spectre angulaire** : Propagation exacte dans l'espace de Fourier
- **Fresnel** : Approximation paraxiale
- **Rayleigh-Sommerfeld** : Propagation rigoureuse

### Focus
- **Tenengrad** : Gradient de Sobel au carré (recommandé)
- **Variance** : Variance locale
- **Laplacien** : Dérivée seconde

### Détection
- **CCL3D** : Composantes connexes 3D (connectivité 6, 18, 26)
- **Seuillage adaptatif** : Basé sur l'écart-type

### Deep Learning
- **U-Net 3D** : Segmentation volumétrique avec skip connections
- **Patchs 3D** : Traitement par fenêtres glissantes
- **Métriques** : Dice Score, IoU, Precision, Recall

## 📄 License

GNU General Public License v3.0 - Voir [LICENSE](LICENSE)

## 👤 Auteurs

- Simon BECKER - 2024-2025 (Simulator)
- Équipe PFE HoloTrack 2025-2026