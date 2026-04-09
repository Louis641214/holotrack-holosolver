# Holotrack

## 0. Informations

### Présentation

**Holotrack** est un projet étudiant développé par **Erwan Gouriou** et **Louis Bonnecaze**, étudiants à CentraleSupélec, dans le cadre de leur projet de fin d’études (SDI-Metz 2025-2026). 

Le projet a été encadré par les participants suivants :
- Simon BECKER
- Jérémy FIX
- Liudmyla KLOCHKO
- Nicolas LOUVET

L’objectif du projet est la **reconstruction et la localisation 3D de bactéries à partir d’hologrammes**.


### Architecture du projet

Le projet repose sur une structure modulaire :

- **holotrack_model** : modèle HoloSolver
- **Simulator** : générateur d’hologrammes synthétiques
- **visualizer** : outil de visualisation des résultats

### Technologies utilisées

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

Avec la commande précédente les données physiques du modèle seront remplies automatiquement dans le fichier `config.yaml`, prêt à l'entrainement.

---
## 3. Entraînement du modèle HoloSolver

Pour lancer un entraînement la commande suivante est à entrer : 
```bash
python3 python -m holotrack_model.src.torchholo.main ./holotrack_model/config/config.yaml train
```

Pour générer les résultats post-entraînement : 
```bash
python -m holotrack_model.src.torchholo.main ./holotrack_model/config/config_ecoli.yaml test
```


> **NOTE :** Dans sa configuration actuelle, `HoloSolver` génère automatiquement les résultats durant son entraînement toutes les *n* epochs (voir `main.py`).

---
## 4. Configuration du Modèle (`config.yaml`)

Le fichier de configuration `.yaml` permet de contrôler l'intégralité des paramètres d'entraînement, du modèle physique et de l'architecture NeRF. Voici le détail des différentes sections :

### Données (`data`)
Définit les chemins d'accès aux données d'entrée.
* **`root_dir`** : Le chemin vers l'hologramme cible qui servira de référence pour l'entraînement.

### Optimisation (`optim` & `nepochs`)
Contient les paramètres liés à la descente de gradient et à l'optimiseur.
* **`algo`** : L'algorithme d'optimisation utilisé (ex: `Adam`).
* **`params`** : Paramètres standards de l'optimiseur.
    * **`lr`** : Le pas d'apprentissage (Learning Rate) principal, appliqué au NeRF.
    * **`eps`** & **`betas`** : Paramètres d'ajustement de l'algorithme Adam.
* **`lr_physics`** : Le pas d'apprentissage spécifique aux paramètres physiques apprenables (`phase_shift` et `incident_light`).
* **`nepochs`** : Le nombre total d'époques pour l'entraînement principal.

### Architecture et Modèle (`model`)
C'est le cœur de la configuration.
* **`class`** : Le nom de la classe du modèle hybride à instancier (ici, `HoloSolver`).

#### Régularisations (`regularization`)
Pénalités ajoutées à la fonction de coût (loss) pour guider l'entraînement.
* **`with_bc`** : Active les *Boundary Conditions* (activé par défaut).
* **`with_sparsity`** : Contraint le modèle à générer du vide là où c'est nécessaire (désactivé par défaut).
    * **`sparsity_weight`** : Le poids de cette régularisation dans la loss totale.
* **`with_tv`** : Active la régularisation de *Total Variation*, particulièrement utile avec le *Hash Encoding* pour forcer la cohésion spatiale des éléments.
    * **`tv_weight`** : Le poids de la régularisation TV dans la loss totale.

#### Paramètres Physiques (`physical_params`)
Paramètres physiques liés à l'hologramme cible étudié. *Note : Ces valeurs sont généralement pré-remplies automatiquement par le générateur.*
* Inclut des variables telles que le déphasage (`phase_shift`), la longueur physique (`physicalLength`), la longueur d'onde (`waveLength`), etc.

#### Paramètres du NeRF (`nerf_params`)
Configuration du réseau de neurones.
* **`model`** : Sélection de l'architecture. Choix possibles : `MorpHoloNet`, `Deep_MorpHoloNet` (version plus profonde/plus de neurones), ou `Hash_Grid` (version obsolète/non maintenue).
* **`gaussian_proj`** : Taille de l'encodage positionnel (par défaut : 64).
* **`gaussian_scale`** : Précision de l'encodeur (par défaut : 10).

#### Optimisation de la Mémoire (`vram_params`)
Essentiel pour faire tourner le modèle sur des machines ayant une VRAM limitée.
* **`activ`** : Si mis sur `true`, active le "chunking" (découpage) des calculs.
* **`chunk_size`** : Taille des blocs traités. Fixée à `15000000` par défaut, ce qui est optimisé pour une carte graphique de 40 Go de VRAM sur un hologramme de 512x512.
* **`checkpoint`** : Active le PyTorch Gradient Checkpointing sur les graphes de calcul pour économiser encore plus de mémoire au prix d'un temps de calcul légèrement allongé.

#### Pré-entraînement (`pre_training`)
Permet de pré-entraîner le modèle en lui fournissant les positions initiales (cibles) connues des bactéries.
* **`activ`** : Active cette phase (par défaut sur `true`).
* **`epochs`** : Nombre d'époques allouées à cette phase de pré-entraînement (ex: 1500).
* **`targets`** : Liste des coordonnées des bactéries (peut être remplie manuellement ou par le générateur).
    * **`x0`**, **`y0`** : Coordonnées en voxels.
    * **`z0`** : Profondeur en micromètres.
    * **`r`** : Rayon en voxels.

### 🧪 Inférence et Tests (`test`)
Paramètres utilisés pour la phase de rendu final et d'évaluation.
* **`save_dir`** : Le dossier où seront générés et sauvegardés les résultats visuels/hologrammes simulés.
* **`weights_path`** : Le chemin exact vers le fichier `.pt` contenant les meilleurs poids du modèle entraîné à charger pour l'inférence.

### 📊 Suivi (`logging`)
* **`logdir`** : Dossier de destination des logs de l'entraînement, utilisable pour visualiser les courbes de loss en temps réel via TensorBoard.

---
## 5. Visualisation des résultats (`visualizer/visualizer.py`)

Pour visualiser les résultats vous devez exéctuer ```visualizer.py``` en local sur votre machine. 

Pour ce faire une installation d'un environemment python en local avec les dépendances nécessaires est recommandée. 

Pour transférer les résultats générés par le modèle depuis un cluster de calcul vers votre système local vous pouvez adapter et utiliser le fichier ```get_res.bash```. 

Et utiliser la commande suivante : 
```bash
./get_res.bash
```

Une fois les résultats téléchargés pour les visualiser en local : 
```bash
python visualizer.py ./Results/weights_1000          
```
---
