# 🎯 Point de départ rapide

Bienvenue dans le projet Simu-Bacteria-Holograms!

## 👉 Par où commencer?

### 1️⃣ Pour **simuler des hologrammes** (JSON config) ⭐ PRINCIPAL
```bash
cd simu\ holo/
python main_simu_hologram.py configs/config_bacteria_random.json
```
→ **Utiliser**: [simu holo/README.md](simu%20holo/README.md)

### 2️⃣ Pour **interface graphique**
```bash
cd simu\ bact\ GUI/
python simulation_gui.py
```
→ **Utiliser**: [simu bact GUI/README.md](simu%20bact%20GUI/README.md) (si existe)

### 2️⃣ bis Pour **visualiser les runs 2026** ⭐ NOUVEAU
```bash
cd Simulator/
./run_visualizer.sh
```
→ **Lancer le visualizer** pour explorer tous les dossiers `2026_...`
- Prévisualise: BMP, TIFF, NPY (avec slider Z pour 3D), CSV/TXT/JSON
- Double-clic ou sélection pour ouvrir les fichiers
- Détection automatique de l'axe Z pour volumes 3D

### 3️⃣ Pour **deep learning / IA**
```bash
python test_UNET3D.py
# ou générer des données via simu_holo d'abord
```

## 📁 Structure du projet

```
Simu-Bacteria-Holograms/
├── simu holo/              ← PRINCIPAL: Simulation par configuration JSON
│   ├── README.md
│   ├── configs/            ← Fichiers de configuration
│   ├── docs/               ← Documentation
│   └── examples/           ← Exemples d'utilisation
│
├── simu bact GUI/          ← Interface graphique interactive
│   ├── simulation_gui.py
│   └── ...
│
├── test_UNET3D.py          ← Deep learning (3D U-Net)
├── simu_hologram.py        ← Modules de base
├── propagation.py
└── ... (autres scripts)
```

## 🚀 Démarrage rapide (5 min)

### Installation
```bash
pip install numpy cupy pillow tifffile torch torchmetrics
```

### Première simulation
```bash
cd simu\ holo/

# Copier une configuration
cp configs/config_template.json configs/mon_test.json

# Modifier le nombre d'hologrammes (optionnel)
# Ouvrir configs/mon_test.json avec votre éditeur

# Lancer la simulation
python main_simu_hologram.py configs/mon_test.json

# Résultats dans: ./simu_bacteria/YYYY_MM_DD_HH_MM_SS/
```

## 📚 Documentation

| Document | Contenu |
|----------|---------|
| [simu holo/README.md](simu%20holo/README.md) | Guide principal simulation |
| [simu holo/docs/CONFIG_GUIDE.md](simu%20holo/docs/CONFIG_GUIDE.md) | Référence paramètres |
| [simu holo/docs/CONFIG_SAVE_OPTIONS.md](simu%20holo/docs/CONFIG_SAVE_OPTIONS.md) | Options de sauvegarde |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Organisation du projet |
| [README.md](README.md) | Historique du projet |

## ⚡ Commandes courantes

```bash
# Simulation bactéries aléatoires
python simu\ holo/main_simu_hologram.py simu\ holo/configs/config_bacteria_random.json

# Simulation bactéries prédéfinies
python simu\ holo/main_simu_hologram.py simu\ holo/configs/config_bacteria_list.json

# Simulation sphères
python simu\ holo/main_simu_hologram.py simu\ holo/configs/config_sphere_random.json

# Générer une configuration
python simu\ holo/generate_config.py bacteria_medium simu\ holo/configs/ma_config.json
```

## 🎯 Cas d'usage typiques

### Développement / Testing
```json
## 🎯 Cas d'usage typiques

### Test rapide (1 hologramme)

Créer `simu holo/configs/test_quick.json` :
```json
{
  "mode": "bacteria_random",
  "nb_holo": 1,
  "nb_objects": 10,
  "save_hologram_bmp": true
}
```

Exécuter :
```bash
python "simu holo/main_simu_hologram.py" "simu holo/configs/test_quick.json"
```

### Dataset pour deep learning (1000 hologrammes)

Créer `simu holo/configs/dataset_ml.json` :
```json
{
  "mode": "bacteria_random",
  "nb_holo": 1000,
  "nb_objects": 50,
  "save_npz_data": true,
  "save_hologram_bmp": false
}
```

Exécuter :
```bash
python "simu holo/main_simu_hologram.py" "simu holo/configs/dataset_ml.json"
```

## ❓ FAQ

**Q: Où vont les résultats?**  
R: Par défaut dans `./simu_bacteria/` ou `./simu_sphere/` (horodaté)

**Q: Combien de temps pour une simulation?**  
R: ~45-90 secondes par hologramme (dépend config)

**Q: Puis-je modifier les configurations?**  
R: Oui, éditez les fichiers JSON dans `simu holo/configs/`

**Q: Quel est l'impact des options de sauvegarde?**  
R: Consultez [CONFIG_SAVE_OPTIONS.md](simu%20holo/docs/CONFIG_SAVE_OPTIONS.md)

## 🆘 Besoin d'aide?

1. Consulter [simu holo/README.md](simu%20holo/README.md) pour la documentation principale
2. Consulter [simu holo/docs/](simu%20holo/docs/) pour des guides détaillés
3. Vérifier [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) pour l'organisation
4. Lancer `simu holo/examples/run_examples.sh` pour voir des exemples

---

**Dernière mise à jour**: Février 2026  
**Version**: 2.0 - Architecture modulaire avec `libs/` et `localisation_pipeline/`
